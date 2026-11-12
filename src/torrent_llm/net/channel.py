"""An encrypted, mutually authenticated channel between two peers.

Every connection in the swarm, direct or relayed, starts with this handshake and
then carries only encrypted frames. What it guarantees:

* **Who.** Each side proves it holds the private key behind its peer id, by
  signing a hash of the handshake so far (which includes both fresh ephemeral
  keys, so a signature cannot be replayed into another handshake).
* **Secrecy, with forward secrecy.** Keys come from a fresh X25519 exchange per
  connection; stealing a device's identity key later does not decrypt old traffic.
* **Integrity.** Every frame is ChaCha20-Poly1305 with a per-direction counter as
  the nonce, so a modified, reordered, replayed or dropped frame is detected.
* **Membership.** In an invite swarm the swarm secret (``psk``) is mixed into every
  key. A device without it cannot decrypt the responder's second message, so it
  learns nothing, not even who it reached.
* **Cost of an identity.** In the public swarm each side checks the other's proof of
  work (``identity.py``).

The shape is SIGMA-style (sign-and-MAC via AEAD), three messages::

    I -> R  "TLLM/1" | e_I
    R -> I  e_R | AEAD(k1, 0, pk_R | pow_R | sig_R("tllm-R" | H(msg1 | e_R)))
    I -> R  AEAD(k2, 0, pk_I | pow_I | sig_I("tllm-I" | H(msg1 | msg2)))

    k1 = HKDF(dh | psk, salt=H(msg1 | e_R),          info="tllm k1")
    k2 = HKDF(dh | psk, salt=H(msg1 | msg2),         info="tllm k2")
    session = HKDF(dh | psk, salt=H(msg1|msg2|msg3), info="tllm session") -> k_IR | k_RI
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import struct
from collections.abc import Callable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from torrent_llm.net.identity import (
    PUBLIC_KEY_BYTES,
    SIGNATURE_BYTES,
    Identity,
    peer_id_of,
    pow_ok,
    verify_signature,
)
from torrent_llm.net.stream import (
    MAX_FRAME_BYTES,
    ByteStream,
    read_frame,
    write_frame,
)

PROTOCOL = b"TLLM/1"
PSK_BYTES = 32
#: The public swarm's "secret": everyone knows it, which is the point.
PUBLIC_PSK = bytes(PSK_BYTES)
HANDSHAKE_TIMEOUT_S = 10.0

_TAG = 16
_KEY = 32
_NONCE_LIMIT = 2**64 - 1
_IDENTITY_BLOB = PUBLIC_KEY_BYTES + 8 + SIGNATURE_BYTES  # pk | pow nonce | signature
_MSG1 = len(PROTOCOL) + 32
_MSG2 = 32 + _IDENTITY_BLOB + _TAG
_MSG3 = _IDENTITY_BLOB + _TAG
_U64 = struct.Struct(">Q")


class HandshakeError(ConnectionError):
    """The other side is not who it should be, not in this swarm, or not speaking our protocol."""


class ChannelError(ConnectionError):
    """A frame failed authentication, or the channel is unusable."""


def _sha(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def _hkdf(secret: bytes, salt: bytes, info: bytes, length: int = _KEY) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(secret)


def _nonce(counter: int) -> bytes:
    return bytes(4) + counter.to_bytes(8, "little")


def _x25519_public(key: X25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _dh(private: X25519PrivateKey, public: bytes) -> bytes:
    try:
        shared = private.exchange(X25519PublicKey.from_public_bytes(public))
    except ValueError as exc:  # small-order point: an all-zero secret
        raise HandshakeError("peer sent an invalid key exchange value") from exc
    if not any(shared):
        raise HandshakeError("peer sent an invalid key exchange value")
    return shared


def _check_psk(psk: bytes | None) -> bytes:
    psk = PUBLIC_PSK if psk is None else psk
    if len(psk) != PSK_BYTES:
        raise ValueError(f"swarm secret must be {PSK_BYTES} bytes")
    return psk


def _identity_blob(identity: Identity, transcript: bytes, label: bytes) -> bytes:
    return identity.public_key + _U64.pack(identity.pow_nonce) + identity.sign(label + transcript)


def _open_identity(
    blob: bytes, transcript: bytes, label: bytes, *, min_pow_bits: int
) -> tuple[bytes, str]:
    public_key = blob[:PUBLIC_KEY_BYTES]
    (nonce,) = _U64.unpack(blob[PUBLIC_KEY_BYTES : PUBLIC_KEY_BYTES + 8])
    signature = blob[PUBLIC_KEY_BYTES + 8 :]
    if not verify_signature(public_key, signature, label + transcript):
        raise HandshakeError("peer's signature over the handshake does not check")
    if not pow_ok(public_key, nonce, min_pow_bits):
        raise HandshakeError(f"peer's identity lacks {min_pow_bits} bits of proof of work")
    return public_key, peer_id_of(public_key)


class SecureChannel:
    """An established channel. ``send`` may be called concurrently; ``recv`` by one reader."""

    def __init__(
        self,
        stream: ByteStream,
        *,
        send_key: bytes,
        recv_key: bytes,
        peer_public_key: bytes,
        initiator: bool,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> None:
        self.stream = stream
        self.peer_public_key = peer_public_key
        self.peer_id = peer_id_of(peer_public_key)
        self.initiator = initiator
        self._send_aead = ChaCha20Poly1305(send_key)
        self._recv_aead = ChaCha20Poly1305(recv_key)
        self._send_counter = 0
        self._recv_counter = 0
        self._send_lock = asyncio.Lock()
        self._max = max_frame_bytes
        self.closed = False

    @property
    def remote(self) -> str:
        return self.stream.remote

    async def send(self, plaintext: bytes) -> None:
        if self.closed:
            raise ChannelError("channel is closed")
        if len(plaintext) + _TAG > self._max:
            raise ChannelError(f"message of {len(plaintext)} bytes is over the frame limit")
        async with self._send_lock:
            if self._send_counter >= _NONCE_LIMIT:
                raise ChannelError("nonce space exhausted; reconnect")
            counter = self._send_counter
            self._send_counter += 1
            length = len(plaintext) + _TAG
            sealed = self._send_aead.encrypt(_nonce(counter), plaintext, length.to_bytes(4, "big"))
            await write_frame(self.stream, sealed)

    async def recv(self, *, on_progress: Callable[[], None] | None = None) -> bytes:
        if self.closed:
            raise ChannelError("channel is closed")
        sealed = await read_frame(self.stream, max_bytes=self._max, on_progress=on_progress)
        if len(sealed) < _TAG or self._recv_counter >= _NONCE_LIMIT:
            raise ChannelError("malformed frame")
        try:
            plaintext = self._recv_aead.decrypt(
                _nonce(self._recv_counter), sealed, len(sealed).to_bytes(4, "big")
            )
        except InvalidTag as exc:
            self.closed = True
            raise ChannelError(f"frame from {self.peer_id} failed authentication") from exc
        self._recv_counter += 1
        return plaintext

    async def close(self) -> None:
        self.closed = True
        await self.stream.close()


async def initiate(
    stream: ByteStream,
    identity: Identity,
    *,
    psk: bytes | None = None,
    expect_peer: str | None = None,
    min_pow_bits: int = 0,
    timeout: float = HANDSHAKE_TIMEOUT_S,
) -> SecureChannel:
    """Open a channel as the side that connected.

    Args:
        psk: The swarm secret; ``None`` for the public swarm.
        expect_peer: Refuse unless the responder's peer id is this one.
        min_pow_bits: Proof of work the responder's identity must carry.
    """
    psk = _check_psk(psk)
    try:
        return await asyncio.wait_for(
            _initiate(stream, identity, psk, expect_peer, min_pow_bits), timeout
        )
    except TimeoutError as exc:
        await stream.close()
        raise HandshakeError(f"handshake with {stream.remote} timed out") from exc
    except (ConnectionError, ValueError):
        await stream.close()
        raise


async def _initiate(
    stream: ByteStream,
    identity: Identity,
    psk: bytes,
    expect_peer: str | None,
    min_pow_bits: int,
) -> SecureChannel:
    ephemeral = X25519PrivateKey.generate()
    msg1 = PROTOCOL + _x25519_public(ephemeral)
    await write_frame(stream, msg1)

    msg2 = await read_frame(stream, max_bytes=_MSG2)
    if len(msg2) != _MSG2:
        raise HandshakeError(f"{stream.remote} is not a Torrent-LLM peer (bad handshake)")
    e_r = msg2[:32]
    secret = _dh(ephemeral, e_r) + psk
    transcript1 = _sha(msg1, e_r)
    try:
        blob = ChaCha20Poly1305(_hkdf(secret, transcript1, b"tllm k1")).decrypt(
            _nonce(0), msg2[32:], None
        )
    except InvalidTag as exc:
        raise HandshakeError(
            f"{stream.remote} is in a different swarm (the swarm secret does not match)"
        ) from exc
    peer_public, peer_id = _open_identity(blob, transcript1, b"tllm-R", min_pow_bits=min_pow_bits)
    if peer_id == identity.peer_id:
        raise HandshakeError("connected to itself")
    if expect_peer is not None and not hmac.compare_digest(peer_id, expect_peer):
        raise HandshakeError(f"expected peer {expect_peer} but {stream.remote} is {peer_id}")

    transcript2 = _sha(msg1, msg2)
    k2 = _hkdf(secret, transcript2, b"tllm k2")
    msg3 = ChaCha20Poly1305(k2).encrypt(
        _nonce(0), _identity_blob(identity, transcript2, b"tllm-I"), None
    )
    await write_frame(stream, msg3)

    keys = _hkdf(secret, _sha(msg1, msg2, msg3), b"tllm session", 2 * _KEY)
    return SecureChannel(
        stream,
        send_key=keys[:_KEY],
        recv_key=keys[_KEY:],
        peer_public_key=peer_public,
        initiator=True,
    )


async def accept(
    stream: ByteStream,
    identity: Identity,
    *,
    psk: bytes | None = None,
    min_pow_bits: int = 0,
    timeout: float = HANDSHAKE_TIMEOUT_S,
) -> SecureChannel:
    """Open a channel as the side that was connected to."""
    psk = _check_psk(psk)
    try:
        return await asyncio.wait_for(_accept(stream, identity, psk, min_pow_bits), timeout)
    except TimeoutError as exc:
        await stream.close()
        raise HandshakeError(f"handshake from {stream.remote} timed out") from exc
    except (ConnectionError, ValueError):
        await stream.close()
        raise


async def _accept(
    stream: ByteStream, identity: Identity, psk: bytes, min_pow_bits: int
) -> SecureChannel:
    msg1 = await read_frame(stream, max_bytes=_MSG1)
    if len(msg1) != _MSG1 or not msg1.startswith(PROTOCOL):
        raise HandshakeError(f"{stream.remote} is not a Torrent-LLM peer (bad hello)")
    e_i = msg1[len(PROTOCOL) :]

    ephemeral = X25519PrivateKey.generate()
    e_r = _x25519_public(ephemeral)
    secret = _dh(ephemeral, e_i) + psk
    transcript1 = _sha(msg1, e_r)
    sealed = ChaCha20Poly1305(_hkdf(secret, transcript1, b"tllm k1")).encrypt(
        _nonce(0), _identity_blob(identity, transcript1, b"tllm-R"), None
    )
    msg2 = e_r + sealed
    await write_frame(stream, msg2)

    msg3 = await read_frame(stream, max_bytes=_MSG3)
    if len(msg3) != _MSG3:
        raise HandshakeError(f"{stream.remote} sent a malformed handshake")
    transcript2 = _sha(msg1, msg2)
    try:
        blob = ChaCha20Poly1305(_hkdf(secret, transcript2, b"tllm k2")).decrypt(
            _nonce(0), msg3, None
        )
    except InvalidTag as exc:
        raise HandshakeError(f"{stream.remote} failed the handshake") from exc
    peer_public, peer_id = _open_identity(blob, transcript2, b"tllm-I", min_pow_bits=min_pow_bits)
    if peer_id == identity.peer_id:
        raise HandshakeError("connected to itself")

    keys = _hkdf(secret, _sha(msg1, msg2, msg3), b"tllm session", 2 * _KEY)
    return SecureChannel(
        stream,
        send_key=keys[_KEY:],
        recv_key=keys[:_KEY],
        peer_public_key=peer_public,
        initiator=False,
    )
