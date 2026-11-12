"""Who a device is: an Ed25519 key, the peer id derived from it, and proof of work.

A device's identity is its key pair, nothing else. The peer id is a hash of the
public key, so nobody can claim an id without holding the matching private key:
the secure channel (``channel.py``) makes every peer prove it.

Making an identity is free, which is what a sybil attack exploits (one machine
pretending to be a thousand). In the public swarm every identity must carry a
proof of work: a nonce that makes a hash of the public key start with enough zero
bits. Finding it costs a second or two once, on a device's first start; checking it
costs one hash. Invite swarms set the bar to zero, since the invite already keeps
strangers out.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

#: Bytes of sha256(pubkey) kept in a peer id: 160 bits, as in BitTorrent's DHT.
PEER_ID_BYTES = 20
#: Length of a peer id string (base32 of 20 bytes, no padding).
PEER_ID_LENGTH = 32
PUBLIC_KEY_BYTES = 32
SIGNATURE_BYTES = 64
POW_DOMAIN = b"tllm-pow-v1"
#: Proof of work asked of every identity in the public swarm.
PUBLIC_POW_BITS = 20
_ID_ALPHABET = set("abcdefghijklmnopqrstuvwxyz234567")


def peer_id_of(public_key: bytes) -> str:
    """The peer id for a raw 32-byte Ed25519 public key."""
    if len(public_key) != PUBLIC_KEY_BYTES:
        raise ValueError(f"public key must be {PUBLIC_KEY_BYTES} bytes, got {len(public_key)}")
    digest = hashlib.sha256(public_key).digest()[:PEER_ID_BYTES]
    return base64.b32encode(digest).decode("ascii").lower().rstrip("=")


def is_peer_id(value: object) -> bool:
    """True for a string shaped like a peer id (it says nothing about who holds it)."""
    return isinstance(value, str) and len(value) == PEER_ID_LENGTH and set(value) <= _ID_ALPHABET


def pow_digest(public_key: bytes, nonce: int) -> bytes:
    if not 0 <= nonce < 2**64:
        raise ValueError("proof-of-work nonce must fit in 64 bits")
    return hashlib.sha256(POW_DOMAIN + public_key + nonce.to_bytes(8, "big")).digest()


def leading_zero_bits(digest: bytes) -> int:
    bits = 0
    for byte in digest:
        if byte == 0:
            bits += 8
            continue
        return bits + (8 - byte.bit_length())
    return bits


def pow_ok(public_key: bytes, nonce: int, bits: int) -> bool:
    """Does ``nonce`` prove ``bits`` of work for this key? Zero bits always passes."""
    if bits <= 0:
        return True
    try:
        return leading_zero_bits(pow_digest(public_key, nonce)) >= bits
    except ValueError:
        return False


def mine_pow(public_key: bytes, bits: int, *, start: int = 0) -> int:
    """The first nonce from ``start`` that proves ``bits`` of work (about 2**bits hashes)."""
    if bits <= 0:
        return 0
    if bits > 40:
        raise ValueError("more than 40 bits of proof of work would take days")
    prefix = POW_DOMAIN + public_key
    full_bytes, rest = divmod(bits, 8)
    zeros = bytes(full_bytes)
    limit = 0x100 >> rest  # the next byte must be below this
    nonce = start
    sha = hashlib.sha256
    while True:
        digest = sha(prefix + nonce.to_bytes(8, "big")).digest()
        if digest[:full_bytes] == zeros and (rest == 0 or digest[full_bytes] < limit):
            return nonce
        nonce += 1


def verify_signature(public_key: bytes, signature: bytes, data: bytes) -> bool:
    """True when ``signature`` is ``public_key``'s signature of ``data``."""
    if len(public_key) != PUBLIC_KEY_BYTES or len(signature) != SIGNATURE_BYTES:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, data)
    except (InvalidSignature, ValueError):
        return False
    return True


@dataclass(frozen=True)
class Identity:
    """A device's key pair, with the proof of work found for it."""

    private_key: Ed25519PrivateKey
    pow_nonce: int = 0
    pow_bits: int = 0

    @classmethod
    def generate(cls, pow_bits: int = 0) -> Identity:
        key = Ed25519PrivateKey.generate()
        public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return cls(key, mine_pow(public, pow_bits), max(0, pow_bits))

    @property
    def public_key(self) -> bytes:
        return self.private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    @property
    def peer_id(self) -> str:
        return peer_id_of(self.public_key)

    def sign(self, data: bytes) -> bytes:
        return self.private_key.sign(data)

    def with_pow(self, bits: int) -> Identity:
        """This identity with at least ``bits`` of proof of work (mined if needed)."""
        if pow_ok(self.public_key, self.pow_nonce, bits):
            return self
        return Identity(self.private_key, mine_pow(self.public_key, bits), bits)

    # --- storage -----------------------------------------------------------

    def to_json(self) -> str:
        raw = self.private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        return json.dumps(
            {
                "version": 1,
                "private_key": base64.b64encode(raw).decode("ascii"),
                "pow_nonce": self.pow_nonce,
                "pow_bits": self.pow_bits,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Identity:
        try:
            data = json.loads(text)
            if data.get("version") != 1:
                raise ValueError(f"unknown identity file version {data.get('version')!r}")
            raw = base64.b64decode(data["private_key"], validate=True)
            key = Ed25519PrivateKey.from_private_bytes(raw)
            nonce, bits = int(data.get("pow_nonce", 0)), int(data.get("pow_bits", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"identity file is damaged: {exc}") from exc
        identity = cls(key, nonce, bits)
        if not pow_ok(identity.public_key, nonce, bits):
            raise ValueError("identity file's proof of work does not check")
        return identity

    def save(self, path: Path) -> None:
        """Write atomically, readable by the owner only where the OS supports it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".identity-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover - filesystems without modes
                pass
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> Identity:
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def load_or_create(cls, path: Path, *, pow_bits: int = 0) -> Identity:
        """The identity stored at ``path``, made (and saved) the first time.

        Raising the proof-of-work bar later mines more work for the same key, so
        a device keeps its peer id when it moves to the public swarm.
        """
        if path.exists():
            identity = cls.load(path)
            upgraded = identity.with_pow(pow_bits)
            if upgraded is not identity:
                upgraded.save(path)
            return upgraded
        identity = cls.generate(pow_bits)
        identity.save(path)
        return identity
