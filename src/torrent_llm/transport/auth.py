"""Keeping strangers off the shard ports.

A shard executes whatever tensors reach it, and gRPC on its own accepts any
caller that can open a socket to the port. On a trusted network that is a
convenience; anywhere else it lets anyone borrow the GPU, feed it hostile
inputs, or read what other users are sending (issue #29).

A pool therefore shares one secret. The server refuses calls that do not carry
it, and the client sends it with every call. This is access control, not
encryption: the token and the activations still cross the wire in the clear, so
over an untrusted link the pool belongs inside a VPN such as Tailscale.
"""

from __future__ import annotations

import hmac

import grpc

#: gRPC metadata keys are lower case.
TOKEN_METADATA = "x-pool-token"


class TokenInterceptor(grpc.ServerInterceptor):
    """Rejects every call that does not carry the pool's token."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._deny = grpc.unary_unary_rpc_method_handler(self._refuse)

    @staticmethod
    def _refuse(request, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "this pool needs its shared token")

    def intercept_service(self, continuation, handler_call_details):
        sent = dict(handler_call_details.invocation_metadata or ()).get(TOKEN_METADATA, "")
        # Constant-time, so response timing does not leak how much matched.
        if hmac.compare_digest(sent.encode(), self._token.encode()):
            return continuation(handler_call_details)
        return self._deny
