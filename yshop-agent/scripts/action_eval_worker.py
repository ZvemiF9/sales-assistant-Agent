"""Guard the complete pytest lifetime, including imports and shutdown hooks."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch


_PAIR = ContextVar("owned_stdlib_socketpair", default=False)


@contextmanager
def network_guard(attempts: list, pairs: list):
    connect = socket.socket.connect
    pair = socket.socketpair
    fallback_code = getattr(getattr(socket, "_fallback_socketpair", None), "__code__", None)

    def refuse(*args, **kwargs):
        attempts.append(True)
        raise OSError("evaluation_network_disabled")

    def guarded_connect(sock, address):
        frame = sys._getframe(1)
        listener = frame.f_locals.get("lsock") if frame.f_code is fallback_code else None
        # Windows asyncio uses the standard library's authenticated self-pipe.
        # Only its exact client/listener pair may connect; arbitrary localhost is blocked.
        if (_PAIR.get() and frame.f_code is fallback_code
                and frame.f_locals.get("csock") is sock and isinstance(listener, socket.socket)
                and type(address) is tuple and address == listener.getsockname()[:2]
                and address[0] in {"127.0.0.1", "::1"}):
            pairs.append(True)
            return connect(sock, address)
        return refuse()

    def owned_pair(*args, **kwargs):
        token = _PAIR.set(True)
        try:
            return pair(*args, **kwargs)
        finally:
            _PAIR.reset(token)

    with patch.object(socket.socket, "connect", guarded_connect), \
         patch.object(socket.socket, "connect_ex", refuse), \
         patch.object(socket.socket, "sendto", refuse), \
         patch.object(socket, "getaddrinfo", refuse), \
         patch.object(socket, "socketpair", owned_pair), ExitStack() as platform_patches:
        if sys.platform == "win32":
            from asyncio.windows_events import IocpProactor

            # IOCP calls native ConnectEx/WSAConnect/WSASendTo without socket methods.
            for method in ("connect", "sendto"):
                platform_patches.enter_context(patch.object(IocpProactor, method, refuse))
        yield


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "--network-report" or args[2] != "--":
        return 1
    service = Path(__file__).resolve().parents[1]
    supplied = Path(args[1])
    if (any(p.is_symlink() or p.is_junction() for p in (supplied, *supplied.parents))
            or supplied.name != "network.json" or supplied.parent.parent.resolve() != service / ".tmp"
            or not supplied.parent.name.startswith("day7-action-eval-")):
        return 1
    attempts, pairs = [], []
    code, verified = 1, False
    try:
        with supplied.open("x", encoding="utf-8") as output:
            try:
                with network_guard(attempts, pairs):
                    import pytest
                    code = int(pytest.main(args[3:]))
                    verified = True
            finally:
                json.dump({"verified": verified, "connection_attempts": len(attempts),
                           "local_socketpair_connections": len(pairs)}, output)
    except Exception:
        return 1
    return code if verified and not attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
