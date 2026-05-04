from hmac import compare_digest
from ipaddress import ip_address, ip_network

from fastapi import HTTPException, Request, status


_LOOPBACK_NETWORKS = (
    ip_network("127.0.0.0/8"),
    ip_network("::1/128"),
)
_TEST_CLIENT_HOSTS = {"testclient"}


def request_has_valid_token(request: Request, expected_token: str | None) -> bool:
    if not expected_token:
        return False
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and compare_digest(token, expected_token)


def request_is_loopback(request: Request) -> bool:
    if request.client is None:
        return False
    host = request.client.host
    if host in _TEST_CLIENT_HOSTS:
        return True
    try:
        client_ip = ip_address(host)
    except ValueError:
        return host in {"localhost"}
    return any(client_ip in network for network in _LOOPBACK_NETWORKS)


def require_local_or_token(request: Request, expected_token: str | None) -> None:
    if request_is_loopback(request) or request_has_valid_token(request, expected_token):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="local access or bearer token required",
    )
