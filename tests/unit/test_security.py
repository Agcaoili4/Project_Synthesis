import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core.security import require_local_or_token


def _request(host: str, authorization: str | None = None) -> Request:
    headers = []
    if authorization:
        headers.append((b"authorization", authorization.encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "client": (host, 12345),
        }
    )


def test_security_allows_loopback_without_token():
    require_local_or_token(_request("127.0.0.1"), expected_token=None)


def test_security_allows_remote_with_valid_bearer_token():
    require_local_or_token(
        _request("192.0.2.10", "Bearer secret-token"),
        expected_token="secret-token",
    )


def test_security_rejects_remote_without_token():
    with pytest.raises(HTTPException) as exc:
        require_local_or_token(_request("192.0.2.10"), expected_token=None)
    assert exc.value.status_code == 403
