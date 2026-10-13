import pytest
from fastapi.testclient import TestClient

from app import __version__


@pytest.mark.parametrize("path", ["/health", "/api/v1/health"])
def test_health_returns_ok(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert set(body) == {"status", "app", "version", "env"}
