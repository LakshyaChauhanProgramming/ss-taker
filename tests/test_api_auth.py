from fastapi.testclient import TestClient

from ss_taker.app import app

client = TestClient(app)


def test_health_stays_open_with_token(monkeypatch) -> None:
    monkeypatch.setenv("SS_TAKER_TOKEN", "secret")
    assert client.get("/health").status_code == 200


def test_protected_endpoint_rejects_missing_or_wrong_token(monkeypatch) -> None:
    monkeypatch.setenv("SS_TAKER_TOKEN", "secret")
    assert client.post("/pick-ss", json={}).status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_protected_endpoint_accepts_valid_token(monkeypatch) -> None:
    monkeypatch.setenv("SS_TAKER_TOKEN", "secret")
    assert client.get("/metrics", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200
