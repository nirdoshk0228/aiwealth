"""JWT auth and saved-portfolio CRUD on SQLite in memory."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import create_access_token
from app.db import session as db_session
from app.main import create_app

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
ME = "/api/v1/auth/me"
PORTFOLIOS = "/api/v1/portfolios"
CREDS = {"email": "Investor@Example.com ", "password": "correct horse battery"}


def login(api: TestClient, creds: dict[str, str] = CREDS) -> dict[str, str]:
    api.post(REGISTER, json=creds)
    token = api.post(LOGIN, json=creds).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me(api: TestClient) -> None:
    r = api.post(REGISTER, json=CREDS)
    assert r.status_code == 201
    assert r.json()["email"] == "investor@example.com"  # normalised
    assert "password" not in r.text and "hash" not in r.text

    tok = api.post(LOGIN, json=CREDS)
    assert tok.status_code == 200
    assert tok.json()["token_type"] == "bearer" and tok.json()["expires_in"] == 30 * 60
    me = api.get(ME, headers={"Authorization": f"Bearer {tok.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "investor@example.com"


def test_duplicate_email_is_409(api: TestClient) -> None:
    api.post(REGISTER, json=CREDS)
    other_case = {**CREDS, "email": "INVESTOR@example.com"}
    assert api.post(REGISTER, json=other_case).status_code == 409


def test_wrong_password_and_unknown_email_are_the_same_401(api: TestClient) -> None:
    api.post(REGISTER, json=CREDS)
    wrong = api.post(LOGIN, json={**CREDS, "password": "wrong password!"})
    unknown = api.post(LOGIN, json={"email": "nobody@example.com", "password": "whatever123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_short_password_and_bad_email_are_422(api: TestClient) -> None:
    assert api.post(REGISTER, json={**CREDS, "password": "short"}).status_code == 422
    assert api.post(REGISTER, json={**CREDS, "email": "not-an-email"}).status_code == 422


def test_protected_routes_need_a_valid_token(api: TestClient, test_settings: Settings) -> None:
    assert api.get(ME).status_code == 401
    assert api.get(ME, headers={"Authorization": "Bearer garbage"}).status_code == 401
    expired, _ = create_access_token(
        1, test_settings, now=datetime.now(UTC) - timedelta(minutes=31)
    )
    assert api.get(ME, headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    other_secret = test_settings.model_copy(update={"jwt_secret": None})
    forged, _ = create_access_token(1, other_secret)  # signed with a different secret
    assert api.get(ME, headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert api.get(PORTFOLIOS).status_code == 401


def test_portfolio_crud(api: TestClient, fund_mix: dict[str, object]) -> None:
    h = login(api)
    created = api.post(PORTFOLIOS, json={"name": "Core", "portfolio": fund_mix}, headers=h)
    assert created.status_code == 201
    pid = created.json()["id"]
    assert created.json()["portfolio"] == fund_mix

    assert [p["id"] for p in api.get(PORTFOLIOS, headers=h).json()] == [pid]
    assert api.get(f"{PORTFOLIOS}/{pid}", headers=h).json()["name"] == "Core"

    new = {
        "portfolio_value": 250000.0,
        "holdings": [
            {"asset_id": "equity:TCS", "weight": 0.5},
            {"asset_id": "mutual_fund:122639", "weight": 0.5},  # also in the old version
        ],
    }
    upd = api.put(f"{PORTFOLIOS}/{pid}", json={"name": "Renamed", "portfolio": new}, headers=h)
    assert upd.status_code == 200
    assert upd.json()["portfolio"] == new and upd.json()["name"] == "Renamed"

    assert api.delete(f"{PORTFOLIOS}/{pid}", headers=h).status_code == 204
    assert api.get(f"{PORTFOLIOS}/{pid}", headers=h).status_code == 404


def test_users_cannot_see_each_others_portfolios(
    api: TestClient, fund_mix: dict[str, object]
) -> None:
    alice = login(api, {"email": "alice@example.com", "password": "alice-password"})
    bob = login(api, {"email": "bob@example.com", "password": "bob-password1"})
    pid = api.post(PORTFOLIOS, json={"name": "A", "portfolio": fund_mix}, headers=alice).json()[
        "id"
    ]
    assert api.get(f"{PORTFOLIOS}/{pid}", headers=bob).status_code == 404
    assert api.delete(f"{PORTFOLIOS}/{pid}", headers=bob).status_code == 404
    assert api.get(PORTFOLIOS, headers=bob).json() == []
    assert api.get(f"{PORTFOLIOS}/{pid}", headers=alice).status_code == 200


def test_saved_portfolio_must_use_universe_assets(api: TestClient) -> None:
    h = login(api)
    body = {
        "name": "Bad",
        "portfolio": {"holdings": [{"asset_id": "equity:NOTREAL", "weight": 1.0}]},
    }
    r = api.post(PORTFOLIOS, json=body, headers=h)
    assert r.status_code == 422 and "equity:NOTREAL" in r.json()["detail"]


def test_database_not_configured_is_503(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_db = test_settings.model_copy(update={"database_url": None})
    monkeypatch.setattr(db_session, "get_settings", lambda: no_db)
    db_session.get_engine.cache_clear()
    db_session.get_session_factory.cache_clear()
    try:
        with TestClient(create_app(no_db)) as c:
            assert c.post(REGISTER, json=CREDS).status_code == 503
    finally:
        db_session.get_engine.cache_clear()
        db_session.get_session_factory.cache_clear()
