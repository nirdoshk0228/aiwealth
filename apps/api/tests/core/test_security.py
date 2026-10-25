"""Password hashing and JWT tokens."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.security import (
    SCRYPT_N,
    InvalidToken,
    create_access_token,
    decode_access_token,
    hash_password,
    jwt_secret,
    verify_password,
)

SETTINGS = Settings(jwt_secret="unit-test-secret-0123456789abcdef", jwt_expire_minutes=10)  # type: ignore[arg-type]


def test_hash_round_trip_and_format() -> None:
    h = hash_password("s3cret-pass")
    scheme, n, r, p, salt, dk = h.split("$")
    assert (scheme, int(n), int(r), int(p)) == ("scrypt", SCRYPT_N, 8, 5)
    assert verify_password("s3cret-pass", h)
    assert not verify_password("s3cret-pasS", h)


def test_same_password_gets_different_salts() -> None:
    assert hash_password("same") != hash_password("same")


@pytest.mark.parametrize("stored", ["", "bcrypt$x", "scrypt$1$2$3$!!$!!", "plain-text-password"])
def test_malformed_hash_never_verifies(stored: str) -> None:
    assert verify_password("anything", stored) is False


def test_token_round_trip_and_claims() -> None:
    now = datetime.now(UTC)
    token, lifetime = create_access_token(42, SETTINGS, now=now)
    assert lifetime == 600
    assert decode_access_token(token, SETTINGS) == 42
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["exp"] - claims["iat"] == 600 and claims["type"] == "access"


def test_expired_token_is_rejected() -> None:
    token, _ = create_access_token(1, SETTINGS, now=datetime.now(UTC) - timedelta(minutes=11))
    with pytest.raises(InvalidToken, match="expired"):
        decode_access_token(token, SETTINGS)


def test_wrong_secret_and_wrong_type_are_rejected() -> None:
    token, _ = create_access_token(1, SETTINGS)
    other = SETTINGS.model_copy(
        update={"jwt_secret": SecretStr("a-different-secret-0123456789abcdef")}
    )
    with pytest.raises(InvalidToken):
        decode_access_token(token, other)
    refresh = jwt.encode(
        {"sub": "1", "iat": 0, "exp": 2**40, "type": "refresh"},
        jwt_secret(SETTINGS),
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken, match="not an access token"):
        decode_access_token(refresh, SETTINGS)


def test_none_algorithm_is_rejected() -> None:
    unsigned = jwt.encode({"sub": "1", "iat": 0, "exp": 2**40, "type": "access"}, None, "none")
    with pytest.raises(InvalidToken):
        decode_access_token(unsigned, SETTINGS)


def test_short_secret_is_refused() -> None:
    short = Settings(jwt_secret="only-26-bytes-long-secret!")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        jwt_secret(short)


def test_production_requires_a_secret() -> None:
    prod = Settings(app_env="production", jwt_secret=None)
    with pytest.raises(RuntimeError, match="JWT_SECRET must be set"):
        jwt_secret(prod)
    dev = Settings(app_env="development", jwt_secret=None)
    assert jwt_secret(dev) == jwt_secret(dev)  # stable within the process
