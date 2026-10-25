"""Password hashing (stdlib scrypt) and JWT access tokens (PyJWT, HS256).

Passwords: scrypt with a random 16-byte salt per password and OWASP's listed parameter set
N = 2^14, r = 8, p = 5 (16 MiB of memory per hash, light enough for a small free-tier
instance). The stored string carries its parameters, `scrypt$N$r$p$salt$hash` (base64), so
they can be raised later without breaking existing hashes. Comparison is constant-time.

Tokens: HS256 JWTs with `sub` (user id), `iat`, `exp` and `type = "access"`; all are required
on decode. The secret comes from JWT_SECRET (at least 32 bytes, RFC 7518). Production refuses
to start without one; in development a random per-process secret is used (tokens die with the
process) and a warning is logged once.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt

from app.core.config import Settings

logger = logging.getLogger(__name__)

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 5
SCRYPT_DKLEN = 32
SALT_BYTES = 16
_MAXMEM = 64 * 1024 * 1024  # above 128 * r * N = 16 MiB
JWT_ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32  # RFC 7518 section 3.2: an HS256 key must be at least 256 bits


class InvalidToken(Exception):
    """Missing, malformed, expired or wrongly signed token."""


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=dklen
    )


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    dk = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_DKLEN)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, dk_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = _scrypt(password, salt, int(n), int(r), int(p), len(expected))
    except ValueError:
        return False
    return hmac.compare_digest(dk, expected)


@lru_cache
def _dev_secret() -> str:
    logger.warning(
        "JWT_SECRET is not set: using a random per-process secret (development only); "
        "tokens become invalid when the API restarts"
    )
    return secrets.token_urlsafe(48)


def jwt_secret(settings: Settings) -> str:
    if settings.jwt_secret is not None:
        secret = settings.jwt_secret.get_secret_value()
        if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
            raise RuntimeError(f"JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes")
        return secret
    if settings.app_env == "production":
        raise RuntimeError("JWT_SECRET must be set in production")
    return _dev_secret()


def create_access_token(
    user_id: int, settings: Settings, *, now: datetime | None = None
) -> tuple[str, int]:
    """Return (token, lifetime in seconds)."""
    issued = now or datetime.now(UTC)
    lifetime = timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "iat": int(issued.timestamp()),
        "exp": int((issued + lifetime).timestamp()),
        "type": "access",
    }
    token = jwt.encode(payload, jwt_secret(settings), algorithm=JWT_ALGORITHM)
    return token, int(lifetime.total_seconds())


def decode_access_token(token: str, settings: Settings) -> int:
    """The user id in a valid access token; raises InvalidToken otherwise."""
    try:
        payload = jwt.decode(
            token,
            jwt_secret(settings),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )
        if payload.get("type") != "access":
            raise InvalidToken("not an access token")
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidToken(str(exc)) from exc
