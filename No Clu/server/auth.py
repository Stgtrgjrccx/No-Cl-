"""Authentication for No Clú: password hashing, session cookies, OAuth state.

Covers email/phone + password, issuing/reading a tamper-proof session cookie,
and the signed one-time state that protects the Google sign-in round trip.
"""

import os
import re
import secrets

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeSerializer, URLSafeTimedSerializer

# Signs session cookies so a user id can't be forged. In production set
# SESSION_SECRET to a long random string; a dev default keeps local runs working.
_SECRET = os.getenv("SESSION_SECRET", "dev-only-change-me-in-production")
_serializer = URLSafeSerializer(_SECRET, salt="noclu-session")
# Separate salt: an OAuth state must never be usable as a session, or vice versa.
_state_serializer = URLSafeTimedSerializer(_SECRET, salt="noclu-oauth-state")

SESSION_COOKIE = "noclu_session"
OAUTH_STATE_MAX_AGE = 600  # 10 minutes is ample for one sign-in round trip
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[0-9]{7,15}$")


def hash_password(password: str) -> str:
    # bcrypt hard-caps at 72 bytes; truncate deterministically before hashing.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def make_session(user_id: int) -> str:
    return _serializer.dumps({"uid": user_id})


def read_session(token: str):
    """Return the user id from a session cookie, or None if missing/invalid."""
    if not token:
        return None
    try:
        data = _serializer.loads(token)
        return data.get("uid")
    except BadSignature:
        return None


def make_oauth_state() -> str:
    """A signed, single-use, time-limited value for the OAuth `state` parameter.

    Carries a random nonce so two sign-ins never share a state, and is also
    stored in a cookie — the callback requires both to match, which is what
    stops an attacker starting a login in your browser with their own code.
    """
    return _state_serializer.dumps({"n": secrets.token_urlsafe(16)})


def read_oauth_state(token: str, max_age: int = OAUTH_STATE_MAX_AGE):
    """Return the state payload if the token is genuine and fresh, else None."""
    if not token:
        return None
    try:
        return _state_serializer.loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None


def valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value.strip())) if value else False


def valid_phone(value: str) -> bool:
    return bool(_PHONE_RE.match(value.strip())) if value else False


def password_problem(password: str):
    """Return a user-facing reason the password is unacceptable, or None if OK."""
    if not password or len(password) < 8:
        return "Password must be at least 8 characters."
    return None
