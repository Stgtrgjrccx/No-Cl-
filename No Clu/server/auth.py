"""Authentication for No Clú: password hashing + signed session cookies.

Google OAuth is added in a later step (it needs the public hosted callback
URL). This module covers everything that works without external accounts:
email/phone + password, and issuing/reading a tamper-proof session cookie.
"""

import os
import re

import bcrypt
from itsdangerous import BadSignature, URLSafeSerializer

# Signs session cookies so a user id can't be forged. In production set
# SESSION_SECRET to a long random string; a dev default keeps local runs working.
_SECRET = os.getenv("SESSION_SECRET", "dev-only-change-me-in-production")
_serializer = URLSafeSerializer(_SECRET, salt="noclu-session")

SESSION_COOKIE = "noclu_session"
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


def valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value.strip())) if value else False


def valid_phone(value: str) -> bool:
    return bool(_PHONE_RE.match(value.strip())) if value else False


def password_problem(password: str):
    """Return a user-facing reason the password is unacceptable, or None if OK."""
    if not password or len(password) < 8:
        return "Password must be at least 8 characters."
    return None
