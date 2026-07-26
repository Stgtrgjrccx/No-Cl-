"""Database layer for No Clú accounts + synced scan history.

Uses SQLAlchemy so the same code runs on local SQLite (development, no setup)
and hosted Postgres (production). Set DATABASE_URL to a Postgres URL in
production; leave it unset locally and a `noclu.db` SQLite file is used.
"""

import os
import secrets
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text, create_engine, select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        # Local dev default: a SQLite file next to this module.
        here = os.path.dirname(os.path.abspath(__file__))
        return f"sqlite:///{os.path.join(here, 'noclu.db')}"
    # Render/Neon hand out "postgres://..."; SQLAlchemy needs "postgresql://".
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    google_id = Column(String(255), unique=True, nullable=True)
    email = Column(String(320), unique=True, nullable=True)
    phone = Column(String(32), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)  # bcrypt hash, if password auth
    display_name = Column(String(120), nullable=True)
    # Secret token the iOS Shortcut carries so back-tap scans save to this
    # account without a login session. Bearer secret — only writes scan history.
    scan_token = Column(String(64), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    scans = relationship("Scan", back_populates="user",
                         cascade="all, delete-orphan", order_by="Scan.scanned_at.desc()")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    content_type = Column(String(40), nullable=True)
    year = Column(Integer, nullable=True)
    poster = Column(Text, nullable=True)
    detail = Column(Text, nullable=True)
    scanned_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scans")


_engine = create_engine(
    _database_url(),
    connect_args={"check_same_thread": False} if _database_url().startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(_engine)


# --- User helpers ------------------------------------------------------------
def get_user_by_id(db, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_email(db, email: str) -> Optional[User]:
    return db.scalar(select(User).where(User.email == email.lower().strip()))


def get_user_by_phone(db, phone: str) -> Optional[User]:
    return db.scalar(select(User).where(User.phone == phone.strip()))


def get_user_by_google_id(db, google_id: str) -> Optional[User]:
    return db.scalar(select(User).where(User.google_id == google_id))


def get_user_by_scan_token(db, token: str) -> Optional[User]:
    if not token:
        return None
    return db.scalar(select(User).where(User.scan_token == token))


def create_user(db, *, email=None, phone=None, password_hash=None,
                google_id=None, display_name=None) -> User:
    user = User(
        email=(email.lower().strip() if email else None),
        phone=(phone.strip() if phone else None),
        password_hash=password_hash,
        google_id=google_id,
        display_name=display_name,
        scan_token=secrets.token_urlsafe(24),
    )
    db.add(user)
    db.commit()
    return user


def ensure_scan_token(db, user: User) -> str:
    """Give a user a scan token if they somehow lack one (e.g. pre-existing row)."""
    if not user.scan_token:
        user.scan_token = secrets.token_urlsafe(24)
        db.commit()
    return user.scan_token


# --- Scan helpers ------------------------------------------------------------
def add_scan(db, user_id: int, *, title, content_type=None, year=None,
             poster=None, detail=None) -> Scan:
    scan = Scan(user_id=user_id, title=title, content_type=content_type,
                year=year, poster=poster, detail=detail)
    db.add(scan)
    db.commit()
    return scan


def recent_scans(db, user_id: int, limit: int = 30) -> List[Scan]:
    return list(db.scalars(
        select(Scan).where(Scan.user_id == user_id)
        .order_by(Scan.scanned_at.desc()).limit(limit)
    ))
