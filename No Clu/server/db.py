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
    Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect,
    select, text,
)
from sqlalchemy.exc import OperationalError, ProgrammingError
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
    season = Column(Integer, nullable=True)
    episode = Column(Integer, nullable=True)
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


def ensure_scan_columns(engine=None) -> None:
    """Add nullable season/episode columns to an existing `scans` table.

    SQLAlchemy's create_all() creates missing TABLES but never missing COLUMNS,
    so a table created before these columns existed needs an explicit ALTER.
    Adding a nullable column is additive: no rewrite, no data loss, no downtime.
    Safe to call on every startup.

    The inspector pre-check is a fast path, not a lock: under multiple
    concurrent workers (e.g. `uvicorn --workers N` against Postgres), two
    processes can both see a column missing and both attempt the ALTER. The
    loser gets a "column already exists" / "duplicate column" error even
    though the end state -- the column existing -- is exactly what we want,
    so that specific error is swallowed. Any other error (e.g. permissions)
    still surfaces. Each column gets its own `engine.begin()` so one column's
    failure can't poison the other's transaction.
    """
    engine = engine or _engine
    inspector = inspect(engine)
    if "scans" not in inspector.get_table_names():
        return  # create_all() will build it complete
    existing = {col["name"] for col in inspector.get_columns("scans")}
    for name in ("season", "episode"):
        if name in existing:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE scans ADD COLUMN {name} INTEGER"))
        except (OperationalError, ProgrammingError) as exc:
            message = str(exc).lower()
            if "already exists" in message or "duplicate column" in message:
                # Lost a race with a concurrent worker adding the same
                # column. The column exists either way -- nothing to do.
                continue
            raise


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(_engine)
    ensure_scan_columns(_engine)


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


def link_or_create_google_user(db, *, google_id: str, email: Optional[str] = None,
                               display_name: Optional[str] = None) -> "User":
    """Resolve a Google sign-in to exactly one account.

    Order matters. `google_id` is Google's stable identifier, so it wins even if
    the person later changes their Google email address. Only if that misses do
    we match on email — which links Google onto an account that already exists
    from a password sign-up, instead of silently creating a second account and
    appearing to lose the user's entire scan history.
    """
    user = get_user_by_google_id(db, google_id)
    if user:
        return user

    if email:
        user = get_user_by_email(db, email)
        if user:
            user.google_id = google_id
            if display_name and not user.display_name:
                user.display_name = display_name
            db.commit()
            return user

    return create_user(db, email=email, google_id=google_id, display_name=display_name)


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
             poster=None, detail=None, season=None, episode=None) -> Scan:
    scan = Scan(user_id=user_id, title=title, content_type=content_type,
                year=year, poster=poster, detail=detail,
                season=season, episode=episode)
    db.add(scan)
    db.commit()
    return scan


def get_scan_for_user(db, scan_id: int, user_id: int) -> Optional[Scan]:
    """One scan, but only if it belongs to this user.

    Scoping the query by user_id (rather than fetching then comparing) means a
    caller cannot accidentally leak another account's scan by forgetting a check.
    """
    return db.scalar(
        select(Scan).where(Scan.id == scan_id, Scan.user_id == user_id)
    )


def set_scan_poster(db, scan_id: int, user_id: int, poster: str) -> bool:
    """Attach cover art to a scan that was stored without any. Returns True if saved.

    Scans saved before a poster source could find them keep a NULL poster
    forever, so opening one is the natural moment to fill it in. Scoped by
    user_id for the same reason get_scan_for_user is: the query itself, not a
    caller's memory, is what stops one account writing to another's row.
    """
    scan = get_scan_for_user(db, scan_id, user_id)
    if scan is None or scan.poster:
        return False
    scan.poster = poster
    db.commit()
    return True


def recent_scans(db, user_id: int, limit: int = 30) -> List[Scan]:
    return list(db.scalars(
        select(Scan).where(Scan.user_id == user_id)
        .order_by(Scan.scanned_at.desc()).limit(limit)
    ))
