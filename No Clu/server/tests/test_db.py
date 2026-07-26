"""DB helper tests against an in-memory SQLite database (no files, no setup)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db as dbmod


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    dbmod.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    yield s
    s.close()


def test_create_and_fetch_user_by_email(session):
    user = dbmod.create_user(session, email="Alice@Example.com", password_hash="x")
    assert user.id is not None
    # email is normalized to lowercase on the way in
    found = dbmod.get_user_by_email(session, "alice@example.com")
    assert found is not None and found.id == user.id


def test_fetch_user_by_phone_and_google_id(session):
    u1 = dbmod.create_user(session, phone="+919876543210", password_hash="x")
    u2 = dbmod.create_user(session, google_id="google-abc", display_name="Bob")
    assert dbmod.get_user_by_phone(session, "+919876543210").id == u1.id
    assert dbmod.get_user_by_google_id(session, "google-abc").id == u2.id


def test_add_and_read_recent_scans_scoped_to_user(session):
    alice = dbmod.create_user(session, email="a@x.com", password_hash="x")
    bob = dbmod.create_user(session, email="b@x.com", password_hash="x")
    dbmod.add_scan(session, alice.id, title="Interstellar", content_type="movie", year=2014)
    dbmod.add_scan(session, alice.id, title="Dune", content_type="movie", year=2021)
    dbmod.add_scan(session, bob.id, title="Oppenheimer", content_type="movie", year=2023)

    alice_scans = dbmod.recent_scans(session, alice.id)
    assert [s.title for s in alice_scans] == ["Dune", "Interstellar"]  # newest first
    bob_scans = dbmod.recent_scans(session, bob.id)
    assert [s.title for s in bob_scans] == ["Oppenheimer"]


def test_recent_scans_respects_limit(session):
    alice = dbmod.create_user(session, email="a@x.com", password_hash="x")
    for i in range(5):
        dbmod.add_scan(session, alice.id, title=f"Movie {i}")
    assert len(dbmod.recent_scans(session, alice.id, limit=3)) == 3
