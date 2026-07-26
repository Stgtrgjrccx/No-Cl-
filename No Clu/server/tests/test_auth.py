import auth


def test_hash_and_verify_password_roundtrip():
    h = auth.hash_password("hunter2pass")
    assert h != "hunter2pass"          # never stored in the clear
    assert auth.verify_password("hunter2pass", h) is True
    assert auth.verify_password("wrongpass", h) is False


def test_verify_password_handles_empty_hash():
    assert auth.verify_password("anything", "") is False
    assert auth.verify_password("anything", None) is False


def test_session_roundtrip_returns_user_id():
    token = auth.make_session(42)
    assert auth.read_session(token) == 42


def test_read_session_rejects_tampered_or_missing_token():
    assert auth.read_session("") is None
    assert auth.read_session(None) is None
    token = auth.make_session(7)
    assert auth.read_session(token + "x") is None   # tampered -> rejected


def test_email_and_phone_validation():
    assert auth.valid_email("a@b.com") is True
    assert auth.valid_email("nope") is False
    assert auth.valid_phone("+919876543210") is True
    assert auth.valid_phone("12ab") is False


def test_password_problem_enforces_minimum_length():
    assert auth.password_problem("short") is not None
    assert auth.password_problem("longenough") is None
