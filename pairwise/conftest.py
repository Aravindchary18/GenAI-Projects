import os, tempfile, pytest
from app import db
@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("ENABLE_REAL_DATA", "true")
    db.init_db()
    yield
