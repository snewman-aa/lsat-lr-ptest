import shutil
import duckdb
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_teardown_db():
    """
    Session‐scoped: create (and later remove) tests/data/duckdb_questions.db
    with exactly the tables the server expects.
    """
    data_dir = Path("tests/data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    db_file = data_dir / "duckdb_questions.db"
    con = duckdb.connect(database=str(db_file))

    con.execute("""
      CREATE TABLE questions (
        question_number INTEGER PRIMARY KEY,
        stimulus        TEXT,
        prompt          TEXT,
        A TEXT, B TEXT, C TEXT, D TEXT, E TEXT,
        correct_answer  TEXT,
        explanation     TEXT
      );
    """)
    con.execute("""
      INSERT INTO questions VALUES
        (1, 'dummy stimulus', 'dummy prompt',
         'A1','B1','C1','D1','E1','A','dummy explanation');
    """)

    con.execute("CREATE SEQUENCE IF NOT EXISTS test_id_seq START 1;")
    con.execute("""
      CREATE TABLE tests (
        test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
        prompt      TEXT    NOT NULL,
        created_at  TIMESTAMP DEFAULT now()
      );
    """)

    con.execute("""
      CREATE TABLE test_questions (
        test_id         INTEGER NOT NULL,
        question_number INTEGER NOT NULL
      );
    """)
    con.execute("""
      CREATE TABLE test_responses (
        test_id          INTEGER NOT NULL,
        question_number  INTEGER NOT NULL,
        selected_answer  TEXT    NOT NULL
      );
    """)
    con.close()
    yield
    shutil.rmtree(data_dir)


@pytest.fixture(autouse=True)
def patch_server_deps(monkeypatch):
    """
    Function‐scoped: redirect the app to use temp DB, stub FAISS & Gemini,
    and—critically—clear any cached connection at teardown.
    """
    import app.server as srv

    test_db = Path("tests/data/duckdb_questions.db")
    monkeypatch.setattr(srv, "db_path", test_db)

    monkeypatch.setattr(srv, "load_index",  lambda path: object())
    monkeypatch.setattr(srv, "query_index", lambda idx, q, k, metric: ([0], [0.0]))

    from clients.gemini_client import SampleQuestion
    class DummyGemini:
        def generate_sample_question(self, prompt: str) -> SampleQuestion:
            return SampleQuestion(stimulus="S", prompt="P", explanation="E")
    monkeypatch.setattr(srv, "gemini", DummyGemini())

    yield

    if hasattr(srv.get_db_connection, "cache_clear"):
        srv.get_db_connection.cache_clear()


@pytest.fixture
def client(monkeypatch):
    """
    Patch numpy.load (for the ids array), then import and return TestClient(app).
    """
    import numpy as np
    monkeypatch.setattr(np, "load", lambda path: np.array([1]))

    from app.server import app
    return TestClient(app)
