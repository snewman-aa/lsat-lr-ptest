import shutil
import duckdb
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import app.server as srv
from app.models import SampleQuestion


@pytest.fixture(scope="session", autouse=True)
def setup_teardown_db():
    """
    Session-scoped: create (and later remove) tests/data/duckdb_questions.db
    with exactly the tables the server expects.
    """
    data_dir = project_root / "tests" / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    db_file = data_dir / "duckdb_questions.db"
    con = duckdb.connect(database=str(db_file))

    # main questions table
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
        (1,    'dummy stimulus 1', 'dummy prompt 1', 'A','B','C','D','E', 'A', 'dummy explanation 1'),
        (1001, 'dummy stimulus 1001', 'dummy prompt 1001', 'A','B','C','D','E', 'B', 'dummy explanation 1001'),
        (2002, 'dummy stimulus 2002', 'dummy prompt 2002', 'A','B','C','D','E', 'C', 'dummy explanation 2002');
    """)

    # test-related tables
    con.execute("CREATE SEQUENCE IF NOT EXISTS test_id_seq START 1;")
    con.execute("""
      CREATE TABLE IF NOT EXISTS tests (
        test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
        prompt      TEXT NOT NULL,
        created_at  TIMESTAMP WITH TIME ZONE DEFAULT now() -- Retained TIMESTAMPTZ
      );
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS test_questions (
          test_id         INTEGER NOT NULL,
          question_number INTEGER NOT NULL,
          PRIMARY KEY (test_id, question_number),
          FOREIGN KEY(test_id) REFERENCES tests(test_id),
          FOREIGN KEY(question_number) REFERENCES questions(question_number)
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS test_responses (
          test_id          INTEGER NOT NULL,
          question_number  INTEGER NOT NULL,
          selected_answer  TEXT NOT NULL,
          PRIMARY KEY (test_id, question_number),
          FOREIGN KEY(test_id, question_number) REFERENCES test_questions(test_id, question_number)
        );
    """)
    con.close()
    print(f"\nTest database created at: {db_file}")
    yield
    print(f"\nRemoving test database: {db_file}")
    shutil.rmtree(data_dir)


@pytest.fixture(autouse=True)
def patch_server_deps(monkeypatch):
    """
    Function-scoped: redirect the app to use the test DB, stub FAISS & Gemini,
    and critically clear any cached DB connection before and after each test.
    """
    if hasattr(srv.get_db_connection, "cache_clear"):
        srv.get_db_connection.cache_clear()

    test_db_path = project_root / "tests" / "data" / "duckdb_questions.db"
    monkeypatch.setattr(srv, "DB_PATH", test_db_path)

    class MockFaissIndexPlaceholder:
        pass
    monkeypatch.setattr(srv, "load_index", lambda path: MockFaissIndexPlaceholder())
    monkeypatch.setattr(srv, "query_index", lambda idx, q_vec, k, metric: ([0], [0.9]))

    mocked_faiss_to_actual_qids_list = [1001, 2002, 1]
    monkeypatch.setattr(srv, "question_ids_in_index", mocked_faiss_to_actual_qids_list)

    class DummyGemini:
        def generate_sample_question(self, prompt: str) -> SampleQuestion:
            return SampleQuestion(
                stimulus=f"Mocked stimulus for request: '{prompt}'",
                prompt="This is a mocked question prompt.",
                explanation="This is a mocked explanation for the sample question."
            )
    monkeypatch.setattr(srv, "gemini_client", DummyGemini())

    yield

    if hasattr(srv.get_db_connection, "cache_clear"):
        srv.get_db_connection.cache_clear()


@pytest.fixture
def client(monkeypatch):
    """
    Patch numpy.load (for the ids array loaded in server.py),
    then import and return TestClient(app).
    """
    from app.server import app
    return TestClient(app)