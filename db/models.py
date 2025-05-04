import io
from pathlib import Path
import duckdb
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from encoder.encoder import Encoder


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def init_question_table():
    """Create (or replace) the 'questions' table in DuckDB from TSV"""
    cfg = load_config()
    root = project_root()

    tsv_path = root / "data" / "lsat_questions.tsv"
    db_path  = root / cfg.db.duckdb.path
    table    = cfg.db.duckdb.question_table

    if not tsv_path.exists():
        raise FileNotFoundError(f"Could not find TSV at {tsv_path}")

    # ensure directory
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(db_path))
    # drop + recreate from TSV
    con.execute(f"DROP TABLE IF EXISTS {table};")
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT
          CAST(question_number AS INTEGER)       AS question_number,
          stimulus,
          prompt,
          A, B, C, D, E,
          correct_answer,
          explanation
        FROM read_csv_auto(
          '{tsv_path.as_posix()}',
          delim='\\t',
          header=True,
          auto_detect=True
        );
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_qnum ON {table}(question_number);")
    con.close()
    print(f"'{table}' table initialized in {db_path}")


def init_test_tables():
    """Create the `tests` + `test_responses` tables with an auto-increment sequence"""
    cfg = load_config()
    duckdb_cfg = cfg["db"]["duckdb"]
    root = project_root()

    db_path = root / duckdb_cfg["path"]
    con     = duckdb.connect(database=str(db_path))

    con.execute("CREATE SEQUENCE IF NOT EXISTS test_id_seq START 1;")

    con.execute("""
        CREATE TABLE IF NOT EXISTS tests (
          test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
          prompt      TEXT NOT NULL,
          created_at  TIMESTAMP DEFAULT current_timestamp
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS test_responses (
          test_id          INTEGER NOT NULL,
          question_number  INTEGER NOT NULL,
          selected_answer  TEXT    NOT NULL,
          FOREIGN KEY(test_id) REFERENCES tests(test_id)
        );
        CREATE TABLE IF NOT EXISTS test_questions (
        test_id         INTEGER NOT NULL,
        question_number INTEGER NOT NULL
        );
    """)
    con.close()
    print("'tests' and 'test_responses' tables initialized")


def init_hdv_table():
    """
    Create (or refresh) the HDV store by projecting every question
    into an HDV and inserting (question_number, hdv_blob)
    """
    cfg      = load_config()
    duckdb_cfg = cfg["db"]["duckdb"]
    enc_cfg  = cfg["encoder"]
    root     = project_root()

    db_path        = root / duckdb_cfg["path"]
    question_table = duckdb_cfg["question_table"]
    hdv_table      = duckdb_cfg["hdv_table"]

    con = duckdb.connect(database=str(db_path))
    con.execute(f"DROP TABLE IF EXISTS {hdv_table};")
    con.execute(f"""
        CREATE TABLE {hdv_table} (
          question_number INTEGER PRIMARY KEY,
          hdv_blob        BLOB
        );
    """)

    rows = con.execute(f"""
        SELECT question_number, stimulus, prompt, explanation
        FROM {question_table};
    """).fetchall()

    questions = [
        {
            "question_number": qn,
            "stimulus":       stim,
            "prompt":         pr,
            "explanation":    expl or ""
        }
        for qn, stim, pr, expl in rows
    ]

    encoder = Encoder(output_dim=enc_cfg["output_dim"],
                      emb_model=enc_cfg["emb_model"])
    roles   = encoder.generate_orthogonal_roles(num_roles=3)

    def worker(qobj):
        qnum = qobj["question_number"]
        hdv  = encoder.generate_question_hdv_from_json(qobj, roles)
        return qnum, hdv

    max_workers = cfg.get("parallel", {}).get("n_workers", None)
    hdv_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for qnum, hdv in pool.map(worker, questions):
            hdv_map[qnum] = hdv

    for qnum, hdv in hdv_map.items():
        buf = io.BytesIO()
        np.save(buf, hdv.astype("float32"), allow_pickle=False)
        blob = buf.getvalue()
        con.execute(
            f"INSERT INTO {hdv_table} (question_number, hdv_blob) VALUES (?, ?);",
            (int(qnum), blob)
        )

    con.close()
    print(f"'{hdv_table}' table initialized with {len(hdv_map)} HDVs")


if __name__ == "__main__":
    init_question_table()
    init_test_tables()
    init_hdv_table()
