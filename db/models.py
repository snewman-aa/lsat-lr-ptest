import io
from pathlib import Path
import duckdb
import numpy as np

from encoder.encoder import Encoder


def init_question_table(tsv_path: Path, db_path: Path, question_table: str):
    """(Re)create the 'questions' table in DuckDB from TSV."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Could not find TSV at {tsv_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(db_path))
    con.execute(f"DROP TABLE IF EXISTS {question_table};")
    con.execute(f"""
        CREATE TABLE {question_table} AS
        SELECT
          CAST(question_number AS INTEGER) AS question_number,
          stimulus, prompt, A, B, C, D, E,
          correct_answer, explanation
        FROM read_csv_auto(
          '{tsv_path.as_posix()}',
          delim='\\t', header=True, auto_detect=True
        );
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_qnum ON {question_table}(question_number);")
    con.close()


def init_test_tables(db_path: Path):
    """Create the tests + test_questions + test_responses tables (with auto-increment)."""
    con = duckdb.connect(database=str(db_path))
    con.execute("CREATE SEQUENCE IF NOT EXISTS test_id_seq START 1;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS tests (
          test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
          prompt      TEXT NOT NULL,
          created_at  TIMESTAMP DEFAULT now()
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS test_questions (
          test_id         INTEGER NOT NULL,
          question_number INTEGER NOT NULL
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS test_responses (
          test_id          INTEGER NOT NULL,
          question_number  INTEGER NOT NULL,
          selected_answer  TEXT NOT NULL,
          FOREIGN KEY(test_id) REFERENCES tests(test_id)
        );
    """)
    con.close()


def init_hdv_table(
    db_path: Path,
    question_table: str,
    hdv_table: str,
    encoder: Encoder | None,
    output_dim: int,
    emb_model: str,
) -> None:
    """
    Create or refresh the HDV store by projecting every question
    into an HDV and inserting (question_number, hdv_blob).
    This version runs entirely in the main thread to avoid native‐library
    threading issues.
    :param db_path: Path to the DuckDB file.
    :param question_table: Name of the questions table.
    :param hdv_table: Name of the table to store HDV blobs.
    :param encoder: Optional pre-initialized Encoder; if None, one will be created.
    :param output_dim: HDV dimensionality.
    :param emb_model: Embedding model name for the Encoder.
    :param n_workers: Ignored (retained for signature compatibility).
    """
    if encoder is None:
        encoder = Encoder(output_dim=output_dim, emb_model=emb_model)

    roles = encoder.generate_orthogonal_roles(num_roles=3)

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

    for qn, stim, pr, expl in rows:
        qobj = {
            "question_number": qn,
            "stimulus":        stim or "",
            "prompt":          pr   or "",
            "explanation":     expl or "",
        }
        hdv = encoder.generate_question_hdv_from_json(qobj, roles)

        buf = io.BytesIO()
        np.save(buf, hdv.astype("float32"), allow_pickle=False)
        blob = buf.getvalue()
        con.execute(
            f"INSERT INTO {hdv_table} (question_number, hdv_blob) VALUES (?, ?);",
            (int(qn), blob)
        )

    con.close()
    print(f"Initialized '{hdv_table}' with {len(rows)} HDVs.")
