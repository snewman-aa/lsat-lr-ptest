import io
from pathlib import Path
import duckdb
import numpy as np

from lsat_lr_ptest.encoder import Encoder
from loguru import logger


def init_question_table(
    db_path: Path,
    tsv_path: Path,
    question_table_name: str # Renamed for clarity
):
    """
    (Re)creates the specified question table in DuckDB from a TSV file.
    This function will DROP the table if it already exists.

    Args:
        db_path (Path): Path to the DuckDB database file.
        tsv_path (Path): Path to the TSV file containing question data.
        question_table_name (str): Name of the table to create for questions.

    Raises:
        FileNotFoundError: If the TSV input file is not found.
    """
    if not tsv_path.exists():
        logger.error(f"Could not find TSV at {tsv_path}")
        raise FileNotFoundError(f"Could not find TSV at {tsv_path}")

    # ensure parent directory for db_path exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=str(db_path))
    try:
        logger.info(f"Dropping table '{question_table_name}' if it exists.")
        con.execute(f"DROP TABLE IF EXISTS {duckdb.quote_ident(question_table_name)};")

        logger.info(f"Creating table '{question_table_name}' from TSV: {tsv_path}")
        # ensure column names in TSV: question_number, stimulus, prompt, A, B, C, D, E, correct_answer, explanation
        con.execute(f"""
            CREATE TABLE {duckdb.quote_ident(question_table_name)} AS
            SELECT
              CAST(question_number AS INTEGER) AS question_number,
              stimulus::TEXT AS stimulus,         -- Explicit cast to TEXT
              prompt::TEXT AS prompt,             -- Explicit cast to TEXT
              A::TEXT AS A, B::TEXT AS B, C::TEXT AS C, D::TEXT AS D, E::TEXT AS E, -- Cast options to TEXT
              correct_answer::TEXT AS correct_answer, -- Cast to TEXT
              explanation::TEXT AS explanation      -- Explicit cast to TEXT
            FROM read_csv_auto(
              '{tsv_path.as_posix()}',
              delim='\\t', header=True,
              types={'question_number': 'INTEGER', 'stimulus': 'VARCHAR', ...}
              columns={{
                  'question_number': 'INTEGER', 'stimulus': 'VARCHAR', 'prompt': 'VARCHAR',
                  'A': 'VARCHAR', 'B': 'VARCHAR', 'C': 'VARCHAR', 'D': 'VARCHAR', 'E': 'VARCHAR',
                  'correct_answer': 'VARCHAR', 'explanation': 'VARCHAR'
              }}
            );
        """)

        logger.info(f"Creating primary key and index on 'question_number' for table '{question_table_name}'.")
        con.execute(f"ALTER TABLE {duckdb.quote_ident(question_table_name)} ALTER question_number SET NOT NULL;")
        con.execute(f"ALTER TABLE {duckdb.quote_ident(question_table_name)} ADD CONSTRAINT pk_{question_table_name} PRIMARY KEY (question_number);")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_qnum_{question_table_name} ON {duckdb.quote_ident(question_table_name)}(question_number);")
        logger.info(f"Table '{question_table_name}' initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing question table '{question_table_name}': {e}", exc_info=True)
        raise
    finally:
        con.close()


def init_test_tables(db_path: Path, question_table_name: str):
    """
    Creates or ensures existence of 'tests', 'test_questions', and 'test_responses' tables.
    Initializes 'test_id_seq' sequence for auto-incrementing test IDs.

    Args:
        db_path (Path): Path to the DuckDB database file.
        question_table_name (str): Name of the main question table, for foreign key reference.
    """
    con = duckdb.connect(database=str(db_path))
    try:
        logger.info("Initializing test-related tables and sequence.")
        con.execute("CREATE SEQUENCE IF NOT EXISTS test_id_seq START 1;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS tests (
              test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
              prompt      TEXT NOT NULL,
              created_at  TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS test_questions (
              test_id         INTEGER NOT NULL,
              question_number INTEGER NOT NULL,
              PRIMARY KEY (test_id, question_number), -- Composite primary key
              FOREIGN KEY(test_id) REFERENCES tests(test_id) ON DELETE CASCADE,
              FOREIGN KEY(question_number) REFERENCES {duckdb.quote_ident(question_table_name)}(question_number) ON DELETE RESTRICT
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS test_responses (
              test_id          INTEGER NOT NULL,
              question_number  INTEGER NOT NULL,
              selected_answer  TEXT NOT NULL,
              PRIMARY KEY (test_id, question_number),
              FOREIGN KEY(test_id, question_number) REFERENCES test_questions(test_id, question_number) ON DELETE CASCADE
              -- FOREIGN KEY(test_id) REFERENCES tests(test_id) ON DELETE CASCADE,
              -- FOREIGN KEY(question_number) REFERENCES {duckdb.quote_ident(question_table_name)}(question_number) ON DELETE RESTRICT
            );
        """)
        logger.info("Test-related tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing test tables: {e}", exc_info=True)
        raise
    finally:
        con.close()


def init_hdv_table(
    db_path: Path,
    question_table_name: str,
    hdv_table_name: str,
    encoder_instance: Encoder,  # pre-initialized encoder
) -> int:
    """
    Creates or refreshes the HDV store by projecting every question
    from the `question_table_name` into an HDV and inserting (question_number, hdv_blob)
    into `hdv_table_name`. This version runs entirely in the main thread.

    Args:
        db_path (Path): Path to the DuckDB file.
        question_table_name (str): Name of the questions table.
        hdv_table_name (str): Name of the table to store HDV blobs.
        encoder_instance (Encoder): A pre-initialized Encoder instance.

    Returns:
        int: The number of HDVs generated and stored.

    Raises:
        # Any exceptions from DB operations or encoder.
    """
    roles = encoder_instance.generate_orthogonal_roles(num_roles=3)

    con = duckdb.connect(database=str(db_path))
    try:
        logger.info(f"Dropping HDV table '{hdv_table_name}' if it exists.")
        con.execute(f"DROP TABLE IF EXISTS {duckdb.quote_ident(hdv_table_name)};")
        con.execute(f"""
            CREATE TABLE {duckdb.quote_ident(hdv_table_name)} (
              question_number INTEGER NOT NULL PRIMARY KEY,
              hdv_blob        BLOB NOT NULL,
              FOREIGN KEY(question_number) REFERENCES {duckdb.quote_ident(question_table_name)}(question_number) ON DELETE CASCADE
            );
        """)

        logger.info(f"Fetching questions from '{question_table_name}' to generate HDVs.")
        question_data_rows = con.execute(f"""
            SELECT question_number, stimulus, prompt, explanation
            FROM {duckdb.quote_ident(question_table_name)};
        """).fetchall()

        logger.info(f"Generating and inserting {len(question_data_rows)} HDVs...")
        con.begin()
        for qn, stim, pr, expl in question_data_rows:
            q_obj_for_encoder = {
                "question_number": qn,
                "stimulus":        stim or "",
                "prompt":          pr   or "",
                "explanation":     expl or "",
            }
            hdv = encoder_instance.generate_question_hdv_from_json(q_obj_for_encoder, roles)

            buf = io.BytesIO()
            np.save(buf, hdv.astype("float32"), allow_pickle=False)
            blob_data = buf.getvalue()

            con.execute(
                f"INSERT INTO {duckdb.quote_ident(hdv_table_name)} (question_number, hdv_blob) VALUES (?, ?);",
                (int(qn), blob_data)
            )
        con.commit()
        logger.info(f"Successfully initialized '{hdv_table_name}' with {len(question_data_rows)} HDVs.")
        return len(question_data_rows)
    except Exception as e:
        if con.isconnected():
            try: con.rollback()
            except Exception: pass
        logger.error(f"Error initializing HDV table '{hdv_table_name}': {e}", exc_info=True)
        raise
    finally:
        con.close()