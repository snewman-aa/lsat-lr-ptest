import duckdb
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from loguru import logger

from .models import (
    GenerateRequest,
    AnswerItem,
    SaveTestRequest,
    # Response Models
    GenerateTestResponse,
    TestSummary,
    TestDetails,
    QuestionList,
    StatusResponse,
    HealthStatus,
    # Component Models
    SampleQuestion,
    Question,
    UserAnsweredQuestion,
    QuestionListItem
)

from lsat_lr_ptest.config import load_config, Settings
from lsat_lr_ptest.encoder import Encoder
from ..clients import GeminiClient
from lsat_lr_ptest.vector_index.index import query_index, load_index

# --- Global Configuration and Setup ---
cfg: Settings = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Database Configuration ---
DB_PATH = cfg.db.duckdb.path
QUESTION_TABLE = cfg.db.duckdb.question_table

# --- Encoder Configuration ---
encoder = Encoder(
    output_dim=cfg.encoder.output_dim,
    emb_model=cfg.encoder.emb_model
)
ROLES = encoder.generate_orthogonal_roles(num_roles=3)

# --- Vector Index Configuration ---
INDEX_PATH = PROJECT_ROOT / cfg.vector_index.index_path
IDS_PATH = PROJECT_ROOT / cfg.vector_index.ids_path

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}")
if not IDS_PATH.exists():
    raise FileNotFoundError(f"FAISS IDs file not found at {IDS_PATH}")

vector_index = load_index(INDEX_PATH)
question_ids_in_index = np.load(IDS_PATH).tolist()

if vector_index is None:
    raise ValueError("Failed to load the FAISS index.")

gemini_client = GeminiClient()

app = FastAPI(
    title="LSAT Question WebApp API",
    description="API for generating LSAT-style tests, saving attempts, and retrieving results.",
    version="0.1.0",
)

static_dir = PROJECT_ROOT / "app" / "static"
templates_dir = PROJECT_ROOT / "app" / "templates"

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static"
)
templates = Jinja2Templates(directory=str(templates_dir))

# --- Database Connection ---
@lru_cache()
def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    Establishes and caches a DuckDB database connection.
    The connection is opened lazily on the first call and reused.

    Raises:
        FileNotFoundError: If the DuckDB database file is not found.
    Returns:
        duckdb.DuckDBPyConnection: An active connection to the DuckDB database.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB file not found at {DB_PATH}")
    return duckdb.connect(database=str(DB_PATH), read_only=False)


@app.get("/", response_class=HTMLResponse, summary="Main Page", tags=["General"])
async def read_index_page(request: Request):
    """Serves the main HTML page of the application."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_model=GenerateTestResponse, summary="Generate a New Test", tags=["Tests"])
async def generate_test_from_prompt(req: GenerateRequest):
    """
    Generates a new test based on a user prompt.

    - Creates a test record.
    - Generates a sample question via LLM.
    - Encodes the sample question to an HDV.
    - Finds similar questions using vector search.
    - Stores links between the test and similar questions.
    - Retrieves full details for these similar questions.
    """
    db_conn = get_db_connection()
    try:
        db_conn.begin()
        insert_test_sql = "INSERT INTO tests (prompt) VALUES (?) RETURNING test_id;"
        result = db_conn.execute(insert_test_sql, (req.prompt,)).fetchone()
        if not result or result[0] is None:
            db_conn.rollback()
            raise HTTPException(status_code=500, detail="Failed to create test and retrieve ID.")
        test_id: int = result[0]

        sample_question_obj: SampleQuestion = gemini_client.generate_sample_question(req.prompt)
        sample_question_dict = {
            "stimulus": sample_question_obj.stimulus,
            "prompt": sample_question_obj.prompt,
            "explanation": sample_question_obj.explanation
        }
        query_hdv = encoder.generate_question_hdv_from_json(sample_question_dict, ROLES).astype("float32")

        top_k = cfg.vector_index.top_k
        metric = cfg.vector_index.metric
        indices_from_faiss, distances_from_faiss = query_index(vector_index, query_hdv, k=top_k, metric=metric)
        similar_question_actual_ids = [question_ids_in_index[i] for i in indices_from_faiss]

        insert_test_question_sql = "INSERT INTO test_questions(test_id, question_number) VALUES (?, ?);"
        for q_id in similar_question_actual_ids:
            db_conn.execute(insert_test_question_sql, (test_id, q_id))
        db_conn.commit()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        try:
            db_conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback attempt failed: {rb_e}", exc_info=True)
            pass
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    retrieved_questions_data = []
    select_questions_sql = f"""
        SELECT question_number, stimulus, prompt, A, B, C, D, E, correct_answer, explanation
        FROM {QUESTION_TABLE} WHERE question_number = ?
    """
    for q_id in similar_question_actual_ids:
        row = db_conn.execute(select_questions_sql, (q_id,)).fetchone()
        if row:
            retrieved_questions_data.append({
                "question_number": row[0], "stimulus": row[1], "prompt": row[2],
                "answers": [ans for ans in [row[3], row[4], row[5], row[6], row[7]] if ans is not None],
                "correct_answer": row[8], "explanation": row[9],
            })
    return GenerateTestResponse(
        test_id=test_id,
        sample_question=sample_question_obj,
        similar_questions=[Question(**data) for data in retrieved_questions_data],
        distances=distances_from_faiss
    )


@app.post("/save_test", response_model=StatusResponse, summary="Save User's Test Answers", tags=["Tests"])
async def save_test_answers(req: SaveTestRequest):
    """Saves the answers submitted by a user for a specific test."""
    db_conn = get_db_connection()
    try:
        db_conn.begin()
        insert_response_sql = "INSERT INTO test_responses (test_id, question_number, selected_answer) VALUES (?, ?, ?);"
        for ans_item in req.answers:
            db_conn.execute(insert_response_sql, (req.test_id, ans_item.question_number, ans_item.selected_answer))
        db_conn.commit()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        try:
            db_conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback attempt failed: {rb_e}", exc_info=True)
            pass
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    return StatusResponse(status="ok", message=f"Answers for test_id {req.test_id} saved successfully.")


@app.get("/tests/{test_id}", response_model=TestDetails, summary="Get Specific Test Details", tags=["Tests"])
async def get_test_details(test_id: int):
    """
    Return a detailed test record: prompt, timestamp, each question
    with its answers & user’s selection, and the overall score.
    """
    db_conn = get_db_connection()
    test_meta_row = db_conn.execute(
        "SELECT prompt, created_at FROM tests WHERE test_id = ?;", (test_id,)
    ).fetchone()
    if not test_meta_row:
        raise HTTPException(status_code=404, detail=f"Test with id {test_id} not found.")
    prompt_text: str = test_meta_row[0]
    created_at_ts: datetime = test_meta_row[1]

    fetch_test_details_sql = f"""
        SELECT tq.question_number, q.stimulus, q.prompt, q.A, q.B, q.C, q.D, q.E,
               q.correct_answer, q.explanation, tr.selected_answer
        FROM test_questions AS tq
        JOIN {QUESTION_TABLE} AS q ON q.question_number = tq.question_number
        LEFT JOIN test_responses AS tr ON tr.test_id = tq.test_id AND tr.question_number = q.question_number
        WHERE tq.test_id = ? ORDER BY tq.rowid;
    """
    question_rows = db_conn.execute(fetch_test_details_sql, (test_id,)).fetchall()

    questions_for_response_data = []
    correct_answers_count = 0
    for q_row in question_rows:
        (qn, stim, pr, a, b, c, d, e, corr_ans, expl, selected_ans) = q_row
        # app.models.UserAnsweredQuestion
        questions_for_response_data.append({
            "question_number": qn, "stimulus": stim, "prompt": pr,
            "answers": [ans for ans in [a, b, c, d, e] if ans is not None],
            "correct_answer": corr_ans, "explanation": expl, "selected_answer": selected_ans
        })
        if selected_ans == corr_ans:
            correct_answers_count += 1

    total_questions_in_test = len(questions_for_response_data)
    score_str = f"{correct_answers_count}/{total_questions_in_test}" if total_questions_in_test > 0 else "0/0"

    return TestDetails(
        test_id=test_id,
        prompt=prompt_text,
        created_at=created_at_ts,
        questions=[UserAnsweredQuestion(**data) for data in questions_for_response_data],
        score=score_str
    )


@app.get("/tests", response_model=list[TestSummary], summary="List All Test Summaries", tags=["Tests"])
async def list_all_tests():
    """Return a list of past tests, each with prompt, timestamp, and score."""
    db_conn = get_db_connection()
    list_tests_sql = f"""
        SELECT t.test_id, t.prompt, t.created_at,
               COALESCE(SUM(CASE WHEN tr.selected_answer = q.correct_answer THEN 1 ELSE 0 END), 0) AS correct_responses,
               (SELECT COUNT(*) FROM test_questions tq_inner WHERE tq_inner.test_id = t.test_id) AS total_questions_in_test
        FROM tests t
        LEFT JOIN test_questions tq ON t.test_id = tq.test_id
        LEFT JOIN test_responses tr ON tq.test_id = tr.test_id AND tq.question_number = tr.question_number
        LEFT JOIN {QUESTION_TABLE} q ON tr.question_number = q.question_number
        GROUP BY t.test_id, t.prompt, t.created_at ORDER BY t.created_at DESC;
    """
    test_summary_rows = db_conn.execute(list_tests_sql).fetchall()

    summaries = []
    for ts_row in test_summary_rows:
        test_id_val, prompt_val, created_at_val, correct, total = ts_row
        score = f"{correct}/{total}" if total > 0 else "0/0"
        summaries.append(TestSummary(
            test_id=int(test_id_val), prompt=prompt_val,
            created_at=created_at_val,
            score=score
        ))
    return summaries


@app.get("/questions", response_model=QuestionList, summary="List Stored Questions (Paginated)", tags=["Questions"])
async def list_stored_questions(limit: int = 10, offset: int = 0):
    """Retrieves a paginated list of questions (number and prompt only)."""
    db_conn = get_db_connection()
    try:
        list_questions_sql = f"""
            SELECT question_number, prompt FROM {QUESTION_TABLE}
            ORDER BY question_number LIMIT ? OFFSET ?;
        """
        question_rows = db_conn.execute(list_questions_sql, (limit, offset)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching questions: {str(e)}")

    return QuestionList(
        questions=[QuestionListItem(question_number=qn, prompt=pr) for qn, pr in question_rows]
    )


@app.delete("/tests", response_model=StatusResponse, summary="Delete All Test Data", tags=["Admin"])
async def delete_all_test_data():
    """
    Deletes ALL test-related data by dropping and recreating tables and sequences.
    USE WITH CAUTION - DESTRUCTIVE OPERATION.
    """
    db_conn = get_db_connection()
    try:
        logger.info("Attempting to delete all test data by dropping and recreating tables.")
        db_conn.begin()

        db_conn.execute("DROP TABLE IF EXISTS test_responses;")
        db_conn.execute("DROP TABLE IF EXISTS test_questions;")
        db_conn.execute("DROP TABLE IF EXISTS tests;")
        db_conn.execute("DROP SEQUENCE IF EXISTS test_id_seq;")

        db_conn.execute("CREATE SEQUENCE test_id_seq START 1;")
        db_conn.execute("""
            CREATE TABLE tests (
              test_id     INTEGER DEFAULT nextval('test_id_seq') PRIMARY KEY,
              prompt      TEXT NOT NULL,
              created_at  TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """)
        db_conn.execute(f"""
            CREATE TABLE test_questions (
              test_id         INTEGER NOT NULL,
              question_number INTEGER NOT NULL,
              PRIMARY KEY (test_id, question_number),
              FOREIGN KEY(test_id) REFERENCES tests(test_id),
              FOREIGN KEY(question_number) REFERENCES {QUESTION_TABLE}(question_number)
            );
        """)
        db_conn.execute("""
            CREATE TABLE test_responses (
              test_id          INTEGER NOT NULL,
              question_number  INTEGER NOT NULL,
              selected_answer  TEXT NOT NULL,
              PRIMARY KEY (test_id, question_number),
              FOREIGN KEY(test_id, question_number) REFERENCES test_questions(test_id, question_number)
            );
        """)
        db_conn.commit()
        logger.info("Successfully deleted and recreated test data tables.")
    except Exception as e:
        logger.error(f"Error during full test data clear (drop/recreate): {e}", exc_info=True)
        try:
            db_conn.rollback()
        except Exception as rb_e:
            logger.error(f"Rollback attempt failed: {rb_e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during data clear: {str(e)}")

    return StatusResponse(status="ok", message="All test data cleared and tables reset successfully.")


@app.get("/healthz", response_model=HealthStatus, summary="Health Check", tags=["General"])
def health_check_status(): # Renamed
    """
    Provides a health check endpoint.
    Checks basic service availability and database connectivity.
    """
    db_conn_status = "connected"
    try:
        con = get_db_connection()
        con.execute("SELECT 1;")
    except Exception:
        db_conn_status = "disconnected"
    return HealthStatus(status="ok", database_status=db_conn_status)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
        log_level="info"
    )