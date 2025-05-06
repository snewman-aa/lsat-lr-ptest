import duckdb
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel
from functools import lru_cache

from config import load_config
from encoder.encoder import Encoder
from clients.gemini_client import GeminiClient
from vector_index.index import query_index, load_index


cfg = load_config()
project_root = Path(__file__).resolve().parent.parent

db_type = cfg.db.type
db_path = cfg.db.duckdb.path
question_table = cfg.db.duckdb.question_table
embed_model = cfg.encoder.emb_model
model_name = cfg.encoder.emb_model_name

# ─── Database Connection ──────────────────────────────────────────
@lru_cache()
def get_db_connection():
    """
    Lazily open & cache a DuckDB connection the first time
    any endpoint needs it.  Raises if the file is missing.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path}")
    return duckdb.connect(database=str(db_path))

# ─── Load the Encoder ─────────────────────────────────────────────
# init encoder with defaults (10k and sbert)
encoder = Encoder()

# Generate 3 roles for 'stimulus', 'prompt', and 'explanation'
roles = encoder.generate_orthogonal_roles(num_roles=3)

# ─── Load the Vector Index ───────────────────────────────────────
index_path = project_root / "vector_index" / "faiss_index.bin"
ids_path = project_root / "vector_index" / "ids.npy"
if not index_path.exists():
    raise FileNotFoundError(f"FAISS index file not found at {index_path}")
index = load_index(index_path)
ids = np.load(ids_path).tolist()
if index is None:
    raise ValueError("Failed to load the FAISS index.")

# ─── Load the LLM Client ──────────────────────────────────────────
gemini = GeminiClient()

# ─── FastAPI App Setup ─────────────────────────────────────────────
app = FastAPI()

static_dir = project_root / "app" / "static"
templates_dir = project_root / "app" / "templates"

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static"
)

templates = Jinja2Templates(directory=str(templates_dir))


class GenerateRequest(BaseModel):
    prompt: str


class AnswerItem(BaseModel):
    question_number: int
    selected_answer: str


class SaveTestRequest(BaseModel):
    test_id: int
    answers: list[AnswerItem]


class TestSummary(BaseModel):
    test_id: int
    prompt: str
    created_at: str
    score: str


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Gather top-k questions from the question database
    based on the provided prompt.
    Generate a sample question using the LLM
    Create HDV for the generated question
    Vector search for similar questions
    """
    con_insert = get_db_connection()
    con_insert.execute(
        "INSERT INTO tests (prompt) VALUES (?);",
        (req.prompt,)
        )
    test_id = con_insert.execute("SELECT max(test_id) FROM tests;").fetchone()[0]

    sample = gemini.generate_sample_question(req.prompt)

    json_obj = {
        "question_number": None,
        "stimulus": sample.stimulus,
        "prompt": sample.prompt,
        "explanation": sample.explanation
    }
    query_hdv = encoder.generate_question_hdv_from_json(json_obj, roles).astype("float32")

    k = cfg.vector_index.top_k
    metric = cfg.vector_index.metric
    inds, dists = query_index(index, query_hdv, k=k, metric=metric)
    similar_ids = [ids[i] for i in inds]

    con_q = get_db_connection()
    for qid in similar_ids:
        con_q.execute(
            "INSERT INTO test_questions(test_id, question_number) VALUES (?, ?);",
            (test_id, qid)
            )

    results = []
    con_read = get_db_connection()
    for qid in similar_ids:
        row = con_read.execute(f"""
            SELECT question_number, stimulus, prompt, A, B, C, D, E, correct_answer, explanation
            FROM {question_table}
            WHERE question_number = {qid}
        """).fetchone()
        if row:
            results.append({
                "question_number": row[0],
                "stimulus": row[1],
                "prompt": row[2],
                "answers": [row[3], row[4], row[5], row[6], row[7]],
                "correct_answer": row[8],
                "explanation": row[9],
            })

    return {
        "test_id": test_id,
        "sample_question": sample,
        "similar_questions": results,
        "distances": dists
    }


@app.post("/save_test")
async def save_test(req: SaveTestRequest):
    con = get_db_connection()
    for ans in req.answers:
        con.execute(
            "INSERT INTO test_responses (test_id, question_number, selected_answer) VALUES (?, ?, ?);",
            (req.test_id, ans.question_number, ans.selected_answer)
        )
    return {"status": "ok", "test_id": req.test_id}


@app.get("/tests/{test_id}")
async def get_test(test_id: int):
    """
    Return a detailed test record: prompt, timestamp, each question
    with its answers & user’s selection, and the overall score.
    """
    con = get_db_connection()

    row = con.execute(
        "SELECT prompt, created_at FROM tests WHERE test_id = ?;",
        (test_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Test not found")
    prompt, created_at = row

    sql = f"""
        SELECT
            tq.question_number,
            q.stimulus, q.prompt,
            q.A, q.B, q.C, q.D, q.E,
            q.correct_answer, q.explanation,
            tr.selected_answer
        FROM test_questions    AS tq
        JOIN {question_table} AS q
          ON q.question_number = tq.question_number
        LEFT JOIN test_responses AS tr
          ON tr.test_id         = tq.test_id
         AND tr.question_number = q.question_number
        WHERE tq.test_id = ?
        ORDER BY tq.rowid;
    """
    rows = con.execute(sql, (test_id,)).fetchall()

    questions = []
    for (
        qn, stim, pr,
        A, B, C, D, E,
        corr, expl,
        sel
    ) in rows:
        selected = sel or ""
        questions.append({
            "question_number": qn,
            "stimulus": stim,
            "prompt": pr,
            "answers": [A, B, C, D, E],
            "correct_answer": corr,
            "explanation": expl,
            "selected_answer": selected
        })

    total = len(questions)
    correct_cnt = sum(
        1 for q in questions
        if q["selected_answer"] == q["correct_answer"]
    )
    score_str = f"{correct_cnt}/{total}"

    return {
        "test_id":     test_id,
        "prompt":      prompt,
        "created_at":  str(created_at),
        "questions":   questions,
        "score":       score_str,
    }


@app.get("/tests", response_model=list[TestSummary])
async def list_tests():
    """
    Return a list of past tests, each with prompt, timestamp, and score
    """
    con = get_db_connection()
    rows = con.execute(f"""
        SELECT
          t.test_id,
          t.prompt,
          t.created_at::VARCHAR   AS created_at,
          CAST(
            SUM(CASE WHEN tr.selected_answer = q.correct_answer THEN 1 ELSE 0 END)
            AS INTEGER
          ) AS correct,
          COUNT(tr.question_number) AS total
        FROM tests t
        LEFT JOIN test_responses tr ON t.test_id = tr.test_id
        LEFT JOIN {question_table} q ON tr.question_number = q.question_number
        GROUP BY t.test_id, t.prompt, t.created_at
        ORDER BY t.created_at DESC;
    """).fetchall()

    summaries = []
    for test_id, prompt, created_at, correct, total in rows:
        score = f"{correct}/{total}" if total > 0 else "0/0"
        summaries.append({
            "test_id": int(test_id),
            "prompt": prompt,
            "created_at": created_at,
            "score": score
        })
    return summaries


@app.get("/questions", response_class=JSONResponse)
async def list_questions(limit: int = 10, offset: int = 0):
    if not db_path.exists():
        raise HTTPException(500, f"DB not found at {db_path}")
    con = get_db_connection()
    try:
        sql = f"""
            SELECT question_number, prompt
            FROM {question_table}
            ORDER BY question_number
            LIMIT ? OFFSET ?;
        """
        rows = con.execute(sql, [limit, offset]).fetchall()
    finally:
        con.close()

    return {"questions": [{"question_number": qn, "prompt": pr} for qn, pr in rows]}


@app.delete("/tests")
async def clear_tests():
    """
    Delete all tests, test_questions, and test_responses,
    then restart the sequence at 1.
    """
    con = get_db_connection()
    con.execute("DELETE FROM test_responses;")
    con.execute("DELETE FROM test_questions;")
    con.execute("DELETE FROM tests;")
    con.execute("DROP SEQUENCE IF EXISTS test_id_seq;")
    con.execute("CREATE SEQUENCE test_id_seq START 1;")
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return JSONResponse(content={"status": "ok"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
