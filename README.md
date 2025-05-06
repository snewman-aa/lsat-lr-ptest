# LSAT RAG Practice App
An interactive web application for generating and practicing LSAT-style logic questions using a
Retriever–Augmented Generation (RAG) approach with VSA Holographic Reduced Representation indexing.

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/lsat-rag.git
cd lsat-rag
```

### 2. Create & activate a virtual environment
```bash
bash
python3 -m venv .venv
source .venv/bin/activate      # on macOS/Linux
.venv\Scripts\activate         # on Windows
```

### 3. Install dependencies

#### pip editable package
```bash
pip install --upgrade pip
pip install -e .
```
#### using uv
```bash
uv pip install -e .
```
This will install required libraries (FastAPI, DuckDB, FAISS, Sentence-Transformers, etc.)

## 🗂 Data Preparation

### 1. Create a top-level `data/` folder if it doesn’t already exist:

```bash
mkdir -p data
```

### 2. Download the provided LSAT questions file or run the provided notebook to generate your own:

```bash
data/lsat_questions_deduped.tsv
```

* This TSV must have the columns:
    ```tsv
    question_number\tstimulus\tprompt\tA\tB\tC\tD\tE\tcorrect_answer\texplanation\n
    ```

## ⚙️ Configuration
All runtime settings live in `config.yaml` at the project root:

```yaml
db:
  type: duckdb
  duckdb:
    path: data/duckdb_questions.db
    question_table: questions
    tsv_path: lsat_questions_deduped.tsv
    hdv_table: hdvs

vector_index:
  metric: cosine
  top_k: 5

encoder:
  output_dim: 10000
  emb_model: sbert

llm:
  model: gemini-2.0-flash
  api_key_env: GEMINI_API_KEY

server:
  host: 0.0.0.0
  port: 8000
```

* `tsv_path` must match the filename you placed in data/.

### Ensure your Gemini API key is exported in your shell:

```bash
export GEMINI_API_KEY="your-key-here"
```

## 🚀 Setup
Before running the app, initialize the database tables, project all HDVs, and build the FAISS index:

```bash
python setup.py
```
of if using uv:
```bash
uv run setup.py
```

You should see a four‐step progress:

1. Initializing questions table

2. Initializing test tables

3. Projecting HDVs into DuckDB

4. Building FAISS index

If the TSV is missing, `setup.py` will bail out with a clear error.

## 🏃‍♂️ Running the App
There are two ways to start the server:

### 1. Via the provided runner
```bash
python run.py
```
This script reads `config.yaml`, mounts the FastAPI app, and launches Uvicorn on the configured host and port.

### 2. Directly with Uvicorn
```bash
uvicorn app.server:app --reload \
  --host 0.0.0.0 --port 8000
```
* `--reload` enables auto‐reload on code changes (development only)

* Adjust `--host`/`--port` to match your environment

Once running, navigate to:

```
http://localhost:8000
```
and you’ll see the practice‐test UI.

## 🛠️ Project Layout
```
.
├── app/                 # FastAPI application (templates, static, server code)
├── clients/             # GeminiClient wrapper
├── config.yaml          # BaseSettings config file
├── data/                # LSAT TSV & DuckDB file
├── db/
│   ├── models.py        # DuckDB initialization logic
│   └── duckdb_loader.py # Vector loader for storing HDVs
├── encoder/
│   └── encoder.py       # Role binding & HDV logic
├── vector_index/
│   └── build_index.py   # FAISS index builder
│   └── index.py         # Query & load index
├── setup.py             # One-step project setup (DB, HDVs, FAISS)
├── run.py               # Entry point to start the server
└── tests/               # Pytest suite
```

## 🎯 Next Steps

### HDVs
* Experiment with contextual embeddings of higher dimensions
* Implementing some weighting scheme

### UI/Webapp
* Add a place to provide feedback on question appropriateness
* Add a "more like this" button to generate similar questions

### Server
* I want to potentially host this on a prod server using Postgres DB