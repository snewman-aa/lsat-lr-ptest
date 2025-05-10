import sys
from pathlib import Path
from loguru import logger

from lsat_lr_ptest.config import load_config
from lsat_lr_ptest.db.schema import init_question_table, init_test_tables, init_hdv_table
from lsat_lr_ptest.vector_index.build_index import build_index


def main():
    cfg = load_config()
    root = Path(__file__).parent.resolve()

    db_cfg = cfg.db.duckdb
    enc_cfg = cfg.encoder
    tsv_path = cfg.db.duckdb.tsv_path
    data_dir = root / "data"
    db_path  = root / db_cfg.path
    index_path = root / "vector_index" / "faiss_index.bin"
    ids_path = root / "vector_index" / "ids.npy"

    # 1) ensure data/ is there
    if not data_dir.exists():
        logger.info(f"Creating data directory at {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    # 2) check for TSV
    if not tsv_path.exists():
        logger.error(f"Could not find your LSAT TSV at {tsv_path}"
                     f"\nPlease place provided lsat_questions_deduped.tsv"
                     f" into the data/ folder")
        sys.exit(1)

    # 3) DuckDB tables
    logger.info("[1/4] Initializing questions table…")
    init_question_table(tsv_path, db_path, db_cfg.question_table)

    logger.info("[2/4] Initializing test tables…")
    init_test_tables(db_path)

    logger.info("[3/4] Projecting HDVs into DuckDB…")
    init_hdv_table(
        db_path=db_path,
        question_table=db_cfg.question_table,
        hdv_table=db_cfg.hdv_table,
        encoder=None,  # let it build its own
        output_dim=enc_cfg.output_dim,
        emb_model=enc_cfg.emb_model,
    )

    # 4) FAISS index
    logger.info("[4/4] Building FAISS index…")
    build_index(
        db_path=db_path,
        hdv_table=db_cfg.hdv_table,
        index_path=index_path,
        ids_path=ids_path,
        metric=cfg.vector_index.metric
    )

    logger.info("Setup complete! You can now run `run.py` to start the server.")

if __name__ == "__main__":
    main()
