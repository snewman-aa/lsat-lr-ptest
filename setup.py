from pathlib import Path
from loguru import logger

from config import load_config
from db.models import init_question_table, init_test_tables, init_hdv_table
from vector_index.build_index import build_index


def main():
    cfg = load_config()
    root = Path(__file__).parent.resolve()

    # 1) questions
    db_cfg   = cfg.db.duckdb
    tsv_path = root / db_cfg.tsv_path
    db_path  = root / db_cfg.path
    logger.info(f"[1/4] Initializing questions from {tsv_path} → {db_path}")
    init_question_table(tsv_path, db_path, db_cfg.question_table)

    # 2) tests + link tables
    logger.info("[2/4] Initializing tests and response tables")
    init_test_tables(db_path)

    # 3) HDVs
    enc_cfg = cfg.encoder
    logger.info("[3/4] Generating HDV table")
    init_hdv_table(
        db_path=db_path,
        question_table=db_cfg.question_table,
        hdv_table=db_cfg.hdv_table,
        encoder=None,  # let it build its own
        output_dim=enc_cfg.output_dim,
        emb_model=enc_cfg.emb_model,
        n_workers=cfg.parallel.n_workers if hasattr(cfg, "parallel") else None
    )

    # 4) FAISS index
    idx_dir    = root / "vector_index"
    idx_dir.mkdir(exist_ok=True)
    index_path = idx_dir / "faiss_index.bin"
    ids_path   = idx_dir / "ids.npy"
    logger.info(f"[4/4] Building FAISS index → {index_path}, {ids_path}")
    build_index(
        db_path=db_path,
        hdv_table=db_cfg.hdv_table,
        index_path=index_path,
        ids_path=ids_path,
        metric=cfg.vector_index.metric
    )

    logger.info("Setup complete!")


if __name__ == "__main__":
    main()
