from pathlib import Path
import numpy as np
from loguru import logger

from db.duckdb_loader import DuckDBLoader
from vector_index.index import build_faiss_index, save_index


def build_index(
    db_path: Path,
    hdv_table: str,
    index_path: Path,
    ids_path: Path,
    metric: str = "METRIC_INNER_PRODUCT",
) -> None:
    """
    Build a FAISS index from HDVs stored in DuckDB, and write out the
    serialized index and the corresponding ID map.

    Args:
        db_path:       Path to DuckDB file
        hdv_table:     Name of the table that holds (question_number, hdv_blob)
        index_path:    File path where the FAISS index will be saved (.bin)
        ids_path:      File path where the ID array will be saved (.npy)
        metric:        Similarity metric; e.g. "METRIC_INNER_PRODUCT" or "METRIC_L2"
    """
    loader = DuckDBLoader(db_path, table=hdv_table)
    hdv_array, ids = loader.load()

    index = build_faiss_index(hdv_array, metric=metric)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    save_index(index, index_path)
    np.save(ids_path, np.array(ids, dtype=np.int32), allow_pickle=False)

    logger.info(f"FAISS index saved to: {index_path}")
    logger.info(f"ID map saved to:    {ids_path}")



if __name__ == "__main__":
    build_index()
