import numpy as np
from pathlib import Path

from config import load_config
from db.duckdb_loader import DuckDBLoader
from vector_index.index import build_faiss_index, save_index


def main():
    project_root = Path(__file__).parent.parent
    cfg = load_config(project_root / "config.yaml")

    # database configuration
    db_type = cfg['db']['type']
    db_conf = cfg['db'][db_type]
    db_path = project_root / db_conf['path']
    hdv_table = db_conf['hdv_table']

    loader = DuckDBLoader(db_path, table=hdv_table)
    hdv_array, ids = loader.load()

    metric = cfg['vector_index']['metric']
    index = build_faiss_index(hdv_array, metric=metric)

    out_dir = project_root / "vector_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, out_dir / "faiss_index.bin")
    np.save(out_dir / "ids.npy", np.array(ids, dtype=np.int32))

    print(f"FAISS index and ID map written to: {out_dir}")


if __name__ == "__main__":
    main()
