from pathlib import Path
import duckdb, numpy as np
import io
from .loader import VectorLoader


class DuckDBLoader(VectorLoader):
    """
    Loads (question_number, HDV) pairs from a DuckDB database
    Expects a table with two columns: question_number, hdv_blob
    """
    def __init__(self, db_path: Path, table: str = "hdvs"):
        self.db_path = db_path
        self.table = table

    def load(self) -> tuple[np.ndarray, list[int]]:
        con = duckdb.connect(database=str(self.db_path))
        rows = con.execute(f"SELECT question_number, hdv_blob FROM {self.table}").fetchall()
        con.close()
        ids, blobs = zip(*rows)

        hdv_list = []
        for blob in blobs:
            buf = io.BytesIO(blob)
            arr = np.load(buf, allow_pickle=False)
            hdv_list.append(arr)

        hdv_array = np.vstack(hdv_list).astype('float32')
        return hdv_array, list(ids)
