import psycopg2, numpy as np
from .loader import VectorLoader

class PostgresLoader(VectorLoader):
    def __init__(self, dsn: str, table: str = "hdvs"):
        self.dsn = dsn
        self.table = table

    def load(self):
        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()
        cur.execute(f"SELECT question_number, hdv_blob FROM {self.table}")
        rows = cur.fetchall()
        conn.close()
        ids, hdvs = zip(*rows)
        vectors = np.vstack([np.load(blob, allow_pickle=False) for blob in hdvs]).astype('float32')
        return vectors, list(ids)
