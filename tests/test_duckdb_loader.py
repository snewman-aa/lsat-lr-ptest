import io
import numpy as np
import duckdb
from lsat_lr_ptest.db.duckdb_loader import DuckDBLoader


def test_loader_roundtrip(tmp_path):
    db = tmp_path/"db.db"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE hdvs(question_number INT, hdv_blob BLOB);")
    vec = np.arange(10, dtype="float32")
    buf = io.BytesIO()
    np.save(buf, vec, allow_pickle=False)
    blob = buf.getvalue()
    con.execute("INSERT INTO hdvs VALUES (7, ?);", (blob,))
    con.close()

    loader = DuckDBLoader(db, table="hdvs")
    hdvs, ids = loader.load()
    assert ids == [7]
    assert hdvs.shape == (1,10)
    assert np.allclose(hdvs[0], vec)
