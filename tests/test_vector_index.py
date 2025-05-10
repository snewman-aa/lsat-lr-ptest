import numpy as np
from lsat_lr_ptest.vector_index.index import build_faiss_index, query_index


def test_faiss_build_and_query():
    # 3 points in 4-D
    pts = np.array([[1.,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]], dtype="float32")
    idx = build_faiss_index(pts, metric="cosine")
    # query [1,0,0,0], top 1 => index 0
    inds, dists = query_index(idx, np.array([1,0,0,0], dtype="float32"), k=1)
    assert inds == [0]
