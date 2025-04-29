import faiss
import numpy as np
from pathlib import Path


def build_faiss_index(
    hdv_vectors: np.ndarray,
    metric: str
) -> faiss.Index:
    """Builds and returns a FAISS index for the given (n, d) vector array."""
    dim = hdv_vectors.shape[1]
    if metric == 'METRIC_INNER_PRODUCT':
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(hdv_vectors)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(hdv_vectors)
    return index


def save_index(index: faiss.Index, path: Path):
    """Persist a FAISS index to disk."""
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))


def query_index(
    index: faiss.Index,
    query_vec: np.ndarray,
    k: int = 5,
    metric: str = 'METRIC_INNER_PRODUCT'
):
    """
    Return the top-k neighbor IDs and distances for a single query vector.
    """
    q = query_vec.reshape(1, -1).astype('float32')
    if metric == 'METRIC_INNER_PRODUCT':
        faiss.normalize_L2(q)
    distances, indices = index.search(q, k)
    return indices[0].tolist(), distances[0].tolist()
