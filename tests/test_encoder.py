import numpy as np
from encoder.encoder import Encoder


def test_simhash_simple():
    enc = Encoder(output_dim=5, emb_model="sbert")
    emb = np.array([1.0, 2.0, 3.0])
    hdv = enc.simhash_projection(emb)
    assert hdv.shape == (5,)
    # each index i hashes to some bucket j
    # sum of hdv equals sum of emb
    assert np.isclose(hdv.sum(), emb.sum())


def test_roles_are_orthogonal():
    enc = Encoder(output_dim=50, emb_model="sbert")
    roles = enc.generate_orthogonal_roles(num_roles=3)
    mat = np.stack(list(roles.values()), axis=1)
    # columns should be orthonormal: M^T M ≈ I
    prod = mat.T @ mat
    assert np.allclose(prod, np.eye(3), atol=1e-6)


def test_bind_and_bundle():
    enc = Encoder(output_dim=16, emb_model="sbert")
    roles = enc.generate_orthogonal_roles(num_roles=2)
    # make two orthogonal HDVs
    v1, v2 = np.eye(16)[0], np.eye(16)[1]
    b1 = enc.bind(roles["stimulus"], v1)
    b2 = enc.bind(roles["prompt"],   v2)
    bun = enc.bundle([b1, b2])
    assert bun.shape == (16,)
    # bundling two nonzero vectors must be nonzero
    assert np.any(bun != 0)
