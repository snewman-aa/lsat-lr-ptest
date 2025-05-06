import os

from config import load_config


def test_loads_all_sections():
    cfg = load_config()
    assert cfg.db.type == "duckdb"
    assert cfg.vector_index.top_k > 0
    assert isinstance(cfg.encoder.output_dim, int)
    assert cfg.llm.model.startswith("gemini")
    assert cfg.server.port > 0


# def test_env_override(monkeypatch):
#     monkeypatch.setenv("VECTOR_INDEX__TOP_K", "3")
#     cfg = load_config()
#     assert cfg.vector_index.top_k == 3
