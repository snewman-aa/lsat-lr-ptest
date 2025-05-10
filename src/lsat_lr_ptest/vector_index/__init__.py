from pathlib import Path
from .index import build_faiss_index, query_index

from lsat_lr_ptest.db.duckdb_loader import DuckDBLoader
# from db.postgres_loader import PostgresLoader


def load_index(db_config):
    if db_config['type'] == 'duckdb':
        loader = DuckDBLoader(Path(db_config['path']), table=db_config['table'])
    else:
        # loader = PostgresLoader(db_config['dsn'], table=db_config['table'])
        pass

    vectors, ids = loader.load()
    index = build_faiss_index(vectors, metric=db_config.get('metric','cosine'))
    return index, ids