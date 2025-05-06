from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings_yaml import YamlBaseSettings
from pydantic_settings import SettingsConfigDict


class DuckDBSettings(BaseModel):
    path: Path
    question_table: str
    hdv_table: str


class PostgresSettings(BaseModel):
    dsn: str
    question_table: str
    hdv_table: str


class DBSettings(BaseModel):
    type: Literal["duckdb", "postgres"]
    duckdb: DuckDBSettings
    postgres: PostgresSettings | None = None


class VectorIndexSettings(BaseModel):
    metric: Literal["METRIC_INNER_PRODUCT", "METRIC_L2"]
    top_k: int


class EncoderSettings(BaseModel):
    output_dim: int
    emb_model: str
    emb_model_name: str


class LLMSettings(BaseModel):
    model: str
    api_key_env: str


class ServerSettings(BaseModel):
    host: str
    port: int


class Settings(YamlBaseSettings):
    db: DBSettings
    vector_index: VectorIndexSettings
    encoder: EncoderSettings
    llm: LLMSettings
    server: ServerSettings

    model_config = SettingsConfigDict(
        yaml_file = Path(__file__).resolve().parent / "config.yaml",
        yaml_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_prefix="",
    )


def load_config() -> Settings:
    """
    Load application settings from config.yaml, then overlay any
    environment variables (e.g. DB__TYPE or LLM__PROVIDER).
    """
    return Settings()
