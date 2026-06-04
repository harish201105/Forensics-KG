from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    # Embedding model for semantic graph search (1536 dims for text-embedding-3-small)
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "forensics_kg_2024"
    neo4j_database: str = "neo4j"

    # Paths
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    bloodstain_data_dirs: List[str] = Field(default=[
        "../../1-s2.0-S2352340918301902-mmc2",
        "../../1-s2.0-S2352340918301902-mmc3",
    ])

    # Indian Kanoon API (for court judgment ingestion)
    indian_kanoon_api_token: str = ""

    # Kaggle API (for SOCOFing fingerprints)
    kaggle_username: str = ""
    kaggle_key: str = ""

    # Processing
    image_max_size: int = 1024
    min_stain_area: int = 100
    max_stain_area: int = 50000

    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def resolved_bloodstain_dirs(self) -> List[Path]:
        base = Path(__file__).parent.parent
        return [base / d for d in self.bloodstain_data_dirs]

    @property
    def ontology_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "ontology" / "forensics_ontology.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
