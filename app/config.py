from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    # ── Azure OpenAI (Chat)
    AZURE_OPENAI_ENDPOINT: str = "https://central-prod-translation.openai.azure.com/"
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "central-prod-translation-model"
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"

    # ── Embeddings 
    EMBEDDING_PROVIDER: str = "huggingface"  # "azure" | "huggingface"
    AZURE_EMBEDDING_DEPLOYMENT: Optional[str] = None
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QDRANT_VECTOR_SIZE: int = 384  # 1536 for text-embedding-ada-002

    # ── Qdrant 
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "bpss_documents"

    # ── Data 
    DATA_DIR: str = str(BASE_DIR / "data")

    # ── Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── App 
    APP_NAME: str = "BPSS Compliance Agent"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RETRIEVAL: int = 6


settings = Settings()
