from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "policies"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 800
    chunk_overlap: int = 150


settings = Settings()