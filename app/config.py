from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_user: str = "docquery"
    postgres_password: str = "docquery"
    postgres_db: str = "docquery"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    openai_api_key: str = ""
    embedding_provider: str = "ollama"          # "ollama" | "openai"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768             # nomic=768, text-embedding-3-small=1536
    ollama_base_url: str = "http://localhost:11434"
    chat_provider: str = "ollama"            # "ollama" | "openai"
    chat_model: str = "qwen2.5:14b"          # ollama: any pulled model; openai: "gpt-4o"

    top_k_chunks: int = 5
    file_store_dir: str = "data/files"
    chunk_size: int = 512       # tokens per chunk
    chunk_overlap: int = 50     # tokens of overlap between chunks

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env"}


settings = Settings()
