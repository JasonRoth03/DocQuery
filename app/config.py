from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_user: str = "docquery"
    postgres_password: str = "docquery"
    postgres_db: str = "docquery"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o"

    top_k_chunks: int = 5
    file_store_dir: str = "data/files"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env"}


settings = Settings()
