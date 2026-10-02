from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    openai_api_key: str = ""
    openai_realtime_model: str = "gpt-realtime-2.1"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    database_url: str = "sqlite:///./mavic.db"
    redis_url: str = "redis://localhost:6379/0"

    default_user_id: str = "usr_demo"
    default_user_name: str = "Nitin"
    app_env: str = "development"
    cors_origins: str = "http://localhost:3000"
    jwt_secret: str = ""

    frame_ttl_seconds: int = 90
    max_frames_per_turn: int = 5
    embedding_dimensions: int = 1536


@lru_cache
def get_settings() -> Settings:
    return Settings()
