from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    current_model: str

    class Config:
        # Read settings from .env file
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
