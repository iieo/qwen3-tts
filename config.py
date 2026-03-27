from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    MODEL_NAME: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    VOICES_DIR: Path = Path("./voices")
    
    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 3000

settings = Settings()
