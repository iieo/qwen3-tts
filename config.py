from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    MODEL_NAME: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    VOICES_DIR: Path = Path("./voices")
    
    # BentoML scaling config
    NUM_WORKERS: int = 1
    NUM_GPUS: int = 0  # Set to 1+ for NVIDIA GPU support
    
    # API context
    PORT: int = 3000

settings = Settings()
