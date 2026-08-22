import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    
    elevenlabs_api_key: Optional[str] = Field(None, alias="ELEVENLABS_API_KEY")
    deepgram_api_key: Optional[str] = Field(None, alias="DEEPGRAM_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, alias="HUGGINGFACE_API_KEY")
    firebase_storage_bucket: Optional[str] = Field(None, alias="FIREBASE_STORAGE_BUCKET")

    livekit_url: Optional[str] = Field(None, alias="LIVEKIT_URL")
    livekit_api_key: Optional[str] = Field(None, alias="LIVEKIT_API_KEY")
    livekit_api_secret: Optional[str] = Field(None, alias="LIVEKIT_API_SECRET")

    groq_model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7

    data_path: str = "./articulo.pdf"
    persist_directory: Optional[str] = None

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    audio_retention_hours: int = 24

    @property
    def allowed_cors_origins(self) -> list[str]:
        """Return explicitly configured browser origins, never a credentialed wildcard."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
