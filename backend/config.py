import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    
    elevenlabs_api_key: Optional[str] = Field(None, alias="ELEVENLABS_API_KEY")
    deepgram_api_key: Optional[str] = Field(None, alias="DEEPGRAM_API_KEY")

    livekit_url: Optional[str] = Field(None, alias="LIVEKIT_URL")
    livekit_api_key: Optional[str] = Field(None, alias="LIVEKIT_API_KEY")
    livekit_api_secret: Optional[str] = Field(None, alias="LIVEKIT_API_SECRET")

    groq_model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7

    data_path: str = "./articulo.pdf"
    persist_directory: str = "./chroma_db"

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
