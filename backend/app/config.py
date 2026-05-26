# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "CS Tutor API"      # add this line
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str
    google_api_key: Optional[str] = None
    google_genai_use_vertexai: bool = False
    judge0_api_key: Optional[str] = None
    judge0_url: str = "https://judge0-ce.p.rapidapi.com"
    chroma_persist_dir: str = "./chroma_db"
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",   
        "https://your-vercel-app.vercel.app"
    ]
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()