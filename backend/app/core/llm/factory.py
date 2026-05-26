# app/core/llm/factory.py
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

def get_llm(provider: str | None = None):
    provider = provider or settings.llm_provider
    if provider == "groq":
        return ChatGroq(api_key=settings.groq_api_key, model="llama-3.3-70b-versatile", temperature=0.4)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(api_key=settings.google_api_key, model="gemini-2.5-flash-lite", temperature=0.4)
    else:
        raise ValueError(f"Unknown provider: {provider}")