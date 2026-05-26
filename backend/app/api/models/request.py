# app/api/models/request.py
from pydantic import BaseModel
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]   # each item: {"role": "user" or "assistant", "content": "..."}
    topic: str
    hint_level: int
    misconception: str               # singular, as sent by frontend
    resolved: bool
    session_id: str
    provider: str = "groq"