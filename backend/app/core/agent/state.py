# app/core/agent/state.py
from typing import TypedDict, List, Dict, Any

class TutorState(TypedDict):
    topic: str
    hint_level: int
    misconception: str
    resolved: bool
    messages: List[Dict[str, str]]
    rag_context: str