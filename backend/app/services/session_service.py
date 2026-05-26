# app/services/session_service.py
from typing import Dict, Any
import uuid

class SessionService:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._rag_sessions: Dict[str, str] = {}
    
    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        if session_id in self._sessions:
            self._sessions[session_id].update(updates)
        else:
            self._sessions[session_id] = updates
    
    def set_rag_collection(self, session_id: str, collection_name: str):
        self._rag_sessions[session_id] = collection_name
    
    def get_rag_collection(self, session_id: str) -> str | None:
        return self._rag_sessions.get(session_id)

session_service = SessionService()