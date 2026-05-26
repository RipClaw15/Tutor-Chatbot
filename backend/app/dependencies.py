# app/dependencies.py
from app.services.session_service import session_service

def get_session_service():
    return session_service