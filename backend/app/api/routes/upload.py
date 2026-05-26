# app/api/routes/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.upload_service import UploadService
from app.services.session_service import session_service

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")
    
    upload_service = UploadService()
    session_id, collection_name = await upload_service.process_upload(file)
    session_service.set_rag_collection(session_id, collection_name)
    
    return {"session_id": session_id, "message": "Document uploaded and indexed"}