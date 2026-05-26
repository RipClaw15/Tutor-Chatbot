from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, health   # remove upload import
from app.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="", tags=["chat"])
# app.include_router(upload.router, prefix="/api", tags=["upload"])   # commented out
app.include_router(health.router, prefix="/api", tags=["health"])

@app.get("/")
async def root():
    return {"message": "CS Tutor API is running"}