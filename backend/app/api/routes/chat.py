from fastapi import APIRouter, Response
from app.api.models.request import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter()

@router.options("/chat")
async def options_chat():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"Received: topic={request.topic}, hint_level={request.hint_level}, message={request.message}")
    service = ChatService()
    response = await service.stream_response(request)
    return response