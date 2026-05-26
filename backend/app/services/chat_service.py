# app/services/chat_service.py
import json
import asyncio
from fastapi.responses import StreamingResponse
from app.core.agent.graph import build_graph
from app.services.session_service import session_service
from app.api.models.request import ChatRequest

class ChatService:
    def __init__(self):
        self.graph = build_graph()
    
    async def stream_response(self, request: ChatRequest):
        session_id = request.session_id
        
        session = session_service.get_session(session_id)
        if not session:
            session = {
                "messages": [],
                "topic": "",
                "hint_level": 0,
                "misconception": "",
                "resolved": False,
            }
        
        updated_messages = request.history + [{"role": "user", "content": request.message}]
        rag_context = ""  # placeholder, no RAG for now
        
        state = {
            "topic": request.topic or session["topic"],
            "hint_level": request.hint_level or session["hint_level"],
            "misconception": request.misconception or session["misconception"],
            "resolved": request.resolved or session["resolved"],
            "messages": updated_messages,
            "rag_context": rag_context,
            "response": "",
        }
        
        # Invoke graph (returns final state with full response)
        final_state = self.graph.invoke(state)
        print(f"Final state: {final_state}")
        full_answer = final_state.get("response", "I'm not sure how to respond.")
        
        # Update session with full conversation
        updated_session = {
            "messages": updated_messages + [{"role": "assistant", "content": full_answer}],
            "topic": final_state.get("topic", ""),
            "hint_level": final_state.get("hint_level", 0),
            "misconception": final_state.get("misconception", ""),
            "resolved": final_state.get("resolved", False),
        }
        session_service.update_session(session_id, updated_session)
        
        # Generator that streams the answer in small chunks (simulated streaming)
        async def event_generator():
            # Split the answer into words or characters (here we use words)
            chunks = full_answer.split()
            # Send each word as a token (add space back)
            for i, word in enumerate(chunks):
                token = word + (" " if i < len(chunks)-1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.03)  # small delay to simulate typing
            # Send final state
            yield f"data: {json.dumps({'type': 'state', **final_state})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")