from typing import TypedDict, List, Annotated
from unittest import result
from urllib import response

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from agent.graph import assessment_graph, get_llm
from agent.state import TutorState, ChatRequest, HINT_STRATEGIES
from agent.tools import execute_code, detect_language, contains_code, extract_code

from agent.rag.indexer import build_index
from agent.rag.retriever import get_relevant_context

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field
from dotenv import load_dotenv

import json
import os
import uuid
import tempfile

load_dotenv()
limiter = Limiter(key_func=get_remote_address)




app = FastAPI(title="CS Tutor Agent")
sessions: dict = {}  # In-memory session store, keyed by session ID
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = json.loads(os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def deserialize_history(history: List[dict]) -> List[BaseMessage]:
    messages = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "tutor":
            messages.append(AIMessage(content=content))
    return messages

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body:ChatRequest):

    llm = get_llm(body.provider)

    history = deserialize_history(body.history)
    new_message = HumanMessage(content=body.message)

    initial_state: TutorState = {
        "messages": history + [new_message],
        "topic": body.topic,
        "hint_level": body.hint_level,
        "misconception": body.misconception,
        "resolved": body.resolved,
    }

    assessment_state = assessment_graph.invoke(initial_state)

    # RAG retrieval if document was uploaded
    rag_context = ""
    if body.session_id and body.session_id in sessions:
        rag_context = get_relevant_context(
            sessions[body.session_id],
            body.message
        )

    # Handle unknown topic before defining event_stream
    if assessment_state["topic"] == "unknown":

        # If student has uploaded a document, try to answer about it
        if body.session_id and body.session_id in sessions:
            rag_context = get_relevant_context(sessions[body.session_id], body.message)
            async def doc_stream():
                prompt = f"""The student uploaded a document and is asking: {body.message}
                
    Relevant content from their document:
    {rag_context}

    Answer their question based on the document content. Be concise and helpful."""
                async for chunk in llm.astream([HumanMessage(content=prompt)]):
                    token = chunk.content
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                yield f"data: {json.dumps({'type': 'state', 'topic': '', 'hint_level': 0, 'misconception': '', 'resolved': False})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(doc_stream(), media_type="text/event-stream")
        
        # No document uploaded, ask what they want to learn
        async def unknown_stream():

            
            async for chunk in llm.astream([HumanMessage(
                content="You are a CS tutor. The student hasn't told you what they want to learn yet. " \
                "Greet the student and politely ask them what CS or programming concept they'd like to explore today.")]):
                token = chunk.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'state', 'topic': '', 'hint_level': 0, 'misconception': '', 'resolved': False})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(unknown_stream(), media_type="text/event-stream")



    async def event_stream():
        try:
            # Step 1: run extract_topic and assess_understanding
            
            print("assessment topic:", assessment_state["topic"])

            # Execute code if detected in the latest message

            misconception_note = ""
            
            if contains_code(body.message):
                print("Code detected in the message, extracting and executing...")
                code = extract_code(body.message)
                language = detect_language(code)
                code_output = await execute_code(code, language)
                print("Code execution output:", code_output)

                asking_for_output = any(phrase in body.message.lower() for phrase in [
                    "what is the output",
                    "what will this output",
                    "what does this print",
                    "what is the result",
                    "run this",
                    "execute this",
                ])

                if asking_for_output:
                    misconception_note += f"\n\nIMPORTANT: The student is directly asking for the output. The code was executed and produced:\n{code_output}\nTell them the output directly. Do not ask questions."
                else:
                    misconception_note += f"\n\nThe student's code was executed and produced:\n{code_output}\nUse this to give more accurate feedback."
            else:
                code_output = ""

            # Step 2: build the streaming prompt directly
            strategy = HINT_STRATEGIES[assessment_state["hint_level"]]
            misconception_note = (
                f"The student's specific misconception is: {assessment_state['misconception']}"
                if assessment_state["misconception"]
                else "You don't yet know their specific misconception."
            )

            if code_output:
                misconception_note += f"\n\nThe student's code was executed and produced this result:\n{code_output}\nUse this to give more accurate feedback."

            # RAG context
            rag_context = ""
            if body.session_id and body.session_id in sessions:
                rag_context = get_relevant_context(
                    sessions[body.session_id],
                    body.message
                )

            rag_note = (
                f"\n\nRelevant context from the student's uploaded document:\n{rag_context}"
                if rag_context
                else ""
            )

            misconception_note = misconception_note + rag_note

            if assessment_state["resolved"]:
                system_content = f"""You are a Socratic CS tutor. The student has just successfully understood: {assessment_state['topic']}
                                     Give a warm, brief (2-3 sentence) congratulation. Reinforce the key insight they discovered."""
            else:
                system_content = f"""You are a Socratic CS tutor teaching: {assessment_state['topic']}

                                     Your current strategy: {strategy}

                                     {misconception_note}

                                     Rules:
                                     - Be concise and conversational (3-6 sentences max).
                                     - Never lecture. Guide with questions and analogies.
                                     - {'You may now reveal the answer fully and clearly.' if assessment_state['hint_level'] == 3 else 'Do NOT give the direct answer.'}"""

            if body.provider == "gemini":
                # Gemini handles SystemMessage poorly, fold into conversation
                messages = assessment_state["messages"]
                # Prepend system instructions to the last user message
                last_msg = messages[-1]
                messages = messages[:-1] + [HumanMessage(content=system_content + "\n\nStudent message: " + last_msg.content)]
            else:
                messages = [SystemMessage(content=system_content)] + assessment_state["messages"]

            # Step 3: stream the response token by token
            full_reply = ""
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_reply += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Step 4: send final state
            yield f"data: {json.dumps({'type': 'state', 'topic': assessment_state['topic'], 'hint_level': assessment_state['hint_level'], 'misconception': assessment_state['misconception'], 'resolved': assessment_state['resolved']})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "llama3.2")}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Accept a PDF upload, index it and return a session_id

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    # Build the index for the uploaded PDF
    vectorstore = build_index(tmp_file_path)

    # Generate a unique session ID
    session_id = str(uuid.uuid4())

    # Store the vectorstore in the session
    sessions[session_id] = vectorstore

    # Return the session ID
    return {"session_id": session_id, "message": "Document indexed successfully."}
