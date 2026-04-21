from typing import TypedDict, List, Annotated
from unittest import result
from urllib import response

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import os

load_dotenv()

HINT_STRATEGIES = {
    0:"Use a real-world analogy to explain the concept. Then ask a broad open question to probe understanding. Do NOT give the answer.",
    1:"Give a narrower hint that points directly at the gap in their understanding. Ask a more specific follow-up question. Do NOT give the answer.",
    2:"Ask a leading question that almost gives the answer away. The user should be able to complete the thought themselves. Do NOT give the answer.",
    3:"The user has struggled enough. Clearly reveal and explain the answer. Then summarize the key insight they should take away.",
}

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    if provider == "ollama":
        return ChatOllama(model=model, temperature=0.4)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0.4)
    
    raise ValueError(f"Unsupported LLM provider: {provider}")

llm = get_llm()

class TutorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    hint_level: int
    misconception: str
    resolved: bool


class ChatRequest(BaseModel):
    message: str
    topic: str = ""
    hint_level: int = 0 
    misconception: str  = ""
    resolved: bool = False
    history: List[dict] = []

def extract_topic_node(state: TutorState) -> dict:
    if state["topic"]:
        return {}
    first_message = state["messages"][0].content

    prompt = f"""Extract the CS/programming topic the user want to learn about.
    Return ONLY the topic name, nothing else. 2-5 words max.and

    User message: {first_message}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"topic": response.content.strip()  }
    
def assess_understanding_node(state: TutorState) -> dict:
    if len(state["messages"]) < 2:
        return {"hint_level": 0, "misconception": "", "resolved": False}
    
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
        for m in state["messages"]
    )

    prompt = f"""You are evaluating a student learning about: {state['topic']}

        Conversation so far:
        {history_text}

        Current hint level: {state['hint_level']} (0=analogy, 1=hint, 2=leading-Q, 3=reveal)

        Respond in JSON with exactly these fields:
        {{
        "resolved": true/false,
        "hint_level": 0-3,
        "misconception": "..."
        }}

        Rules:
        - IMPORTANT: Re-evaluate from scratch based on the full conversation. Do not assume previous misconceptions still exist if the user has corrected them.
        - If the student's latest message contains correct, working code or a correct explanation, set resolved=true immediately.
        - If the student says "yes" or confirms understanding after a leading question, consider setting resolved=true.
        - Only increase hint_level if the user is still clearly confused after the previous hint.
        - Never decrease hint_level.
        - misconception should be "" if resolved=true.
        - Return ONLY the JSON object, no other text."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        if raw.startswith("```json"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw.strip())

        return {
            "resolved": bool(data.get("resolved", False)),
            "hint_level": max(state["hint_level"], int(data.get("hint_level", state["hint_level"]))),
            "misconception": data.get("misconception", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "resolved": False,
            "hint_level": min(state["hint_level"] + 1, 3),
            "misconception": state["misconception"],
        }


def choose_strategy_node(state: TutorState)-> dict:
    return {}

def respond_node(state: TutorState) -> dict:
    strategy = HINT_STRATEGIES[state["hint_level"]]

    misconception_note = (
        f"The student's specific misconception seems to be: {state['misconception']}." 
        if state["misconception"] 
        else "You don't know yet their specific misconception."
    )

    system_prompt = f"""You are a Socratic CS tutor teaching {state['topic']}
    Your current strategy is: {strategy}
    {misconception_note}

    Rules:
    -  Be concice and conversational (3-6 sentences max).
    - Never lecture. Guide with questions and analogies.
    - {'You may now reveal the answer fully and clearly.' if state['hint_level'] == 3 else 'Do NOT reveal the answer yet.'}
    - If resolved = True, warmly congratulate the student and reinforce the key insight.
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def congratulate_node(state: TutorState) -> dict:
    system_prompt = f"""You are a Socratic CS tutor. The student has just succesfully understood: {state['topic']}

    Give a warm, brief (2-3 sentence) congratulation. Reinforce the key insight they discovered.
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

def route_after_assessment(state: TutorState) -> str:
    if state["resolved"]:
        return "congratulate"
    else:
        return "choose_strategy"    
    
def build_graph() -> StateGraph:
    workflow = StateGraph(TutorState)

    workflow.add_node("extract_topic", extract_topic_node)
    workflow.add_node("assess_understanding", assess_understanding_node)
    workflow.add_node("choose_strategy", choose_strategy_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("congratulate", congratulate_node)

    workflow.set_entry_point("extract_topic")
    workflow.add_edge("extract_topic", "assess_understanding")
    # workflow.add_edge("extract_topic", "choose_strategy")
    workflow.add_conditional_edges("assess_understanding", route_after_assessment,
                                   {"congratulate": "congratulate",
                                    "choose_strategy": "choose_strategy"},)
    workflow.add_edge("choose_strategy", "respond")
    workflow.add_edge("respond", END)
    workflow.add_edge("congratulate", END)

    return workflow.compile()

graph = build_graph()

def build_assessment_graph() -> StateGraph:
    workflow = StateGraph(TutorState)

    workflow.add_node("extract_topic", extract_topic_node)
    workflow.add_node("assess_understanding", assess_understanding_node)

    workflow.set_entry_point("extract_topic")
    workflow.add_edge("extract_topic", "assess_understanding")
    workflow.add_edge("assess_understanding", END)

    return workflow.compile()

assessment_graph = build_assessment_graph()

app = FastAPI(title="CS Tutor Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
async def chat(request: ChatRequest):

    history = deserialize_history(request.history)
    new_message = HumanMessage(content=request.message)

    initial_state: TutorState = {
        "messages": history + [new_message],
        "topic": request.topic,
        "hint_level": request.hint_level,
        "misconception": request.misconception,
        "resolved": request.resolved,
    }

    async def event_stream():
        try:
            # Step 1: run extract_topic and assess_understanding
            assessment_state = assessment_graph.invoke(initial_state)

            # Step 2: build the streaming prompt directly
            strategy = HINT_STRATEGIES[assessment_state["hint_level"]]
            misconception_note = (
                f"The student's specific misconception is: {assessment_state['misconception']}"
                if assessment_state["misconception"]
                else "You don't yet know their specific misconception."
            )

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