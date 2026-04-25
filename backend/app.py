from typing import TypedDict, List, Annotated
from unittest import result
from urllib import response

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import os

load_dotenv()
limiter = Limiter(key_func=get_remote_address)

HINT_STRATEGIES = {
    0:"Use a real-world analogy to explain the concept. Then ask a broad open question to probe understanding. Do NOT give the answer.",
    1:"Give a narrower hint that points directly at the gap in their understanding. Ask a more specific follow-up question. Do NOT give the answer.",
    2:"Ask a leading question that almost gives the answer away. The user should be able to complete the thought themselves. Do NOT give the answer.",
    3:"The user has struggled enough. Clearly reveal and explain the answer. Then summarize the key insight they should take away.",
}


# This function initializes the language model based on environment variables. It supports multiple providers (currently "ollama" and "groq") and allows you to specify the model name and temperature. By abstracting this logic into a function, we can easily switch between different LLM providers or models without changing the core logic of our application.
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

# This is the state schema for our tutor agent. It includes the conversation history (messages), the current topic being discussed, the hint level (which determines the strategy for the next hint), any specific misconception identified, and whether the student's confusion has been resolved. The messages field is annotated with add_messages to allow the graph to automatically append new messages to the history as we generate responses.
class TutorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    hint_level: int
    misconception: str
    resolved: bool


# This is the request body schema for the /chat endpoint. It includes the user's new message, as well as optional fields for the current topic, hint level, misconception, resolved status, and conversation history. The history is a list of message objects that represent the conversation so far, which allows the backend to reconstruct the full context when processing the new message.
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    topic: str = Field(default="", max_length=100)
    hint_level: int = Field(default=0, ge=0, le=3) 
    misconception: str  = Field(default="", max_length=500)
    resolved: bool = False
    history: List[dict] = Field(default=[],max_length=50)

def extract_topic_node(state: TutorState) -> dict:
    
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_message = user_messages[-1].content

    prompt = f"""   The user wants to learn about a CS or programming concept.
                    Extract the topic from their message.

                    Examples:
                        - "explaint recursion" -> "recursion"
                        - "what is a binary search tree?" -> "binary search tree"
                        - "i keep hearing about transformers in ai, what are they?" -> "transformers"
                        - "how do hash tables work?" -> "hash tables"
                        - "hello there!" -> "unknown"
                        - "i want to learn about machine learning" -> "machine learning"
                    If the message contains ANY reference to a CS or programming concept, return that concept.
                    Only return "unknown" if the message is purely social with zero technical content.

                    Return ONLY the topic name or "unknown". Nothing else.

                 User message: {latest_message}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    print("extracted topic:", result)
    if result.lower() == "same":
        return {}

    print("extracted topic:", result)
    # Post-process the response to handle common variations of "unknown" and ensure we have a clean topic string.
    topic = result.lower()
    if topic in ["unknown", "none", "no topic", "not mentioned"]:
        topic = "unknown"
    return {"topic": topic}
    
def assess_understanding_node(state: TutorState) -> dict:

    # If we don't know the topic yet, we can't really assess their understanding, so we'll just return the default state with no misconceptions and hint level 0. The tutor will then prompt them to clarify the topic in the next step.
    if state["topic"] == "unknown":
        return {"hint_level": 0, "misconception": "", "resolved": False}
    
    if len(state["messages"]) < 2:
        return {"hint_level": 0, "misconception": "", "resolved": False}
    
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Tutor'}: {m.content}"
        for m in state["messages"]
    )
    # The prompt should instruct the LLM to analyze the conversation history and determine if the student's latest message indicates they have resolved their confusion, or if they still have misconceptions. It should also decide what the next hint level should be based on the student's current state of understanding.
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
        - Iincrease hint_level if the user is still clearly confused after the previous hint.
        - If the user says 'I don't know' or 'I have no idea' two or more times in a row, increase hint_level immediately.
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

    if state["topic"] == "unknown":
        lines = []
        for m in state["messages"]:
            role = "User" if isinstance(m, HumanMessage) else "Tutor"
            lines.append(f"{role}: {m.content}")
        history_text = "\n".join(lines)

        system_prompt = f"""You are a friendly CS tutor. The user is asking about something, but they haven't mentioned any specific CS topic yet.

        Conversation so far:
        {history_text}

        Politely ask them to clarify what specific CS topic or concept they want to learn about."""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}
    
    strategy = HINT_STRATEGIES[state["hint_level"]]

    # If we have a specific misconception identified, include that in the prompt to help the tutor target their response. If not, just proceed without it.
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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute")
async def chat(request: Request, body:ChatRequest):

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

    # Handle unknown topic before defining event_stream
    if assessment_state["topic"] == "unknown":
        async def unknown_stream():

            
            async for chunk in llm.astream([SystemMessage(content="You are a CS tutor. The student hasn't told you what they want to learn yet. Greet the student and politely ask them what CS or programming concept they'd like to explore today.")]):
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