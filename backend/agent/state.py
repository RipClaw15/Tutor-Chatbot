from langchain_core.messages import BaseMessage

from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


HINT_STRATEGIES = {
    0:"Use a real-world analogy to explain the concept. Then ask a broad open question to probe understanding. Do NOT give the answer.",
    1:"Give a narrower hint that points directly at the gap in their understanding. Ask a more specific follow-up question. Do NOT give the answer.",
    2:"Ask a leading question that almost gives the answer away. The user should be able to complete the thought themselves. Do NOT give the answer.",
    3:"The user has struggled enough. Clearly reveal and explain the answer. Then summarize the key insight they should take away.",
}

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
    session_id: str = Field(default="")
    provider: str = Field(default="groq")