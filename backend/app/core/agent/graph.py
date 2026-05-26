# app/core/agent/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

# Import node functions (to be implemented in separate files)
from .nodes.extract_topic import extract_topic_node
from .nodes.assess_understanding import assess_understanding_node
from .nodes.response import response_node

# --- State Definition ---
class TutorState(TypedDict):
    topic: str
    hint_level: int
    misconception: str
    resolved: bool
    messages: List[Dict[str, str]]
    rag_context: str
    # Optional: final response to be streamed
    response: str

# --- Graph Builder ---
def build_graph():
    workflow = StateGraph(TutorState)

    # Add nodes
    workflow.add_node("extract_topic", extract_topic_node)
    workflow.add_node("assess_understanding", assess_understanding_node)
    workflow.add_node("respond", response_node)

    # Set entry point
    workflow.set_entry_point("extract_topic")

    # Add edges
    workflow.add_edge("extract_topic", "assess_understanding")
    workflow.add_edge("assess_understanding", "respond")
    workflow.add_edge("respond", END)

    # Compile and return
    return workflow.compile()