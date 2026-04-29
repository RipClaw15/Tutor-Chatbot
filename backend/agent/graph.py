import json
import os

from typing import Dict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from slowapi import Limiter
from .state import TutorState, HINT_STRATEGIES

from .prompts import EXTRACT_TOPIC_PROMPT, ASSESS_UNDERSTANDING_PROMPT, CONGRATS_PROMPT

load_dotenv()



# This function initializes the language model based on environment variables. It supports multiple providers (currently "ollama" and "groq") and allows you to specify the model name and temperature. By abstracting this logic into a function, we can easily switch between different LLM providers or models without changing the core logic of our application.
def get_llm(provider: str = None):
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    if provider == "ollama":
        return ChatOllama(model=model, temperature=0.4)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0.4)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.4)

    raise ValueError(f"Unknown provider: {provider}")

llm = get_llm()

def extract_topic_node(state: TutorState) -> dict:
    
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_message = user_messages[-1].content

    prompt = EXTRACT_TOPIC_PROMPT.format(current_topic=state["topic"], latest_message=latest_message)
    
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
    prompt = ASSESS_UNDERSTANDING_PROMPT.format(
        topic=state["topic"],
        history_text=history_text,
        hint_level=state["hint_level"]
    )

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



def route_after_assessment(state: TutorState) -> str:
    if state["resolved"]:
        return "congratulate"
    else:
        return "choose_strategy"    
    

def build_assessment_graph() -> StateGraph:
    workflow = StateGraph(TutorState)

    workflow.add_node("extract_topic", extract_topic_node)
    workflow.add_node("assess_understanding", assess_understanding_node)

    workflow.set_entry_point("extract_topic")
    workflow.add_edge("extract_topic", "assess_understanding")
    workflow.add_edge("assess_understanding", END)

    return workflow.compile()

assessment_graph = build_assessment_graph()