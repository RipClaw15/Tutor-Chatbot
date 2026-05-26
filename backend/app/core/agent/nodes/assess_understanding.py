import json
from app.utils.formatters import load_prompt
from app.core.llm.factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

def assess_understanding_node(state: dict) -> dict:
    llm = get_llm()
    last_user_msg = next((m for m in reversed(state["messages"]) if m["role"] == "user"), None)
    if not last_user_msg:
        return {"hint_level": 0, "misconception": "", "resolved": False}

    prompt = load_prompt("assess_understanding").format(
        topic=state["topic"],
        hint_level=state["hint_level"],
        misconception=state["misconception"],
        student_answer=last_user_msg["content"],
        rag_context=state.get("rag_context", "")
    )
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Assess understanding.")])
    try:
        result = json.loads(response.content)
        return {
            "hint_level": result.get("hint_level", state["hint_level"]),
            "misconception": result.get("misconception", ""),
            "resolved": result.get("resolved", False)
        }
    except:
        # fallback: increase hint level slightly
        return {"hint_level": min(state["hint_level"] + 1, 3)}