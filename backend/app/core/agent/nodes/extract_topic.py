from app.utils.formatters import load_prompt
from app.core.llm.factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

def extract_topic_node(state: dict) -> dict:
    if state.get("topic"):
        return {}  # already have topic

    llm = get_llm()
    history = state.get("messages", [])
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = load_prompt("extract_topic").format(history=history_text)
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Extract the topic.")])
    topic = response.content.strip()
    return {"topic": topic}