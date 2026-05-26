import re
from app.utils.formatters import load_prompt, format_conversation_history
from app.core.llm.factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from app.utils.code_formatter import format_code_blocks   # <-- new import

def response_node(state: dict) -> dict:
    llm = get_llm()
    raw_prompt = load_prompt("respond")

    replacements = {
        "{topic}": state.get("topic", "general programming"),
        "{hint_level}": str(state.get("hint_level", 0)),
        "{misconception}": state.get("misconception", ""),
        "{rag_context}": state.get("rag_context", ""),
        "{history}": format_conversation_history(state.get("messages", [])),
    }
    prompt = raw_prompt
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Generate response.")])
    raw_content = response.content

    # Apply code formatter to ensure proper markdown code blocks
    fixed_content = format_code_blocks(raw_content)

    return {"response": fixed_content}