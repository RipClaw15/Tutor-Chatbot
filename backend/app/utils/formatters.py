# app/utils/formatters.py
from pathlib import Path

PROMPT_DIR = Path(__file__).parent.parent / "core" / "agent" / "prompts"

def load_prompt(name: str) -> str:
    """Load a prompt from a .md file in the prompts directory."""
    with open(PROMPT_DIR / f"{name}.md", "r") as f:
        return f.read()

def format_conversation_history(messages: list[dict]) -> str:
    """Convert list of {role, content} to a readable string."""
    lines = []
    for msg in messages:
        role = "Student" if msg["role"] == "user" else "Tutor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)