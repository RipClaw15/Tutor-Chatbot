# backend/test_graph.py
from app.core.agent.graph import build_graph

def main():
    graph = build_graph()
    state = {
        "topic": "",
        "hint_level": 0,
        "misconception": "",
        "resolved": False,
        "messages": [{"role": "user", "content": "What is recursion?"}],
        "rag_context": ""
    }
    result = graph.invoke(state)
    print("Response:", result.get("response", "No response"))

if __name__ == "__main__":
    main()