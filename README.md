# Tutor Chatbot

A Socratic-style AI tutor for CS and programming concepts. Instead of giving you the answer directly, it guides you toward understanding through analogies, hints, and leading questions.

## How it works

The backend uses a LangGraph state machine with 5 nodes that run on every message:

1. **extract_topic** — extracts the concept the student wants to learn, runs once per session
2. **assess_understanding** — evaluates the conversation and decides if the student understood or is still struggling
3. **choose_strategy** — determines the response approach based on the current hint level
4. **respond** — generates the tutor's reply using the chosen strategy
5. **congratulate** — triggered when the student demonstrates understanding

The hint system escalates with each wrong or confused answer:

| Level | Strategy |
|-------|----------|
| 0 | Real-world analogy + broad question |
| 1 | Narrower hint pointing at the gap |
| 2 | Leading question that almost gives it away |
| 3 | Full answer revealed with explanation |

## Tech Stack

**Backend:** Python, FastAPI, LangChain, LangGraph, Groq API

**Frontend:** Next.js, Tailwind CSS

## Prerequisites

- Python 3.10+
- Node.js
- Groq API key (free at console.groq.com)

## Running locally

### Backend

```bash
cd backend
python3 -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Create a `.env` file in the backend folder:
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_key_here

Start the server:

```bash
python -m uvicorn app:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Project Structure
TutorChatAgent/
├── backend/
│   ├── app.py        # FastAPI app + LangGraph agent
│   └── requirements.txt
└── frontend/
└── app/
├── page.tsx
└── components/
└── TutorChat.tsx