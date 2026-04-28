"use client";


import {useState, useRef, useEffect} from "react";  

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface TutorState {
    topic: string;
    hint_level: number;
    misconceptions: string;
    resolved: boolean;
}

const HINT_LABELS: Record<number, {label: string; color: string}> = {
    0: {label: "Analogy", color: "gray"},
    1: {label: "Hint", color: "blue"},
    2: {label: "Leading Q", color: "green"},
    3: {label: "Revealing", color: "orange"},
};

export default function TutorChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [state, setState] = useState<TutorState>({
    topic: "",
    hint_level: 0,
    misconceptions: "",
    resolved: false,
  });
  const [sessionId, setSessionId]     = useState("");
  const [uploading, setUploading]     = useState(false);
  const [docUploaded, setDocUploaded] = useState(false);
  const fileInputRef                  = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({behavior: "smooth"});
  }, [messages]);


async function send() {
    const text = input.trim();
    if (!text || streaming) return;

    const userMessage: Message = { role: "user", content: text };
    const updatedMessages = [...messages, userMessage];

    setMessages(updatedMessages);
    setInput("");
    setStreaming(true);

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    abortRef.current = new AbortController();

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          message: text,
          history: messages,
          topic: state.topic,
          hint_level: state.hint_level,
          misconception: state.misconceptions,
          resolved: state.resolved,
          session_id: sessionId,
        }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer    = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        console.log("buffer:", buffer);
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;

          try {
            const event = JSON.parse(raw);

            if (event.type === "token") {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  role: "assistant",
                  content: next[next.length - 1].content + event.content,
                };
                return next;
              });
            }

            if (event.type === "state") {
              setState({
                topic:         event.topic,
                hint_level:    event.hint_level,
                misconceptions: event.misconception,
                resolved:      event.resolved,
              });
            }

            if (event.type === "error") {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = {
                  role: "assistant",
                  content: `Error: ${event.content}`,
                };
                return next;
              });
            }
          } catch {
            // malformed SSE line, skip
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: "Connection error. Is the backend running?",
          };
          return next;
        });
      }
    } finally {
      setStreaming(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function reset() {
    abortRef.current?.abort();
    setMessages([]);
    setState({ topic: "", hint_level: 0, misconceptions: "", resolved: false });
    setInput("");
    setStreaming(false);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();
      setSessionId(data.session_id);
      setDocUploaded(true);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Document "${file.name}" uploaded successfully. I will now use it to help answer your questions.`,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Failed to upload document. Please try again." },
      ]);
    } finally {
      setUploading(false);
    }
  }

  // JSX

  const hint = HINT_LABELS[state.hint_level];

  return (
    <div className="flex flex-col h-screen bg-neutral-950 text-zinc-100">

      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <span className="text-lg font-semibold text-white">CS Tutor</span>
          {state.topic && (
            <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">
              {state.topic}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          {state.topic && (
            <div className="flex items-center gap-2 text-xs">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: hint.color }}
              />
              <span className="text-zinc-400">{hint.label}</span>
            </div>
          )}
          <button
            onClick={reset}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            new session
          </button>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <p className="text-zinc-500 text-sm max-w-sm">
              Ask me to explain any CS concept. I will guide you to the answer
              with analogies and questions instead of just telling you.
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {["Explain recursion", "How does a hash table work?", "What is Big O notation?"].map((q) => (
                <button
                  key={q}
                  onClick={() => { setInput(q); inputRef.current?.focus(); }}
                  className="text-xs px-3 py-1.5 rounded border border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-zinc-800 text-zinc-100 rounded-br-sm"
                  : "bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-bl-sm"
              }`}
            >
              {msg.content}
              {streaming && i === messages.length - 1 && msg.role === "assistant" && (
                <span className="inline-block w-0.5 h-3.5 bg-zinc-400 ml-0.5 animate-pulse align-middle" />
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-zinc-800">
        <div className="flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question or answer mine..."
            rows={5}
            className="flex-1 resize-y bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors"
            style={{ minHeight: "80px", maxHeight: "300px" }}
          />

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleUpload}
            className="hidden"
            aria-label="Upload PDF document"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className={`px-4 py-3 rounded-xl text-sm font-medium transition-colors border ${
              docUploaded
                ? "border-green-500 text-green-500"
                : "border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
            } disabled:opacity-30 disabled:cursor-not-allowed`}
          >
            {uploading ? "..." : docUploaded ? "Doc ✓" : "Upload"}
          </button>

          <button
            onClick={send}
            disabled={streaming || !input.trim()}
            className="px-4 py-3 rounded-xl bg-white text-zinc-950 text-sm font-medium hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {streaming ? "..." : "Send"}
          </button>
        </div>
        <p className="text-xs text-zinc-600 mt-2">
          Enter to send · Shift+Enter for new line
        </p>
      </div>

    </div>
  );
}

