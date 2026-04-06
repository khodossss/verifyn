# Backend

## Overview

The backend is a **FastAPI** application that exposes the fact-checking agent as a REST API. It supports both blocking and streaming (SSE) endpoints, with mode selection (`fast`/`precise`) controlling the agent's reasoning depth.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Service info + endpoint listing |
| `GET /health` | Health check (`{"status": "ok"}`) |
| `GET /history` | Recent fact-check history (from SQLite) |
| `POST /analyze` | Blocking fact-check — returns full `FactCheckResult` |
| `POST /analyze/stream` | SSE streaming — yields progress events in real-time |

## Request Schema

```json
{
  "text": "News article or claim to fact-check (min 10 chars)",
  "verbose": false,
  "mode": "fast"  // "fast" (low reasoning) or "precise" (medium reasoning)
}
```

## Mode Mapping

| Mode | Reasoning Effort | Typical Latency | Use Case |
|------|-----------------|-----------------|----------|
| `fast` | `low` | 15-30s | Quick checks, high-volume |
| `precise` | `medium` | 45-120s | Thorough verification |

## Streaming (SSE)

The `/analyze/stream` endpoint wraps the agent's synchronous generator in an async queue, yielding Server-Sent Events:

```
data: {"type": "thinking", "text": "Let me search for this claim..."}

data: {"type": "tool_call", "tool": "web_search", "query": "claim text", "label": "Searching the web"}

data: {"type": "tool_result", "tool": "web_search"}

data: {"type": "extracting"}

data: {"type": "result", "data": {"verdict": "FAKE", "confidence": 0.92, ...}}
```

The agent runs in a thread executor (`run_in_executor`) since LangGraph's streaming is synchronous.

## Error Handling

- **422** — Invalid input (too short, empty, wrong type)
- **500** — Agent failure (LLM timeout, extraction failure) — includes error detail in response

## CORS

All origins allowed (`*`) for development. In production, restrict to the website domain.

## Deployment

The backend runs as a Docker container (see [INFRASTRUCTURE.md](INFRASTRUCTURE.md)):

```dockerfile
FROM python:3.11-slim
# ... install dependencies ...
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Port 8000 is bound to `127.0.0.1` by default (not exposed publicly). The Nginx frontend proxies `/api/*` to the backend.
