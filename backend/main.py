"""FastAPI backend for the fake-news detection agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from functools import partial

from dotenv import load_dotenv

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verifyn.api")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Make the agent package importable when the process starts from this directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import FactCheckResult, analyze_news, analyze_news_stream  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFFORT_MAP = {"fast": "low", "precise": "medium"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Fake News Detection API",
    description=(
        "AI-powered fact-checking agent. Follows an 8-step verification methodology: "
        "claim extraction → primary source → date context → lateral reading → "
        "confirmations → old-content check → professional fact-checkers → classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article or claim to fact-check")
    verbose: bool = Field(False, description="Stream agent steps to server stdout")
    mode: str = Field("fast", description="Inference mode: 'fast' (low reasoning) or 'precise' (high reasoning)")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["info"])
async def root():
    return {
        "service": "Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Fact-check a news article or claim",
            "GET /health": "Health check",
        },
    }


@app.get("/health", tags=["info"])
async def health():
    return {"status": "ok"}


@app.get("/history", tags=["info"])
async def history(limit: int = 50):
    """Return recent fact-check query history."""
    from agent.db import get_query_history

    return get_query_history(limit=limit)


@app.post("/analyze/stream", tags=["fact-check"])
async def analyze_stream(request: AnalyzeRequest):
    """SSE endpoint — streams agent progress events, then the final result."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    logger.info("POST /analyze/stream: mode=%s text_len=%d", request.mode, len(request.text))
    reasoning_effort = _EFFORT_MAP.get(request.mode, "low")

    loop = asyncio.get_event_loop()

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        def run():
            try:
                for event in analyze_news_stream(request.text, reasoning_effort=reasoning_effort):
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "message": str(exc)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.get_event_loop().run_in_executor(None, run)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze", response_model=FactCheckResult, tags=["fact-check"])
async def analyze(request: AnalyzeRequest):
    """Submit a news text for AI fact-checking.

    The agent follows an 8-step verification methodology and returns a
    structured FactCheckResult with verdict, confidence, evidence, and
    step-by-step reasoning.
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    logger.info("POST /analyze: mode=%s text_len=%d", request.mode, len(request.text))
    reasoning_effort = _EFFORT_MAP.get(request.mode, "low")

    loop = asyncio.get_event_loop()
    try:
        result: FactCheckResult = await loop.run_in_executor(
            None,
            partial(analyze_news, request.text, verbose=request.verbose, reasoning_effort=reasoning_effort),
        )
    except Exception as exc:
        logger.error("POST /analyze failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("POST /analyze done: verdict=%s confidence=%.2f", result.verdict.value, result.confidence)
    return result
