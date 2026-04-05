"""Fake-news detection agent.

Flow
----
1. ReAct research phase  — LangGraph prebuilt agent uses tools to gather evidence
   following the 8-step fact-check methodology.  The agent is instructed to end
   its final message with a ```json ... ``` block containing a FactCheckResult.
2. Extraction phase      — Try to parse that JSON block directly with Pydantic.
   If parsing fails, make one fallback LLM call to repair the JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Generator
from datetime import date
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from .models import FactCheckResult
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS

logger = logging.getLogger("verifyn")

# ---------------------------------------------------------------------------
# LLM setup — multi-provider via LLM_PROVIDER env var
# ---------------------------------------------------------------------------
# Supported providers:
#   openai    — OpenAI / Azure OpenAI  (default)
#   anthropic — Anthropic Claude
#   ollama    — Local models via Ollama
#
# Set LLM_PROVIDER + the corresponding API key / model env vars.
# ---------------------------------------------------------------------------

_PROVIDER_ENV = {
    "openai": {"key": "OPENAI_API_KEY", "model": "OPENAI_MODEL", "default_model": "gpt-4o-mini"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-4-20250514"},
    "ollama": {"key": None, "model": "OLLAMA_MODEL", "default_model": "llama3.1"},
}


def _build_llm(temperature: float = 0.0, reasoning_effort: str | None = None) -> BaseChatModel:
    """Build a chat model based on the LLM_PROVIDER environment variable.

    Supported providers: openai (default), anthropic, ollama.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower().strip()
    if provider not in _PROVIDER_ENV:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Supported: {', '.join(_PROVIDER_ENV)}")

    cfg = _PROVIDER_ENV[provider]
    model = os.environ.get(cfg["model"], cfg["default_model"])

    logger.info("Building LLM: provider=%s model=%s temperature=%s", provider, model, temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = dict(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            api_key=os.environ[cfg["key"]],
        )
        if reasoning_effort:
            kwargs["model_kwargs"] = {"reasoning_effort": reasoning_effort}
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            api_key=os.environ[cfg["key"]],
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    raise ValueError(f"Provider {provider!r} not implemented")


# ---------------------------------------------------------------------------
# Extraction helper
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]+?)\s*```", re.IGNORECASE)

REPAIR_SYSTEM = (
    "You are a JSON repair assistant. The user will give you either a broken/incomplete "
    "FactCheckResult JSON or a research narrative written by a fact-checking agent. "
    "Your job: extract or reconstruct a valid FactCheckResult JSON from whatever is provided. "
    "Return ONLY a valid JSON object — no markdown, no explanation, no code fences. "
    "Required fields: verdict, confidence (float 0.0–1.0), confidence_level, manipulation_type, reasoning, summary. "
    "verdict must be one of: REAL, FAKE, PARTIALLY_FAKE, MISLEADING, UNVERIFIABLE, SATIRE, NO_CLAIMS. "
    "confidence_level must be HIGH, MEDIUM, or LOW. "
    "manipulation_type must be one of: NONE, FABRICATED, CONTEXT_MANIPULATION, "
    "OLD_CONTENT_RECYCLED, MISLEADING_HEADLINE, PARTIAL_TRUTH, SATIRE_MISREPRESENTED, "
    "COORDINATED_DISINFO, IMPERSONATION. "
    "If the input contains no search results or evidence (e.g. it is only a plan or introduction), "
    "set verdict=UNVERIFIABLE, confidence=0.1, confidence_level=LOW and explain in reasoning that "
    "no research was completed. Do NOT invent evidence."
)


def _sanitize_json(data: dict) -> dict:
    """Fix common LLM JSON quirks before Pydantic validation."""
    # manipulation_type: "A|B" → take first value
    mt = data.get("manipulation_type", "")
    if isinstance(mt, str) and "|" in mt:
        data["manipulation_type"] = mt.split("|")[0].strip()
    # confidence_level: normalise to uppercase
    cl = data.get("confidence_level")
    if isinstance(cl, str):
        data["confidence_level"] = cl.upper().strip()
    # verdict: normalise to uppercase
    v = data.get("verdict")
    if isinstance(v, str):
        data["verdict"] = v.upper().strip()
    return data


MAX_EXTRACT_ATTEMPTS = 3


def _try_parse(text: str) -> FactCheckResult | None:
    """Try to parse FactCheckResult from text using json_repair. Returns None on failure."""
    from json_repair import repair_json

    candidates: list[str] = []

    # 1. ```json block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(1))

    # 2. First { ... } span containing "verdict"
    start = text.find("{")
    if start != -1:
        candidates.append(text[start:])

    # 3. The whole text
    candidates.append(text)

    for candidate in candidates:
        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, dict) and "verdict" in repaired:
                return FactCheckResult.model_validate(_sanitize_json(repaired))
        except Exception:
            continue
    return None


def _extract_result(research_narrative: str) -> FactCheckResult:
    """Parse FactCheckResult from the agent's narrative.

    Loop up to MAX_EXTRACT_ATTEMPTS times:
      1. Pydantic parse (with sanitization)
      2. LLM repair call → Pydantic parse
    """
    # Attempt 0: direct parse from narrative
    result = _try_parse(research_narrative)
    if result:
        logger.info(
            "Extraction succeeded on direct parse: verdict=%s confidence=%.2f", result.verdict.value, result.confidence
        )
        return result

    logger.warning("Direct parse failed, starting LLM repair (max %d attempts)", MAX_EXTRACT_ATTEMPTS)
    last_error: Exception | None = None
    repair_input = research_narrative

    for attempt in range(MAX_EXTRACT_ATTEMPTS):
        repair_llm = _build_llm(temperature=0.0)
        response = repair_llm.invoke(
            [
                SystemMessage(content=REPAIR_SYSTEM),
                HumanMessage(
                    content=f"Extract a valid FactCheckResult from the following agent output:\n\n{repair_input}"
                ),
            ]
        )

        result = _try_parse(response.content)
        if result:
            logger.info(
                "Extraction succeeded after %d repair attempt(s): verdict=%s", attempt + 1, result.verdict.value
            )
            return result

        # Feed the failed output back as input for next attempt
        last_error = None
        try:
            json.loads(response.content)
        except Exception as exc:
            last_error = exc
        repair_input = response.content
        logger.debug("Repair attempt %d failed", attempt + 1)

    logger.error(
        "Extraction failed after %d repair attempts: %s", MAX_EXTRACT_ATTEMPTS, last_error or "validation error"
    )
    raise RuntimeError(
        f"Extraction failed after {MAX_EXTRACT_ATTEMPTS} repair attempts: {last_error or 'validation error'}"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_user_message(news_text: str) -> str:
    today = date.today().strftime("%Y-%m-%d")
    return (
        "Please fact-check the following news text.\n"
        "Hard limit: 5 unique searches. STOP RULE: if you have 3+ credible sources agreeing and none contradicting — write your conclusion immediately, no more searches.\n\n"
        f"TODAY'S DATE: {today}\n"
        "If no publication date is mentioned in the news text, treat the date as UNKNOWN — "
        "do not assume or infer a date.\n\n"
        f'NEWS TEXT:\n"""\n{news_text}\n"""'
    )


def _extract_narrative(research_messages: list[Any]) -> str:
    """Pick the best AI message as research narrative for extraction."""
    # 1. Prefer message with JSON block
    for msg in reversed(research_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and _JSON_BLOCK_RE.search(msg.content):
            return msg.content

    # 2. Last non-empty AI message
    for msg in reversed(research_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content

    # 3. Concatenate all AI text
    parts: list[str] = []
    for msg in research_messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            parts.append(msg.content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@traceable(name="analyze_news")
def analyze_news(news_text: str, *, verbose: bool = False, reasoning_effort: str | None = None) -> FactCheckResult:
    """Analyse a news text and return a structured FactCheckResult.

    Parameters
    ----------
    news_text:
        The raw news article or claim to be verified.
    verbose:
        If True, stream and print each step of the agent's reasoning.

    Returns
    -------
    FactCheckResult
        Typed verdict with evidence, sources, reasoning, and confidence score.
    """
    import time as _time

    t0 = _time.perf_counter()

    llm = _build_llm(reasoning_effort=reasoning_effort)
    logger.info("analyze_news started: text_len=%d reasoning_effort=%s", len(news_text), reasoning_effort)

    # ------------------------------------------------------------------
    # Phase 1: ReAct research agent
    # ------------------------------------------------------------------
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    user_message = _build_user_message(news_text)
    research_messages: list[Any] = []
    tool_call_count = 0

    if verbose:
        print("\n[AGENT] Starting research phase...\n")

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_message)]},
        stream_mode="values",
        # 14 iterations = system + user + up to 5 tool calls (call+result each) + final answer + buffer
        config={"recursion_limit": 14},
    ):
        msgs = chunk.get("messages", [])
        if msgs:
            if verbose:
                _print_message(msgs[-1])
            research_messages = msgs
            last = msgs[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                tool_call_count += len(last.tool_calls)

    research_elapsed = _time.perf_counter() - t0
    logger.info(
        "Research phase done: %.1fs, %d tool calls, %d messages",
        research_elapsed,
        tool_call_count,
        len(research_messages),
    )

    # ------------------------------------------------------------------
    # Phase 2: Structured extraction
    # ------------------------------------------------------------------
    if verbose:
        print("\n[AGENT] Extracting structured verdict...\n")

    result = _extract_result(_extract_narrative(research_messages))

    # ------------------------------------------------------------------
    # Phase 3: Update domain reputation DB
    # ------------------------------------------------------------------
    mode = "fast" if reasoning_effort == "low" else "precise"
    try:
        from .db import save_query, update_reputation_from_result

        rep_count = update_reputation_from_result(result, mode=mode, query_text=news_text)
        save_query(news_text, mode, result, reputation_updated=1 if rep_count > 0 else 0)
    except Exception as exc:
        logger.warning("Failed to update DB: %s", exc)

    total_elapsed = _time.perf_counter() - t0
    logger.info(
        "analyze_news done: verdict=%s confidence=%.2f manipulation=%s elapsed=%.1fs tools=%d",
        result.verdict.value,
        result.confidence,
        result.manipulation_type.value,
        total_elapsed,
        tool_call_count,
    )
    return result


_TOOL_LABELS: dict[str, str] = {
    "web_search": "Searching the web",
    "search_fact_checkers": "Checking fact-checkers",
    "check_if_old_news": "Checking for recycled content",
    "extract_article_content": "Reading article",
    "check_domain_reputation": "Checking domain reputation",
}


@traceable(name="analyze_news_stream")
def analyze_news_stream(news_text: str, reasoning_effort: str | None = None) -> Generator[dict, None, None]:
    """Like analyze_news but yields progress events as dicts.

    Event shapes:
        {"type": "thinking",  "text": "..."}   — agent is reasoning
        {"type": "tool_call", "tool": "...", "query": "..."}  — tool invoked
        {"type": "extracting"}                 — phase 2 started
        {"type": "result",    "data": {...}}   — final FactCheckResult dict
        {"type": "error",     "message": "..."} — something went wrong
    """
    try:
        llm = _build_llm(reasoning_effort=reasoning_effort)
        agent = create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    user_message = _build_user_message(news_text)
    research_messages: list[Any] = []
    seen_msg_ids: set[str] = set()

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            stream_mode="values",
            config={"recursion_limit": 14},
        ):
            msgs = chunk.get("messages", [])
            if not msgs:
                continue
            research_messages = msgs
            last = msgs[-1]
            msg_id = getattr(last, "id", None)
            if msg_id in seen_msg_ids:
                continue
            if msg_id:
                seen_msg_ids.add(msg_id)

            if isinstance(last, AIMessage):
                content = last.content
                text = ""
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block["text"].strip()
                            break
                if text:
                    yield {"type": "thinking", "text": text}

                for tc in getattr(last, "tool_calls", []):
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})
                    query = (
                        args.get("query")
                        or args.get("claim")
                        or args.get("claim_or_title")
                        or args.get("url")
                        or args.get("domain")
                        or ""
                    )
                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "label": _TOOL_LABELS.get(tool_name, tool_name),
                        "query": str(query)[:120],
                    }

            elif isinstance(last, ToolMessage):
                yield {"type": "tool_result", "tool": last.name}

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    # Phase 2
    yield {"type": "extracting"}

    try:
        result: FactCheckResult = _extract_result(_extract_narrative(research_messages))

        # Phase 3: Update DB (reputation + query history)
        mode = "fast" if reasoning_effort == "low" else "precise"
        try:
            from .db import save_query, update_reputation_from_result

            rep_count = update_reputation_from_result(result, mode=mode, query_text=news_text)
            save_query(news_text, mode, result, reputation_updated=1 if rep_count > 0 else 0)
        except Exception as exc:
            logger.warning("Failed to update DB: %s", exc)

        yield {"type": "result", "data": result.model_dump(mode="json")}
    except Exception as exc:
        yield {"type": "error", "message": f"Failed to structure result: {exc}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_message(msg: Any) -> None:
    """Pretty-print a LangChain message for verbose mode."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    if isinstance(msg, HumanMessage):
        print(f"[USER] {str(msg.content)[:200]}")
    elif isinstance(msg, AIMessage):
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(f"[AI] {block['text'][:300]}")
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    print(f"[TOOL CALL] {block['name']}({str(block.get('input', {}))[:200]})")
        else:
            print(f"[AI] {str(content)[:300]}")
    elif isinstance(msg, ToolMessage):
        print(f"[TOOL RESULT] {str(msg.content)[:300]}")
