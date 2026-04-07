"""Fact-check agent: ReAct research loop + Pydantic extraction."""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from collections.abc import Generator
from datetime import date
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Suppress LangGraph 1.0 internal deprecation warning (not actionable from user code)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
    from langgraph.prebuilt import create_react_agent

from langsmith import traceable

from .constants import (
    AGENT_RECURSION_LIMIT,
    LLM_DEFAULT_TEMPERATURE,
    LLM_MAX_TOKENS,
    MAX_EXTRACT_ATTEMPTS,
    MAX_SEARCH_BUDGET,
)
from .models import FactCheckResult
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS

logger = logging.getLogger("verifyn")

_PROVIDER_ENV = {
    "openai": {"key": "OPENAI_API_KEY", "model": "OPENAI_MODEL", "default_model": "gpt-4o-mini"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-4-20250514"},
    "ollama": {"key": None, "model": "OLLAMA_MODEL", "default_model": "llama3.1"},
}


def _build_llm(temperature: float = LLM_DEFAULT_TEMPERATURE, reasoning_effort: str | None = None) -> BaseChatModel:
    """Build a chat model from LLM_PROVIDER env config."""
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
            max_tokens=LLM_MAX_TOKENS,
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
            max_tokens=LLM_MAX_TOKENS,
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
    "Return ONLY a valid JSON object: no markdown, no explanation, no code fences. "
    "Required fields: verdict, confidence (float 0.0-1.0), confidence_level, manipulation_type, reasoning, summary. "
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
    """Normalize common LLM JSON quirks before Pydantic validation."""
    mt = data.get("manipulation_type", "")
    if isinstance(mt, str) and "|" in mt:
        data["manipulation_type"] = mt.split("|")[0].strip()
    cl = data.get("confidence_level")
    if isinstance(cl, str):
        data["confidence_level"] = cl.upper().strip()
    v = data.get("verdict")
    if isinstance(v, str):
        data["verdict"] = v.upper().strip()
    return data


def _try_parse(text: str) -> FactCheckResult | None:
    """Parse FactCheckResult from text via json_repair, or return None."""
    from json_repair import repair_json

    candidates: list[str] = []

    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(1))

    start = text.find("{")
    if start != -1:
        candidates.append(text[start:])

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
    """Parse FactCheckResult from the agent narrative, with LLM repair fallback."""
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


def _build_user_message(news_text: str) -> str:
    today = date.today().strftime("%Y-%m-%d")
    return (
        "Please fact-check the following news text.\n"
        f"Hard limit: {MAX_SEARCH_BUDGET} unique searches. "
        "STOP RULE: if you have 3+ credible sources agreeing and none contradicting, "
        "write your conclusion immediately, no more searches.\n\n"
        f"TODAY'S DATE: {today}\n"
        "If no publication date is mentioned in the news text, treat the date as UNKNOWN. "
        "Do not assume or infer a date.\n\n"
        f'NEWS TEXT:\n"""\n{news_text}\n"""'
    )


def _extract_narrative(research_messages: list[Any]) -> str:
    """Select the best AI message to feed into result extraction."""
    # Prefer message containing a JSON block
    for msg in reversed(research_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and _JSON_BLOCK_RE.search(msg.content):
            return msg.content

    # Fall back to the last non-empty AI message
    for msg in reversed(research_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content

    # Last resort: concatenate all AI text
    parts: list[str] = []
    for msg in research_messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            parts.append(msg.content)
    return "\n".join(parts)


@traceable(name="analyze_news")
def analyze_news(news_text: str, *, verbose: bool = False, reasoning_effort: str | None = None) -> FactCheckResult:
    """Run the fact-check agent on news_text and return a structured verdict."""
    import time as _time

    t0 = _time.perf_counter()

    llm = _build_llm(reasoning_effort=reasoning_effort)
    mode = "fast" if reasoning_effort == "low" else "precise"
    eval_mode = os.environ.get("VERIFYN_EVAL_MODE") == "1"
    logger.info(
        "analyze_news started: text_len=%d reasoning_effort=%s mode=%s eval_mode=%s",
        len(news_text),
        reasoning_effort,
        mode,
        eval_mode,
    )

    query_embedding: list[float] | None = None
    if not eval_mode:
        try:
            from .db import compute_embedding

            query_embedding = compute_embedding(news_text)
            logger.info("Computed query embedding: %d dimensions", len(query_embedding))
        except Exception as exc:
            logger.warning("Failed to compute embedding: %s", exc)

    from .tools.similarity import search_similar_queries as _sim_tool

    _sim_tool._current_mode = mode

    # Phase 1: ReAct research
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
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
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

    # Phase 2: structured extraction
    if verbose:
        print("\n[AGENT] Extracting structured verdict...\n")

    result = _extract_result(_extract_narrative(research_messages))

    # Phase 3: persist results (skipped in eval mode)
    if not eval_mode:
        try:
            from .db import save_query, update_reputation_from_result

            rep_count = update_reputation_from_result(result, mode=mode)
            save_query(news_text, mode, result, reputation_updated=1 if rep_count > 0 else 0, embedding=query_embedding)
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
    "search_similar_queries": "Searching previous fact-checks",
    "web_search": "Searching the web",
    "search_fact_checkers": "Checking fact-checkers",
    "check_if_old_news": "Checking for recycled content",
    "extract_article_content": "Reading article",
    "check_domain_reputation": "Checking domain reputation",
}


@traceable(name="analyze_news_stream")
def analyze_news_stream(news_text: str, reasoning_effort: str | None = None) -> Generator[dict, None, None]:
    """Streaming version of analyze_news yielding progress events as dicts.

    Event types: thinking, tool_call, tool_result, extracting, result, error.
    """
    mode = "fast" if reasoning_effort == "low" else "precise"

    query_embedding: list[float] | None = None
    try:
        from .db import compute_embedding

        query_embedding = compute_embedding(news_text)
    except Exception as exc:
        logger.warning("Failed to compute embedding: %s", exc)

    from .tools.similarity import search_similar_queries as _sim_tool

    _sim_tool._current_mode = mode

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
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
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

    yield {"type": "extracting"}

    try:
        result: FactCheckResult = _extract_result(_extract_narrative(research_messages))

        try:
            from .db import save_query, update_reputation_from_result

            rep_count = update_reputation_from_result(result, mode=mode)
            save_query(news_text, mode, result, reputation_updated=1 if rep_count > 0 else 0, embedding=query_embedding)
        except Exception as exc:
            logger.warning("Failed to update DB: %s", exc)

        yield {"type": "result", "data": result.model_dump(mode="json")}
    except Exception as exc:
        yield {"type": "error", "message": f"Failed to structure result: {exc}"}


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
