# Agent Architecture

## Overview

Verifyn's core is a **ReAct (Reason-Act-Observe)** agent built on [LangGraph](https://github.com/langchain-ai/langgraph). Unlike traditional binary classifiers, the agent actively searches the live web, queries its own history, and follows a structured 9-step verification methodology before issuing a verdict.

## ReAct Pattern

```
User input
    │
    ▼
┌─────────────────────────┐
│  SystemMessage           │  ← 9-step methodology prompt
│  HumanMessage            │  ← wrapped news text + date + rules
└────────────┬────────────┘
             │
    ┌────────▼────────┐
    │  LLM Reasoning   │  ← "I need to check if this was fact-checked before..."
    │  → Tool Call      │  ← search_similar_queries("claim text")
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Tool Execution   │  ← cosine similarity search in DB
    │  → ToolMessage    │  ← "Found 1 similar: verdict=FAKE, confidence=0.92..."
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  LLM Reasoning   │  ← "Previous check found it fake. Let me verify..."
    │  → Tool Call      │  ← web_search("claim verification")
    └────────┬────────┘
             │
         ... repeat ...
             │
    ┌────────▼────────┐
    │  Final AIMessage  │  ← JSON FactCheckResult block
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Extraction       │  ← Parse JSON → Pydantic → Repair if needed
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Post-processing  │  ← Update domain reputation + save with embedding
    └─────────────────┘
```

## 9-Step Verification Methodology

| Step | Action | Tools Used |
|------|--------|------------|
| 1 | **Search previous fact-checks** | `search_similar_queries` |
| 2 | Extract verifiable claims | `extract_article_content` (if URL) |
| 3 | Find primary source | `web_search` |
| 4 | Check date, place, context | `check_if_old_news`, `extract_article_content` |
| 5 | Lateral reading (3-5 sources) | `web_search` (multiple angles) |
| 6 | Count independent confirmations | Analysis of results |
| 7 | Check for recycled content | `check_if_old_news` |
| 8 | Check professional fact-checkers | `search_fact_checkers` |
| 9 | Evaluate and classify | Agent synthesis |

## Early Stopping

The 9 steps are a guide, not a rigid checklist. After every tool call, the agent checks:

> "Do I have 3+ credible sources agreeing AND 0 contradicting?"

If yes, it stops immediately and writes the conclusion. This prevents unnecessary API calls and reduces latency/cost.

## Multi-Provider LLM Support

The agent supports three LLM providers via `LLM_PROVIDER` env var:

## Extraction Pipeline

The agent's final message contains a `FactCheckResult` JSON block. Extraction follows a three-level strategy:

1. **Direct parse** — regex for ```json block, then `json_repair` + Pydantic validation
2. **LLM repair** — if direct parse fails, a repair LLM reconstructs valid JSON (up to 3 attempts)
3. **Sanitization** — fixes common LLM quirks: pipe-separated enums, case normalization, truncated JSON

## Tool Budget

- **Hard limit: 5 unique web searches** — prevents runaway costs
- `search_similar_queries` does NOT count toward this budget
- `extract_article_content` and `check_domain_reputation` don't count either
- The agent must use distinct query angles — rephrasing the same query is forbidden

## Streaming

Both `analyze_news` (sync) and `analyze_news_stream` (generator) are available:

```python
# Sync
result: FactCheckResult = analyze_news("claim text")

# Streaming — yields progress events
for event in analyze_news_stream("claim text"):
    # {"type": "thinking", "text": "..."}
    # {"type": "tool_call", "tool": "web_search", "query": "..."}
    # {"type": "tool_result", "tool": "web_search"}
    # {"type": "extracting"}
    # {"type": "result", "data": {...}}
```

## Verdict Taxonomy

| Verdict | When Used |
|---------|-----------|
| `REAL` | Confirmed by 3+ independent sources |
| `FAKE` | Core facts fabricated — events didn't happen, quotes invented |
| `MISLEADING` | Real facts distorted through framing, missing context, selective stats |
| `PARTIALLY_FAKE` | Mix of true facts and specific false claims |
| `UNVERIFIABLE` | Insufficient evidence to confirm or deny |
| `SATIRE` | Intentionally fictional/humorous content |
| `NO_CLAIMS` | Input contains no verifiable factual claims |
