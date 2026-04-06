# Similarity Search

## Overview

Before running the full agent pipeline (web searches, API calls), the system checks whether a similar claim has already been fact-checked. This is implemented via **OpenAI embeddings + cosine similarity** over the query history stored in SQLite.

## Architecture

```
New query
    │
    ▼
┌──────────────────────────┐
│  compute_embedding()      │  ← OpenAI text-embedding-3-small (1536 dims)
│  via OpenAI Embeddings API│
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  find_similar_queries()   │  ← Load all stored embeddings from SQLite
│  cosine similarity        │  ← numpy dot product (no external index)
│  threshold filtering      │  ← default >= 0.75
│  mode-aware filtering     │  ← fast mode accepts precise+fast results
│  deduplication            │  ← latest result per normalized query
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Agent decides            │  ← Reuse? Verify? Start fresh?
│  (Step 2 of methodology)  │
└──────────────────────────┘
```

## Mode Logic

The system has two inference modes: `fast` and `precise`. Similarity search respects mode hierarchy:

| Running Mode | Accepts Results From | Rationale |
|-------------|---------------------|-----------|
| `fast` | `precise` + `fast` | Fast mode can trust precise results (more thorough) |
| `precise` | `precise` only | Precise mode should not rely on fast-mode shortcuts |

When both modes match, `precise` results are ranked higher (sorted first at equal similarity).

## Why numpy, Not FAISS/ChromaDB

At current scale (hundreds to low thousands of queries), **brute-force cosine similarity via numpy** is the right choice:

| Approach | Latency (1K vectors) | Latency (100K vectors) |
|----------|---------------------|----------------------|------------|
| numpy dot product | ~1ms | ~50ms |
| FAISS (flat) | ~0.5ms | ~5ms |

**Trade-off**: numpy is O(n) per query vs FAISS O(log n), but for n < 10K the constant factor dominates. The entire vector set fits in a single numpy array loaded from SQLite.

**Migration path**: if the query history grows beyond ~50K entries, switch to FAISS `IndexFlatIP` (inner product on normalized vectors = cosine similarity). The `find_similar_queries()` function is the only place that needs changing — the rest of the pipeline (embedding computation, storage, tool interface) stays the same.

## Embedding Model (by default)

- **Model**: `text-embedding-3-small` (configurable via `EMBEDDING_MODEL` env var)
- **Dimensions**: 1536
- **Cost**: ~$0.00002 per query (negligible vs LLM inference)
- **Latency**: ~100ms per embedding call

## Storage

Embeddings are stored as JSON arrays in the `query_history.embedding` column (TEXT type in SQLite). This avoids schema changes for different embedding dimensions and keeps the database portable.

```sql
-- Example stored embedding (truncated)
embedding = '[0.0123, -0.0456, 0.0789, ...]'  -- 1536 floats as JSON
```

## Deduplication

When searching, only the **most recent result per normalized query** is considered. Normalization strips punctuation, lowercases, and collapses whitespace:

```
"Is the Earth flat?"  →  "is the earth flat"
"  IS  THE  EARTH  FLAT  "  →  "is the earth flat"
```

This prevents the same claim checked multiple times from flooding results.

## Agent Integration

The `search_similar_queries` tool is available to the agent as Step 2 of the methodology. The agent is instructed to:

1. **Not blindly reuse** old results — always evaluate currency
2. **Note previous results** in reasoning — explain trust/distrust
3. **Verify with fresh searches** if the topic is fast-moving
4. **Adopt previous verdicts** only if recent + stable fact + high confidence

The tool call does NOT count toward the 5-search budget.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity to return |
| `OPENAI_API_KEY` | (required) | API key for embeddings |

## Testing

Similarity search is tested with deterministic synthetic embeddings (no API calls):

- Identical/similar/dissimilar vector matching
- Mode filtering (fast vs precise)
- Top-k limiting
- Deduplication by normalized query text
- Graceful handling of missing embeddings
