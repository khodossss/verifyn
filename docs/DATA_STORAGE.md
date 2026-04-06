# Data Storage

## Overview

Verifyn uses **SQLAlchemy + SQLite** for persistent storage, with two tables: `domain_reputation` (self-learning source credibility) and `query_history` (past fact-checks with embeddings).

## Schema

### domain_reputation

Tracks how reliable a news source domain is, learned from agent verdicts over time.

| Column | Type | Description |
|--------|------|-------------|
| `domain` | TEXT PK | Root domain (e.g., `reuters.com`) |
| `true_points` | REAL | Points from appearing in credible contexts |
| `false_points` | REAL | Points from appearing in unreliable contexts |
| `total_checks` | INTEGER | Number of fact-checks referencing this domain |
| `first_seen` | TIMESTAMP | When first encountered |
| `last_checked` | TIMESTAMP | Most recent check |
| `comment` | TEXT | Optional notes |

### query_history

Stores every fact-check query, its result, and an embedding vector for similarity search.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `query` | TEXT | Original news text |
| `embedding` | TEXT | JSON array of floats (OpenAI embedding, 1536 dims) |
| `mode` | TEXT | `fast` or `precise` |
| `result` | TEXT | JSON-serialized `FactCheckResult` |
| `reputation_updated` | INTEGER | 1 if this query updated domain scores |
| `created_at` | TIMESTAMP | When the query was processed |

## Domain Reputation Scoring

After each fact-check, domains from evidence URLs are scored based on the verdict:

| Verdict | Evidence For (supports claim) | Evidence Against (refutes claim) |
|---------|-------------------------------|----------------------------------|
| REAL | +1.0 true | +1.0 false |
| FAKE | +1.0 false | +1.0 true |
| MISLEADING | +0.7 false, +0.3 true | +0.3 false, +0.7 true |
| PARTIALLY_FAKE | +0.5 each | +0.5 each |
| SATIRE / UNVERIFIABLE | No change | No change |

### Mode Multiplier

Fast mode applies a **0.5x multiplier** to all score updates (fewer sources checked = less reliable signal).

### Credibility Threshold

Once a domain accumulates `true_points + false_points >= 50` (configurable via `CREDIBILITY_THRESHOLD`), its credibility score is computed:

```
credibility = true_points / (true_points + false_points)
```

- **>= 0.75** → HIGH credibility
- **>= 0.50** → MEDIUM credibility
- **< 0.50** → LOW credibility

Below the threshold, the agent falls back to web-based reputation search (Media Bias/Fact Check, AllSides, NewsGuard).

## Embedding Storage

Query embeddings are stored as JSON text in SQLite. This approach:

- Avoids binary column types (portable across SQLite/PostgreSQL)
- Supports variable embedding dimensions without schema changes
- Enables easy inspection and debugging
- Trade-off: ~12KB per embedding vs ~6KB binary — acceptable at current scale

## PostgreSQL Compatibility

The schema uses standard SQLAlchemy types and is compatible with PostgreSQL. To switch:

```bash
DATABASE_URL=postgresql://user:pass@host/verifyn
```

For production with concurrent writes, PostgreSQL is recommended.

## Data Flow

```
Agent completes fact-check
    │
    ├── update_reputation_from_result()
    │   ├── Extract domains from evidence URLs
    │   ├── Look up scoring table (verdict × supports_claim)
    │   ├── Apply mode multiplier
    │   └── Upsert domain_reputation records
    │
    └── save_query()
        ├── Serialize FactCheckResult to JSON
        ├── Store embedding (if computed)
        └── Insert into query_history
```
