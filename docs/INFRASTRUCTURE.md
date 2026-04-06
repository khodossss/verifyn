# Infrastructure

## Overview

Verifyn uses **Docker Compose** for deployment, **GitHub Actions** for CI, and **Ruff + pre-commit** for code quality enforcement.

## Docker Compose

Two services orchestrated via `docker-compose.yml`:

```
┌─────────────────────────────────────────┐
│  website (Nginx, port 3000)              │
│  ├── Serves static files                 │
│  ├── Reverse proxies /api/* → backend    │
│  └── Depends on: backend (healthy)       │
├─────────────────────────────────────────┤
│  backend (FastAPI/Uvicorn, port 8000)    │
│  ├── Runs the agent pipeline             │
│  ├── Healthcheck: GET /health            │
│  └── Volume: ./data → /app/data (SQLite) │
└─────────────────────────────────────────┘
```

### Key Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `BACKEND_HOST` | `127.0.0.1` | Backend bind address (localhost only) |
| `BACKEND_PORT` | `8000` | Backend port |
| `WEBSITE_HOST` | `0.0.0.0` | Website bind address (all interfaces) |
| `WEBSITE_PORT` | `3000` | Website port |

### Health Check

The backend container includes a health check that the website depends on:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

### Data Persistence

SQLite database is persisted via Docker volume mount: `./data:/app/data`. This preserves domain reputation scores and query history (with embeddings) across container restarts.

## CI Pipeline (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests:

### Jobs

| Job | Runner | Steps |
|-----|--------|-------|
| **lint** | ubuntu-latest, Python 3.12 | `ruff check .` + `ruff format --check .` |
| **test** | ubuntu-latest, Python 3.12 | `pytest agent/tests/ backend/tests/ -v --tb=short` |
| **docker** | ubuntu-latest | `docker compose build` |

All three jobs run in parallel. The test job does not require API keys — all external calls are mocked.

## Code Quality

### Ruff

Fast Python linter + formatter (replaces flake8 + isort + black):

```toml
# ruff.toml
line-length = 120
target-version = "py311"

[lint]
select = ["E", "F", "I"]  # pycodestyle + pyflakes + isort
ignore = ["E501"]          # line length handled by formatter
```

### Pre-commit Hooks

`.pre-commit-config.yaml` runs on every `git commit`:

1. **ruff** — auto-fix lint issues
2. **ruff-format** — format code
3. **pytest** — run full test suite

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: pytest
        entry: pytest agent/tests/ backend/tests/ -q --tb=short
```

## Environment Variables

All configuration is via `.env` file (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (if using OpenAI) | For LLM + embeddings |
| `ANTHROPIC_API_KEY` | If using Anthropic | For Claude models |
| `TAVILY_API_KEY` | No | For Tavily search (fallback: DuckDuckGo) |
| `LLM_PROVIDER` | No (default: openai) | `openai`, `anthropic`, or `ollama` |
| `DATABASE_URL` | No (default: sqlite) | SQLAlchemy connection string |
| `EMBEDDING_MODEL` | No | OpenAI embedding model name |
| `SIMILARITY_THRESHOLD` | No (default: 0.75) | Cosine similarity cutoff |
| `CREDIBILITY_THRESHOLD` | No (default: 50) | Min points for domain credibility |
| `LOG_LEVEL` | No (default: INFO) | Python logging level |

## Deployment Notes

- Backend binds to `127.0.0.1` by default — not exposed publicly
- Nginx handles SSL termination, gzip, and SSE proxy settings
- For production, consider switching from SQLite to PostgreSQL for concurrent writes
- LangSmith tracing is available for debugging agent behavior in production
