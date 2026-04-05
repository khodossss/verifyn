# Verifyn — AI Fact-Check Engine

An autonomous AI agent that classifies news as **REAL**, **FAKE**, **MISLEADING**, **PARTIALLY_FAKE**, **UNVERIFIABLE**, or **SATIRE** — with evidence, sources, and step-by-step reasoning. Exposes a REST API consumed by a newspaper-style web frontend.

**→ [github.com/khodossss/verifyn](https://github.com/khodossss/verifyn)**

---

## Why This Agent

Most "fake news detectors" are binary classifiers trained on static datasets. They score a text against patterns learned months ago and give you a number. They cannot look anything up.

**Verifyn works differently:**

| Feature | Typical classifier | Verifyn |
| --- | --- | --- |
| Verdict options | Real / Fake | REAL, FAKE, MISLEADING, PARTIALLY_FAKE, UNVERIFIABLE, SATIRE |
| Evidence | None | Structured `evidence_for` and `evidence_against` with source URLs |
| Sources | None | Live web search + professional fact-checker cross-reference |
| Methodology | ML heuristics | 8-step methodology used by professional journalists |
| Manipulation type | None | 9 types: FABRICATED, CONTEXT_MANIPULATION, OLD_CONTENT_RECYCLED, … |
| Explainability | Score only | Full step-by-step reasoning narrative + plain-language summary |
| Freshness | Frozen at training | Searches the live web — handles breaking news |

The core is a [ReAct](https://arxiv.org/abs/2210.03629) agent (Reason → Act → Observe → Repeat) built with **LangGraph** that autonomously decides which tools to call and in what order — until it has enough evidence to reach a verdict.

---

## How It Works

### ReAct Research Loop

```text
                News text
                    │
                    ▼
┌─────────────────────────────────────────┐
│               ReAct Agent               │
│                                         │
│       Think → pick tool → call tool     │
│         ↑                      │        │
│         └──── observe result ◄─┘        │
│                                         │
│          Stops when confident           │
└─────────────────────────────────────────┘
                    │
                    ▼
               Final message 
```

The agent follows an 8-step fact-checking methodology:

| Step | Action |
| --- | --- |
| 1 | Extract specific, verifiable claims from the text |
| 2 | Find the **primary source** — original document, transcript, or official statement |
| 3 | Check **date and context** — is the quote/statistic shown in full? |
| 4 | **Lateral reading** — 3–5 independent external sources, not just the original site |
| 5 | Count independent **confirmations or refutations** |
| 6 | Detect **recycled content** — old photos or quotes repackaged as new |
| 7 | Search **professional fact-checkers**: Snopes, PolitiFact, Reuters, AFP, Full Fact |
| 8 | Classify manipulation type and calibrate confidence |

### Tools Available to the Agent

| Tool | Purpose |
| --- | --- |
| `web_search(query)` | Lateral reading via Tavily (DuckDuckGo fallback) |
| `search_fact_checkers(claim)` | Scoped search on Snopes, PolitiFact, Reuters, AFP, Full Fact |
| `check_if_old_news(claim)` | Detect recycled content by finding earliest appearances |
| `extract_article_content(url)` | Fetch full article text, author, date via trafilatura |
| `check_domain_reputation(domain)` | WHOIS lookup + credibility/flagged domain lists |

---

## Web Frontend — Verifyn

Newspaper-style web interface served by Nginx at port 3000.

**Start it:**

```bash
docker compose up --build
# Open http://localhost:3000
```

**UI overview:**

#TODO: insert image

---

## Setup

### Prerequisites

- Docker + Docker Compose (recommended), **or** Python 3.11+
- **OPENAI_API_KEY** — from [platform.openai.com](https://platform.openai.com)
- **TAVILY_API_KEY** — from [app.tavily.com](https://app.tavily.com) (free: 1000 req/month)

### 1. Clone & configure

```bash
git clone https://github.com/khodossss/verifyn.git
cd verifyn/fake_news_detection
cp .env.example .env   # then fill in your API keys
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### 3. Run without Docker

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Open website/index.html directly in a browser
```

### 4. CLI usage

```bash
python agent/run.py --text "NASA confirmed drinking bleach cures COVID-19"
python agent/run.py --file article.txt --verbose
python agent/run.py --json --text "..."
```

### 5. Use as a Python library

```python
from dotenv import load_dotenv
load_dotenv()

from agent import analyze_news

result = analyze_news("Your news text here")
print(result.verdict)           # Verdict.FAKE
print(result.confidence)        # 0.95
print(result.manipulation_type) # ManipulationType.FABRICATED
print(result.summary)
```

---

## Output Schema

```python
class FactCheckResult(BaseModel):
    verdict: Verdict              # REAL | FAKE | MISLEADING | PARTIALLY_FAKE | UNVERIFIABLE | SATIRE
    confidence: float             # 0.0 – 1.0
    confidence_level: str         # HIGH (≥0.8) | MEDIUM (≥0.5) | LOW (<0.5)
    manipulation_type: str        # FABRICATED | CONTEXT_MANIPULATION | OLD_CONTENT_RECYCLED |
                                  # MISLEADING_HEADLINE | PARTIAL_TRUTH | SATIRE_MISREPRESENTED |
                                  # COORDINATED_DISINFO | IMPERSONATION | NONE

    main_claims: list[str]
    primary_source: str | None
    date_context: str | None

    evidence_for: list[EvidenceItem]
    evidence_against: list[EvidenceItem]
    fact_checker_results: list[str]
    sources_checked: list[str]

    reasoning: str   # full 8-step walkthrough
    summary: str     # 2–4 sentence plain-language verdict
```

---

## Project Structure

```text
fake_news_detection/
├── docker-compose.yml
├── requirements.txt
├── .env.example
│
├── backend/
│   ├── main.py                # FastAPI app — POST /analyze, GET /health
│   └── Dockerfile
│
├── website/
│   ├── index.html             # Newspaper-style UI
│   ├── css/                   # Modular CSS (variables, ticker, masthead, results…)
│   ├── js/                    # Modular JS (app.js, render.js, loading.js…)
│   ├── nginx.conf
│   └── Dockerfile             # Nginx static server
│
└── agent/
    ├── agent.py               # analyze_news() — ReAct loop + Pydantic extraction
    ├── models.py              # FactCheckResult, Verdict, ManipulationType
    ├── run.py                 # CLI entry point (Rich terminal UI)
    ├── evaluate.py            # Evaluation script (20-item labeled dataset)
    ├── prompts/
    │   └── system.py          # 8-step methodology system prompt
    ├── tools/
    │   ├── search.py          # web_search, search_fact_checkers, check_if_old_news
    │   ├── extractor.py       # extract_article_content
    │   └── domain.py          # check_domain_reputation
    └── eval/
        └── dataset.json       # 20 labeled news items for evaluation
```

---

## Key Design Decisions

**Why a ReAct agent?**
A fixed chain cannot adapt — if a source is paywalled, the agent tries another. If a claim needs more lateral reading, it searches again. The graph structure lets the agent reason about whether it has enough evidence before stopping.

**Why Tavily instead of Google/Bing?**
Tavily returns pre-extracted text optimised for LLM consumption and supports `search_depth="advanced"` to fetch full page content — no additional scraping needed.

**Why lateral reading?**
Research by the Stanford History Education Group shows that reading a source in isolation (vertical reading) is less reliable than immediately checking what *other* sources say about it. Professional fact-checkers use the same approach.

**Why is UNVERIFIABLE a valid verdict?**
Forcing a binary true/false when evidence is insufficient creates false confidence. If the agent cannot find enough independent corroboration, UNVERIFIABLE is the honest answer.
