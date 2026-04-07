# Website

## Overview

The frontend is a **single-page application** built with vanilla JavaScript, served by Nginx. It communicates with the backend via SSE streaming for real-time progress updates during fact-checking.

## Screenshots

![Verifyn home page](website_screenshots/1.png)
![Live agent reasoning](website_screenshots/2.png)
![Structured fact-check verdict](website_screenshots/3.png)

## Architecture

```
┌─────────────────────────────────────┐
│  Nginx (port 3000)                  │
│  ├── Static files (HTML/CSS/JS)     │
│  └── Reverse proxy: /api/* → :8000  │
└─────────────────────────────────────┘
```

## File Structure

```
website/
├── index.html          # Single-page app (~400 lines)
├── Dockerfile           # Nginx Alpine image
├── nginx.conf           # Reverse proxy + gzip + SSE support
├── css/
│   ├── variables.css    # Design tokens (colors, fonts, spacing)
│   ├── base.css         # Typography, resets
│   ├── ticker.css       # Breaking news ticker animation
│   ├── masthead.css     # Header design
│   ├── input.css        # Text input + mode toggle
│   ├── loading.css      # Spinner animation
│   ├── results.css      # Verdict card, evidence grid
│   ├── footer.css
│   └── animations.css   # Transitions, fade-in
└── js/
    ├── app.js           # Main logic, SSE stream handling
    ├── constants.js     # Verdict color mappings
    ├── ui.js            # DOM helpers (hide, show, toggle)
    ├── loading.js       # Timer animation
    └── render.js        # Results rendering (verdict, evidence, tables)
```

## SSE Stream Handling

The frontend connects to `/api/analyze/stream` and processes events in real-time:

1. **thinking** — shows agent's reasoning in a live text area
2. **tool_call** — displays which tool is being used (with friendly labels)
3. **tool_result** — updates progress indicator
4. **extracting** — shows "Structuring results..." phase
5. **result** — renders the full verdict card with evidence grid

## Verdict Display

Each verdict has a distinct color and icon in the results card:

- **REAL** — green
- **FAKE** — red
- **MISLEADING** — orange
- **PARTIALLY_FAKE** — amber
- **UNVERIFIABLE** — gray
- **SATIRE** — purple
- **NO_CLAIMS** — muted

Evidence items are displayed in a grid with source, URL, summary, and credibility badge.

## Nginx Configuration

Key settings for SSE compatibility:

```nginx
location /api/ {
    proxy_pass http://backend:8000/;
    proxy_buffering off;       # Required for SSE
    proxy_cache off;
    proxy_read_timeout 300s;   # Agent can take 60-120s
}
```

Gzip compression is enabled for CSS, JS, and JSON responses.
