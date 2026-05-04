from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.application.converse import InMemoryConversationStore

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Project Synthesis — Dashboard</title>
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <style>
    :root {
      --bg: #0b0d10; --panel: #14181d; --fg: #e6edf3; --muted: #7d8590;
      --accent: #58a6ff; --user: #2ea043; --assistant: #58a6ff;
    }
    body { background: var(--bg); color: var(--fg); font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 2rem; }
    h1 { margin: 0 0 .25rem 0; font-weight: 500; letter-spacing: -.02em; }
    .sub { color: var(--muted); font-size: .9rem; margin-bottom: 2rem; }
    .panel { background: var(--panel); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
    .session { color: var(--muted); font-size: .8rem; margin-bottom: .75rem; text-transform: uppercase; letter-spacing: .08em; }
    .msg { padding: .6rem .9rem; border-radius: 8px; margin: .35rem 0; max-width: 80%; line-height: 1.45; }
    .msg.user { background: rgba(46,160,67,.12); border-left: 3px solid var(--user); }
    .msg.assistant { background: rgba(88,166,255,.12); border-left: 3px solid var(--assistant); }
    .msg.system { background: rgba(125,133,144,.08); border-left: 3px solid var(--muted); font-style: italic; color: var(--muted); }
    .role { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: .15rem; }
    .empty { color: var(--muted); text-align: center; padding: 3rem 1rem; }
  </style>
</head>
<body>
  <h1>Project Synthesis</h1>
  <div class="sub">Live transcript · auto-refreshing every 2s</div>
  <div id="transcripts"
       hx-get="/dashboard/transcripts"
       hx-trigger="load, every 2s"
       hx-swap="innerHTML">
    <div class="empty">Loading…</div>
  </div>
</body>
</html>
"""


def build_router(store: InMemoryConversationStore) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    @router.get("/dashboard/transcripts", response_class=HTMLResponse)
    def transcripts_fragment() -> HTMLResponse:
        sessions = store.all_sessions()
        if not sessions:
            return HTMLResponse(
                '<div class="empty">No conversations yet. Say "Hey Synthesis" to start.</div>'
            )
        parts: list[str] = []
        for sid in sessions:
            conv = store.get_or_create(sid)
            msgs_html = "".join(
                f'<div class="msg {m.role.value}">'
                f'<div class="role">{m.role.value}</div>{_escape(m.content)}'
                f"</div>"
                for m in conv.messages
            )
            parts.append(
                f'<div class="panel">'
                f'<div class="session">session · {sid}</div>{msgs_html}</div>'
            )
        return HTMLResponse("".join(parts))

    return router


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
