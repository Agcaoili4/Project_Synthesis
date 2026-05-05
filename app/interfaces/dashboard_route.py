import asyncio
from html import escape
from inspect import isawaitable
from typing import Callable, Protocol

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.application.converse import InMemoryConversationStore
from app.domain.conversation import Role


class DashboardTTSEngine(Protocol):
    async def speak(self, text: str) -> None: ...


class DashboardTTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)


_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" role="img" aria-label="Synthesis">
  <rect width="256" height="256" rx="48" fill="#0d1117"/>
  <rect x="38" y="38" width="180" height="180" rx="32" fill="#101f28" stroke="#3f8fb5" stroke-width="8"/>
  <path d="M84 144c12 18 30 27 54 27 24 0 38-10 38-27 0-20-17-26-44-32-25-6-43-16-43-41 0-27 22-44 55-44 22 0 40 6 54 19l-18 26c-11-9-24-14-38-14-16 0-25 6-25 16 0 13 14 18 36 23 31 8 52 19 52 48 0 31-24 51-66 51-35 0-59-13-74-38z" fill="#dff5ff"/>
</svg>
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0d1117">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Synthesis">
  <link rel="manifest" href="/manifest.webmanifest">
  <title>Synthesis</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1117;
      --surface: #111820;
      --surface-2: #151f2a;
      --line: #263241;
      --fg: #edf2f7;
      --muted: #91a0ad;
      --quiet: #627181;
      --accent: #59c2ff;
      --green: #42d392;
      --amber: #f4b860;
      --danger: #ff7575;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea {
      font: inherit;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
    }
    .rail {
      border-right: 1px solid var(--line);
      background: #0f151c;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 44px;
    }
    .mark {
      width: 36px;
      height: 36px;
      border: 1px solid #3b6f88;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--accent);
      font-weight: 700;
      background: #101f28;
    }
    h1 {
      margin: 0;
      font-size: 1.05rem;
      line-height: 1.2;
      font-weight: 650;
    }
    .caption {
      color: var(--muted);
      font-size: .82rem;
      margin-top: 2px;
    }
    .status-grid {
      display: grid;
      gap: 8px;
    }
    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--surface);
      min-width: 0;
    }
    .status-row span:first-child {
      color: var(--muted);
    }
    .status-row span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--green);
      display: inline-block;
      margin-right: 7px;
    }
    .sessions {
      display: grid;
      gap: 8px;
      min-height: 0;
      overflow: auto;
    }
    .session-button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--fg);
      padding: 10px;
      text-align: left;
      cursor: pointer;
      min-height: 44px;
    }
    .session-button:hover,
    .session-button.active {
      border-color: #3f8fb5;
      background: #132330;
    }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
    }
    .topbar {
      min-height: 64px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 22px;
      gap: 16px;
      background: #0d1117;
    }
    .topbar h2 {
      margin: 0;
      font-size: 1rem;
      font-weight: 620;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .session-input {
      width: min(28vw, 260px);
      min-width: 140px;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--fg);
      padding: 0 10px;
    }
    .icon-button, .send-button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--fg);
      cursor: pointer;
      padding: 0 12px;
    }
    .send-button {
      border-color: #3f8fb5;
      background: #123045;
      color: #dff5ff;
      min-width: 84px;
    }
    .icon-button:hover, .send-button:hover {
      filter: brightness(1.08);
    }
    .transcript {
      overflow: auto;
      padding: 22px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      margin-bottom: 12px;
      overflow: hidden;
    }
    .session {
      color: var(--muted);
      font-size: .78rem;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .messages {
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .msg {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      max-width: 860px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .msg.user {
      border-color: rgba(66, 211, 146, .35);
      background: rgba(66, 211, 146, .08);
    }
    .msg.assistant {
      border-color: rgba(89, 194, 255, .35);
      background: rgba(89, 194, 255, .08);
    }
    .msg.system {
      border-color: rgba(145, 160, 173, .25);
      background: rgba(145, 160, 173, .06);
      color: var(--muted);
      font-style: italic;
    }
    .role {
      font-size: .7rem;
      color: var(--quiet);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 4px;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 48px 16px;
      background: rgba(255, 255, 255, .02);
    }
    .composer {
      border-top: 1px solid var(--line);
      background: #0d1117;
      padding: 14px 22px 18px;
      display: grid;
      gap: 12px;
    }
    .voice-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 12px;
      display: grid;
      gap: 12px;
    }
    .voice-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .voice-head strong {
      font-size: .9rem;
      font-weight: 620;
    }
    .wave {
      height: 58px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b131a;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      padding: 8px 10px;
      overflow: hidden;
    }
    .bar {
      width: 5px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent);
      opacity: .35;
      transform-origin: center;
      transition: height .18s ease, opacity .18s ease;
    }
    .wave.speaking .bar {
      opacity: .92;
      animation: speakWave 940ms ease-in-out infinite;
    }
    .wave.speaking .bar:nth-child(2n) { animation-duration: 780ms; }
    .wave.speaking .bar:nth-child(3n) { animation-duration: 1120ms; }
    .wave.speaking .bar:nth-child(4n) { animation-delay: 100ms; }
    .wave.speaking .bar:nth-child(5n) { animation-delay: 180ms; }
    @keyframes speakWave {
      0%, 100% { height: 10px; }
      35% { height: 38px; }
      65% { height: 22px; }
    }
    .composer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--fg);
      padding: 12px;
      line-height: 1.4;
    }
    .notice {
      color: var(--muted);
      min-height: 18px;
      font-size: .82rem;
    }
    .notice.error {
      color: var(--danger);
    }
    @media (max-width: 760px) {
      .app {
        grid-template-columns: 1fr;
      }
      .rail {
        display: none;
      }
      .main {
        height: 100vh;
      }
      .topbar {
        align-items: stretch;
        flex-direction: column;
      }
      .actions {
        width: 100%;
      }
      .session-input {
        width: 100%;
      }
      .composer-row {
        grid-template-columns: 1fr;
      }
      .send-button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="rail">
      <div class="brand">
        <div class="mark">S</div>
        <div>
          <h1>Synthesis</h1>
          <div class="caption">Local voice assistant</div>
        </div>
      </div>
      <div class="status-grid">
        <div class="status-row"><span>Brain</span><span><i class="dot"></i>Local</span></div>
        <div class="status-row"><span>Audio</span><span>Daemon-owned</span></div>
        <div class="status-row"><span>Cloud audio</span><span>Off</span></div>
      </div>
      <div>
        <div class="caption" style="margin-bottom: 8px;">Sessions</div>
        <div id="sessions" class="sessions"></div>
      </div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div>
          <h2>Conversation</h2>
          <div class="caption">Voice loop monitor and local text console</div>
        </div>
        <div class="actions">
          <input id="sessionId" class="session-input" value="app" autocomplete="off" aria-label="Session ID">
          <button id="refresh" class="icon-button" type="button" title="Refresh transcript">Refresh</button>
        </div>
      </header>
      <section id="transcripts"
               class="transcript"
               data-source="/dashboard/transcripts">
        <div class="empty">Loading...</div>
      </section>
      <form id="composer" class="composer">
        <section class="voice-panel" aria-label="Text to speech">
          <div class="voice-head">
            <div>
              <strong>TTS Studio</strong>
              <div class="caption">Speak text through the local voice only</div>
            </div>
            <div id="ttsState" class="caption">Idle</div>
          </div>
          <div id="wave" class="wave" aria-hidden="true">
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
            <span class="bar"></span><span class="bar"></span><span class="bar"></span>
          </div>
          <div class="composer-row">
            <textarea id="ttsText" name="ttsText" placeholder="Type text for Synthesis to speak..." aria-label="TTS text"></textarea>
            <button id="ttsSpeak" class="send-button" type="button">Speak</button>
          </div>
        </section>
        <div class="composer-row">
          <textarea id="message" name="message" placeholder="Type to the local brain..." aria-label="Message"></textarea>
          <button class="send-button" type="submit">Send</button>
        </div>
        <div id="notice" class="notice" role="status"></div>
      </form>
    </main>
  </div>
  <script>
    const sessionInput = document.getElementById("sessionId");
    const messageInput = document.getElementById("message");
    const ttsInput = document.getElementById("ttsText");
    const ttsButton = document.getElementById("ttsSpeak");
    const ttsState = document.getElementById("ttsState");
    const wave = document.getElementById("wave");
    const notice = document.getElementById("notice");
    const transcripts = document.getElementById("transcripts");
    const sessions = document.getElementById("sessions");

    function setNotice(text, isError = false) {
      notice.textContent = text;
      notice.classList.toggle("error", isError);
    }

    function activeSessionId() {
      const value = sessionInput.value.trim();
      return value || "app";
    }

    async function refreshTranscripts() {
      const response = await fetch(transcripts.dataset.source, { credentials: "same-origin" });
      if (!response.ok) {
        setNotice("Unable to refresh transcript.", true);
        return;
      }
      transcripts.innerHTML = await response.text();
      buildSessionList();
      setNotice("Ready.");
    }

    function buildSessionList() {
      const panels = [...transcripts.querySelectorAll("[data-session-id]")];
      sessions.innerHTML = "";
      if (panels.length === 0) {
        const empty = document.createElement("div");
        empty.className = "caption";
        empty.textContent = "No sessions yet";
        sessions.appendChild(empty);
        return;
      }
      for (const panel of panels) {
        const sid = panel.dataset.sessionId;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "session-button";
        button.textContent = sid;
        if (sid === activeSessionId()) button.classList.add("active");
        button.addEventListener("click", () => {
          sessionInput.value = sid;
          panel.scrollIntoView({ behavior: "smooth", block: "start" });
          buildSessionList();
        });
        sessions.appendChild(button);
      }
    }

    document.getElementById("refresh").addEventListener("click", refreshTranscripts);
    document.getElementById("composer").addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = messageInput.value.trim();
      if (!text) return;
      setNotice("Thinking...");
      const response = await fetch("/converse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ transcript: text, session_id: activeSessionId() }),
      });
      if (!response.ok) {
        const detail = await response.text();
        setNotice(detail || "The local brain did not respond.", true);
        return;
      }
      messageInput.value = "";
      await refreshTranscripts();
    });
    ttsButton.addEventListener("click", async () => {
      const text = ttsInput.value.trim();
      if (!text) return;
      ttsButton.disabled = true;
      wave.classList.add("speaking");
      ttsState.textContent = "Speaking";
      setNotice("Speaking through local TTS...");
      const response = await fetch("/dashboard/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ text }),
      });
      wave.classList.remove("speaking");
      ttsButton.disabled = false;
      if (!response.ok) {
        const detail = await response.text();
        ttsState.textContent = "Error";
        setNotice(detail || "TTS failed.", true);
        return;
      }
      ttsState.textContent = "Idle";
      setNotice("TTS complete.");
    });
    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        document.getElementById("composer").requestSubmit();
      }
    });
    ttsInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        ttsButton.click();
      }
    });
    sessionInput.addEventListener("input", buildSessionList);
    refreshTranscripts();
    setInterval(refreshTranscripts, 2000);
  </script>
</body>
</html>
"""


def build_router(
    store: InMemoryConversationStore,
    tts_engine_factory: Callable[[], DashboardTTSEngine] | None = None,
) -> APIRouter:
    router = APIRouter()
    tts_factory = tts_engine_factory or _default_tts_engine_factory
    tts_engine: DashboardTTSEngine | None = None
    tts_warmed = False
    tts_lock = asyncio.Lock()

    async def get_tts_engine() -> DashboardTTSEngine:
        nonlocal tts_engine, tts_warmed
        if tts_engine is None:
            tts_engine = tts_factory()
        warm_up = getattr(tts_engine, "warm_up", None)
        if warm_up is not None and not tts_warmed:
            result = warm_up()
            if isawaitable(result):
                await result
            tts_warmed = True
        return tts_engine

    @router.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    @router.get("/manifest.webmanifest")
    def manifest() -> JSONResponse:
        return JSONResponse(
            {
                "name": "Project Synthesis",
                "short_name": "Synthesis",
                "start_url": "/",
                "scope": "/",
                "display": "standalone",
                "background_color": "#0d1117",
                "theme_color": "#0d1117",
                "description": "A fully local voice assistant console.",
                "icons": [
                    {
                        "src": "/synthesis-icon.svg",
                        "sizes": "any",
                        "type": "image/svg+xml",
                        "purpose": "any maskable",
                    }
                ],
            },
            media_type="application/manifest+json",
        )

    @router.get("/synthesis-icon.svg")
    def icon() -> HTMLResponse:
        return HTMLResponse(_ICON, media_type="image/svg+xml")

    @router.post("/dashboard/tts")
    async def tts_preview(req: DashboardTTSRequest) -> JSONResponse:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text cannot be blank")
        async with tts_lock:
            try:
                engine = await get_tts_engine()
                await engine.speak(text)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"local TTS unavailable: {exc}",
                ) from exc
        return JSONResponse({"ok": True})

    @router.get("/dashboard/transcripts", response_class=HTMLResponse)
    def transcripts_fragment() -> HTMLResponse:
        sessions = store.all_sessions()
        if not sessions:
            return HTMLResponse(
                '<div class="empty">No conversations yet. Wake Synthesis or send a local text message.</div>'
            )
        parts: list[str] = []
        for sid in sessions:
            conv = store.get_or_create(sid)
            msgs_html = "".join(
                f'<div class="msg {m.role.value}">'
                f'<div class="role">{_escape(m.role.value)}</div>{_escape(m.content)}'
                f"</div>"
                for m in conv.messages
                if m.role != Role.SYSTEM
            )
            parts.append(
                f'<div class="panel" data-session-id="{_escape(sid)}">'
                f'<div class="session">session · {_escape(sid)}</div>'
                f'<div class="messages">{msgs_html}</div></div>'
            )
        return HTMLResponse("".join(parts))

    return router


def _escape(s: str) -> str:
    return escape(s, quote=True)


def _default_tts_engine_factory() -> DashboardTTSEngine:
    from app.core.config import get_settings
    from app.infrastructure.tts.factory import build_tts_engine

    return build_tts_engine(get_settings())
