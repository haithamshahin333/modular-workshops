"""Custom HTTP routes packaged with the Chat LangChain deployment (Module 6).

``langgraph.chat_langchain.json`` registers this app under ``http.app``; the
Agent Server mounts it next to its built-in ``/threads``, ``/runs`` and ``/docs``
routes, so ``/chat`` is served from the deployment itself (and by ``langgraph dev``).
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI()


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page() -> str:
    """A single-file chat UI that talks to this server's own /threads and /runs API."""
    return (UI_DIR / "index.html").read_text()
