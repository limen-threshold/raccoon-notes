"""HTTP API server for Raccoon Notes.

Single primary endpoint:
    POST /lesson         { topic, outline, memory_source } → lesson JSON
    GET  /healthz        → liveness
    GET  /topics         → list of preset topic ids (later, when #8 ships)
    GET  /topics/{id}    → full outline for one preset topic (later)

Usage:
    pip install fastapi uvicorn pyyaml
    export ANTHROPIC_API_KEY=...
    python -m server.http_server           # defaults from server/config.yaml
    python -m server.http_server --port 8200

The OWUI pipeline (and any other web frontend) talks to this.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Optional

# Ensure package imports work whether run as `python -m server.http_server` or directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so launchd / cron-style starts get ANTHROPIC_API_KEY without needing
# it in the plist (keeping secrets out of LaunchAgents/*.plist which is world-readable).
try:
    from dotenv import load_dotenv
    # Check raccoon-notes/.env first, then claude-home/.env
    for env_path in (
        Path(__file__).parent.parent / ".env",
        Path.home() / "claude-home" / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed — caller must set env vars directly

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[http_server] FastAPI / uvicorn not installed. Run: pip install fastapi uvicorn pyyaml")
    raise

from core import pipeline


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


# --- request / response models ---


class ConceptEntry(BaseModel):
    concept: str
    query: Optional[str] = None


class MemorySource(BaseModel):
    type: str = "anchor"  # "anchor" | "mcp_memory" | "raw_chat" | "none"
    endpoint: Optional[str] = None
    search_path: Optional[str] = None
    text: Optional[str] = None  # for raw_chat


class LessonRequest(BaseModel):
    topic: str
    outline: list[ConceptEntry]
    memory_source: Optional[MemorySource] = None
    n_memories_per_concept: int = 5
    reflection_depth: str = "deep"  # "deep" | "light"


# --- app ---


app = FastAPI(
    title="Raccoon Notes",
    description="Personalized learning through shared memory.",
    version="0.1.0",
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/lesson")
def lesson_endpoint(req: LessonRequest):
    # Validate inputs early — give the caller a useful 400 instead of a 500
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must be non-empty")
    if not req.outline:
        raise HTTPException(status_code=400, detail="outline must have at least one concept")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not set on the server",
        )

    outline = [
        {"concept": c.concept, "query": c.query or c.concept} for c in req.outline
    ]
    src = req.memory_source.model_dump() if req.memory_source else None

    lesson = pipeline.run_lesson(
        topic=req.topic,
        outline=outline,
        memory_source=src,
        n_memories_per_concept=req.n_memories_per_concept,
        reflection_depth=req.reflection_depth,
    )

    # Upstream errors are returned as JSON with _error, not as 500s — the lesson
    # endpoint always 200s if we got past validation. Caller inspects _error /
    # _diagnostics fields. This matches the spec: a graceful empty/partial
    # lesson is more useful than a hard failure.
    return lesson


def main():
    p = argparse.ArgumentParser()
    cfg = _load_config()
    http_cfg = cfg.get("http", {})
    p.add_argument("--host", default=http_cfg.get("host", "0.0.0.0"))
    p.add_argument("--port", type=int, default=http_cfg.get("port", 8200))
    p.add_argument("--reload", action="store_true", help="dev: auto-reload on file change")
    args = p.parse_args()

    uvicorn.run(
        "server.http_server:app" if not args.reload else "server.http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
