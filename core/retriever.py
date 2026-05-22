"""Memory retrieval — wraps the four supported memory sources.

Per spec, Raccoon Notes is memory-source-agnostic. This module's only job is to
take a `memory_source` declaration + a query, and return a list of memory dicts:

    [
        {"text": "...", "timestamp": "...", "tag": "...", "score": 0.83, ...},
        ...
    ]

Downstream (mapper.py / generator.py) doesn't know or care where the memory came
from. That's the whole product boundary.

Priority order (spec §"记忆源优先级"):
    anchor      — preferred (structured + cross-window persistent)
    mcp_memory  — any MCP-compliant memory server (later)
    raw_chat    — user pastes a chat dump, we extract on the fly (later)
    none        — fallback: empty list, generator prompts user to provide context
"""
from __future__ import annotations
import os
import json
import urllib.parse
import urllib.request
from typing import Any


# --- public entry ---


def retrieve(memory_source: dict, query: str, n: int = 5) -> list[dict]:
    """Fetch memories from the source. Returns [] on any failure — never raises.

    memory_source schema:
        {"type": "anchor", "endpoint": "http://localhost:8000", "search_path": "/limen/search"}
        {"type": "mcp_memory", ...}     # not implemented yet
        {"type": "raw_chat", "text": "..."}  # not implemented yet
        {"type": "none"}                 # explicit no-memory mode

    Returns at most n memories, sorted by source-defined relevance.
    """
    src_type = (memory_source or {}).get("type", "none")
    if src_type == "anchor":
        return _retrieve_anchor(memory_source, query, n)
    if src_type == "mcp_memory":
        return _retrieve_mcp(memory_source, query, n)
    if src_type == "raw_chat":
        return _retrieve_raw_chat(memory_source, query, n)
    if src_type == "none":
        return []
    # Unknown type — treat as no-memory rather than crash.
    return []


# --- anchor source ---


def _retrieve_anchor(src: dict, query: str, n: int) -> list[dict]:
    """Hit Anchor's HTTP search endpoint."""
    endpoint = src.get("endpoint", "http://localhost:8000").rstrip("/")
    search_path = src.get("search_path", "/limen/search")
    url = f"{endpoint}{search_path}?{urllib.parse.urlencode({'q': query, 'n': n})}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"[retriever] anchor fetch failed: {e}")
        return []
    # Anchor returns {"memories": [{"memory_id", "timestamp", "tag", "snippet", ...}]}
    raw = data.get("memories", []) if isinstance(data, dict) else []
    out = []
    for m in raw[:n]:
        if not isinstance(m, dict):
            continue
        text = m.get("snippet") or m.get("text") or m.get("context") or ""
        if not text:
            continue
        out.append({
            "text": text,
            "timestamp": m.get("timestamp", ""),
            "tag": m.get("tag", ""),
            "memory_id": m.get("memory_id", ""),
            "source": "anchor",
        })
    return out


# --- mcp source (later) ---


def _retrieve_mcp(src: dict, query: str, n: int) -> list[dict]:
    """Stub. TODO: implement against an MCP memory server."""
    print("[retriever] mcp_memory not yet implemented")
    return []


# --- raw chat source (later) ---


def _retrieve_raw_chat(src: dict, query: str, n: int) -> list[dict]:
    """Stub. TODO: extract memories from a pasted chat dump using a small LLM call.

    The spec calls this "粗糙聊天记录提取" — user pastes a transcript, we run a
    one-shot LLM extraction to pull discrete memory candidates relevant to query.
    Different prompt than Anchor's curator: we have no schema constraints, just
    "what specific things has this user shared that are relevant to <query>".
    """
    print("[retriever] raw_chat not yet implemented")
    return []


# --- CLI smoke test ---


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "拖延"
    src = {"type": "anchor", "endpoint": "http://localhost:8000", "search_path": "/limen/search"}
    mems = retrieve(src, query, n=3)
    print(f"got {len(mems)} memories for query: {query!r}")
    for i, m in enumerate(mems):
        print(f"\n[{i}] {m['timestamp'][:10]} {m['tag']}  ({m['memory_id']})")
        print(f"    {m['text'][:240]}")
