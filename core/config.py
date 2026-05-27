"""Shared config loader.

Load order (later sources override earlier):
  1. server/config.yaml         — checked into git, safe defaults
  2. server/config.local.yaml   — gitignored, private overrides
  3. environment vars           — RACCOON_<SECTION>_<KEY>, e.g.
                                   RACCOON_MEMORY_ENDPOINT, RACCOON_ANTHROPIC_MODEL

Anything missing falls back to a hard-coded minimal default.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
    },
    "memory": {
        "endpoint": "http://localhost:8000",
        "search_path": "/memories/search",
        "default_n_results": 5,
    },
    "tavily": {
        "max_results": 5,
    },
    "http": {
        "host": "0.0.0.0",
        "port": 8200,
    },
    "mcp": {
        "transport": "stdio",
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Overlay's keys win; nested dicts merged recursively."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _apply_env(cfg: dict) -> dict:
    """RACCOON_<SECTION>_<KEY> → cfg[section][key.lower()]."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    prefix = "RACCOON_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section not in out or not isinstance(out[section], dict):
            out[section] = {}
        # Try to int/float-coerce
        coerced: Any = env_val
        try:
            coerced = int(env_val)
        except ValueError:
            try:
                coerced = float(env_val)
            except ValueError:
                pass
        out[section][key] = coerced
    return out


_CACHED: dict | None = None


def load(force: bool = False) -> dict:
    """Load merged config. Cached after first call (set force=True to reread)."""
    global _CACHED
    if _CACHED is not None and not force:
        return _CACHED
    server_dir = Path(__file__).parent.parent / "server"
    cfg = _deep_merge(_DEFAULTS, _load_yaml(server_dir / "config.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(server_dir / "config.local.yaml"))
    cfg = _apply_env(cfg)
    _CACHED = cfg
    return cfg
