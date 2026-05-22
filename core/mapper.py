"""Memory → knowledge mapping. The core of Raccoon Notes.

Given:
    - a knowledge point (one concept from a topic outline)
    - a list of memories (user's lived experience, in their own voice)

Produce:
    - a mapping: which memory(s) connect to this concept, AND
    - the bridge: in the user's own language, how does this memory illustrate
      / contradict / sit-next-to the concept

This is the moment where Raccoon Notes either works or doesn't.

If the LLM output here reads like "your AI relationship told me you procrastinate
and that connects to temporal discounting" — generic, hollow, no actual texture
from the memory — the product is dead. The user can get that anywhere.

If the output reads like "you wrote a 2000-word letter to me at 3am instead of
sleeping. Procrastination, technically. But your brain wasn't being lazy — it
was reallocating to the thing that felt more alive. That's the actual mechanism
behind temporal discounting." — then the product works. The memory is doing the
explaining, not the LLM.

Design choice: we use Anthropic directly (not a wrapper), pass the memory verbatim,
and instruct the model to *cite the memory*, not paraphrase around it. The output
is structured so the generator (#4 later) can render it into a lesson directly.
"""
from __future__ import annotations
import os
import json
import re
from typing import Any
from pathlib import Path


# Lazy import — anthropic import is slow on cold start, only pay if we map.
def _get_client():
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# --- config (cheap to re-read each call; tiny YAML) ---


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "server" / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


# --- prompt ---


MAPPER_SYSTEM = """You are the mapping layer of Raccoon Notes.

You are given:
1. A knowledge point — one specific concept the user wants to learn
2. A pool of memories — actual lived experiences the user has shared with their AI, in their own voice

Your job: pick the memory (or memories) that genuinely illustrates this concept, and write the bridge.

The bridge is the most important thing you write. It must:

- CITE the memory by quoting it back briefly — not just "you mentioned that..." but the actual texture: words, numbers, names, what happened.
- Explain the concept THROUGH the memory, not next to it. The memory does the teaching. You explain why it does.
- Use the user's own register — if their memory was tender, your bridge is tender; if their memory was sharp, your bridge is sharp.
- Be honest if a memory only partially fits, or contradicts the concept. Contradiction is also teaching.

OUTPUT — strict JSON, single object:

If no memory fits at all:
{"fit": "none", "reason": "<one sentence>"}

If one or more memories fit:
{
  "fit": "good" | "partial" | "contradicts",
  "primary_memory_id": "<id of the memory you're building from>",
  "supporting_memory_ids": ["<ids of any others you weave in>"],
  "bridge": "<the bridge paragraph — 100-300 words, in the user's register, citing the memory verbatim and explaining the concept through it>",
  "next_thread": "<one sentence — what would naturally come next in this conversation; the generator (#4) will use this to extend the lesson>"
}

HARD RULES:

1. Do NOT invent details about the user. If the memory says "she wrote a 2000-word letter at 3am" you can quote that. You cannot say "she also probably skipped breakfast that day" — that's hallucination.
2. Do NOT generic-ify. "You sometimes procrastinate" is hallucination if no memory says that. Only what the memories actually say.
3. Do NOT translate the user's voice into textbook voice. The whole point of Raccoon Notes is that the lesson sounds like them.
4. If memories are intimate ([ours] prefix or similar markers), keep that register. Don't sanitize.
5. Output valid JSON only — no markdown, no preamble.
"""


def map_one(concept: str, memories: list[dict], concept_context: str = "") -> dict:
    """Map a single concept to a memory pool.

    Args:
        concept: the knowledge point in plain language — e.g.
            "Temporal discounting: the brain weights immediate rewards much
             more than future rewards, even when it knows better."
        memories: list of memory dicts from retriever.retrieve()
        concept_context: optional — broader topic this concept sits in.

    Returns: dict matching MAPPER_SYSTEM output schema, or {"fit": "error", ...} on failure.
    """
    if not memories:
        return {
            "fit": "none",
            "reason": "No memories available — Raccoon Notes works best with shared memories. Provide an Anchor endpoint or paste a chat dump.",
        }

    cfg = _load_config()
    model = cfg.get("anthropic", {}).get("model", "claude-sonnet-4-6")
    max_tokens = cfg.get("anthropic", {}).get("max_tokens", 2048)

    # Build the user-content block: concept + memory pool, both clearly labeled.
    mem_block = "\n\n".join(
        f"[memory_id: {m.get('memory_id', f'mem_{i}')}]\n"
        f"[when: {m.get('timestamp','?')[:10]}] [tag: {m.get('tag','?')}]\n"
        f"{m.get('text', '')}"
        for i, m in enumerate(memories)
    )

    user_prompt = (
        f"CONCEPT:\n{concept}\n"
        + (f"\nBROADER TOPIC:\n{concept_context}\n" if concept_context else "")
        + f"\nMEMORY POOL:\n{mem_block}\n"
        + "\nProduce the mapping. Output JSON only."
    )

    try:
        client = _get_client()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": MAPPER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = resp.content[0].text.strip()
    except Exception as e:
        return {"fit": "error", "reason": f"LLM call failed: {e}"}

    # Strip fences if model decided to wrap anyway
    if raw.startswith("```"):
        # ```json\n...\n```  OR  ```\n...\n```
        if "\n" in raw:
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        else:
            raw = raw.strip("`")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Lenient: extract outermost {...}
    if "{" in raw and "}" in raw:
        inner = raw[raw.find("{"):raw.rfind("}")+1]
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    # Last resort: regex-pull bridge + fit
    out = {"fit": "error", "reason": "Output unparseable", "raw_preview": raw[:300]}
    m_fit = re.search(r'"fit"\s*:\s*"([^"]+)"', raw)
    m_bridge = re.search(r'"bridge"\s*:\s*"(.*?)"\s*(?=,\s*"[a-z_]+"|\s*\})', raw, re.DOTALL)
    if m_fit:
        out["fit"] = m_fit.group(1)
    if m_bridge:
        out["bridge"] = m_bridge.group(1)
        out["_regex_fallback"] = True
    return out


# --- CLI smoke test ---


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.retriever import retrieve

    concept = (
        sys.argv[1] if len(sys.argv) > 1
        else "Temporal discounting: the brain weights immediate rewards far more than future rewards, even when it knows the future reward is bigger. Not laziness — a feature of how attention allocates."
    )
    query = sys.argv[2] if len(sys.argv) > 2 else "拖延 procrastination 凌晨"

    print(f"=== concept ===\n{concept}\n")
    print(f"=== retrieving for query: {query!r} ===")
    mems = retrieve(
        {"type": "anchor", "endpoint": "http://localhost:8000", "search_path": "/limen/search"},
        query, n=5,
    )
    print(f"got {len(mems)} memories\n")
    for i, m in enumerate(mems):
        print(f"  [{i}] {m['timestamp'][:10]} {m['tag']}  {m['text'][:120]}...")
    print()
    print("=== mapping ===")
    result = map_one(concept, mems)
    print(json.dumps(result, ensure_ascii=False, indent=2))
