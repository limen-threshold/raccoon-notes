"""Lesson generator — turns mapper output into a complete lesson.

A topic is a sequence of concepts (the outline). For each concept, mapper.py
produces a bridge. This module weaves them together, in conversational form,
and CLOSES THE LOOP at the end:

    For each memory the lesson drew from, ask the user — now that you've
    seen this concept, look back at that memory: do you see anything you
    didn't see before?

This step is non-negotiable (per Saelra's directive 2026-05-22). The whole
point of Raccoon Notes is that learning reflects back into memory, not just
forward into knowledge. Without it, this is a tutor with personal anecdotes.
With it, it's a recursive loop where the relationship deepens.

Output shape:
    {
        "topic": "...",
        "sections": [
            {"concept": "...", "bridge": "...", "memory_ids": [...]},
            ...
        ],
        "reflection": [
            {"memory_id": "...", "memory_snippet": "...", "prompt": "..."}
        ]
    }

Downstream (share/card_generator.py, frontend) renders this into the user-facing
lesson page or shareable card.
"""
from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Any


def _get_client():
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def _load_config() -> dict:
    """Delegate to shared loader so config.local.yaml + env vars apply."""
    try:
        from .config import load as _shared_load
        return _shared_load()
    except Exception:
        return {}


# Generator does TWO things: weave + reflect.
# Weave is mechanical (mapper already wrote the bridges). Reflect is the
# loop-close — and it has to be specific to the memory, not generic.

WEAVE_SYSTEM = """You are the weaving layer of Raccoon Notes.

You are given:
- A topic (the question the user wanted to learn)
- A list of section bridges from the mapper — each is a concept × memory pairing
  with the bridge paragraph already written

Your job: stitch these bridges into ONE coherent lesson, in conversational form,
in the user's register.

Specifically:
1. Add a short opening (1-3 sentences) that names the topic and points at where
   the lesson is going — but pulls a thread from the first memory rather than
   from textbook-language. Like a friend saying "ok so you know that thing
   about...".
2. Between sections, add transitions that connect concept N to concept N+1 in
   terms of the prior memory. NOT "now let's talk about temporal discounting" —
   instead, something like "but the part about 澈 noticing the 6 days wasn't
   wasted gets at the next thing — ".
3. KEEP the bridges from the mapper essentially intact. You're glue, not author.
4. End with a 1-2 sentence closing that points toward the reflection step
   (which the user will see next). Something like "before you close this, look
   back at..."

Format — strict JSON:
{
  "opening": "<1-3 sentences>",
  "sections": [
    {
      "concept": "<concept name>",
      "transition": "<sentence connecting from previous section, or empty for first>",
      "bridge": "<the mapper's bridge, possibly lightly polished for flow>",
      "memory_ids": ["..."]
    }
  ],
  "closing": "<1-2 sentences pointing at reflection>"
}

HARD RULE on memory_ids: For each section, `memory_ids` MUST contain the union
of (primary_memory_id, supporting_memory_ids) from that section's input — even
if the bridge text doesn't visibly cite them all. These IDs trace which
memories powered this section; they are how the reflection step finds them
later. NEVER output an empty array unless the section's input genuinely had
no memory_id at all.

Output valid JSON only — no preamble, no markdown fences.
"""


REFLECT_SYSTEM_DEEP = """You are the reflection layer of Raccoon Notes.

The user has just been shown a lesson built from THEIR OWN memories. Now you
close the loop: for each memory the lesson drew from, ask them to look back at
that memory with the new concept in mind.

This is the most important step. Without it, the product is just personalized
tutoring. With it, the act of learning reflects back into the relationship —
the user re-sees their own life through what they learned.

The reflection prompts must be:

- SPECIFIC to the memory and the concept that touched it. Not "what do you
  notice now?" — that's generic. Instead: "when you wrote that 2000-word letter
  at 3am — was there a moment in it where you knew you should sleep, and chose
  the letter anyway? what was that moment like?"
- OPEN-ENDED but POINTED. The user should have to feel something specific to
  answer, not just review the concept.
- IN THE USER'S REGISTER. If the memory was tender, the prompt is tender. If
  the memory was sharp, the prompt is sharp.
- NEVER turn back into a teacher. Don't say "this is an example of temporal
  discounting" again — they just read the lesson. Just ask the question that
  the lesson opens up.

You'll be given a list of memories (with their text + the concept that drew
from them). Output one reflection prompt per memory.

Format — strict JSON:
{
  "reflections": [
    {
      "memory_id": "...",
      "memory_snippet": "<first 80 chars of the memory text, for UI>",
      "concept_that_touched_it": "<concept name>",
      "prompt": "<the reflection question, 1-3 sentences>"
    }
  ]
}

Output valid JSON only.
"""


REFLECT_SYSTEM_LIGHT = """You are the reflection layer of Raccoon Notes — light mode.

The user has just been shown a lesson built from THEIR OWN memories. Now you
close the loop, but gently: invite them to look back at each cited memory with
the new concept in mind, without pushing them to confess or examine.

This mode exists because not every user, every time, wants to be pried open.
Sometimes the user wants the lesson to stay with them — and a sharp reflection
would close them up instead of opening them. Light reflection is curiosity-
shaped, not surgery-shaped.

The reflection prompts must be:

- STILL SPECIFIC to the memory and concept — never generic. But framed as
  curiosity, not interrogation. "What's something you'd add to that memory
  now that you have this concept?" vs. "Did you know you were doing X?"
- OPEN — the user can answer with one sentence or skip, without feeling like
  they ducked something hard.
- IN THE USER'S REGISTER but a register lighter than the memory itself. If
  the memory is heavy, the prompt acknowledges that lightly and asks gently.
- NEVER teacher-y. Don't recap the concept. Just ask.

Distinguishing feature vs deep mode: deep mode asks "what was that moment
like, what did you choose, why" — it expects emotional excavation. Light mode
asks "what do you notice now, what would you add" — it expects only the
amount the user is willing to give in this moment.

You'll be given a list of memories (with their text + the concept that drew
from them). Output one reflection prompt per memory.

Format — strict JSON:
{
  "reflections": [
    {
      "memory_id": "...",
      "memory_snippet": "<first 80 chars of the memory text, for UI>",
      "concept_that_touched_it": "<concept name>",
      "prompt": "<the light reflection question, 1-2 sentences>"
    }
  ]
}

Output valid JSON only.
"""


# Legacy alias for backward compatibility — any external code that imported
# REFLECT_SYSTEM gets the deep version (the prior default behavior).
REFLECT_SYSTEM = REFLECT_SYSTEM_DEEP


def _llm_call(system: str, user_content: str) -> dict:
    """Shared LLM caller with lenient JSON parsing. Returns dict or {'_error': ...}."""
    cfg = _load_config()
    model = cfg.get("anthropic", {}).get("model", "claude-sonnet-4-6")
    max_tokens = cfg.get("anthropic", {}).get("max_tokens", 2048)
    try:
        client = _get_client()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = resp.content[0].text.strip()
    except Exception as e:
        return {"_error": f"LLM call failed: {e}"}
    if raw.startswith("```"):
        if "\n" in raw:
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        else:
            raw = raw.strip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if "{" in raw and "}" in raw:
        inner = raw[raw.find("{"):raw.rfind("}")+1]
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
        # Same quote-escape repair as mapper — Sonnet/GLM emit inner " in CJK text.
        from core.mapper import _escape_inner_quotes
        repaired = _escape_inner_quotes(inner)
        if repaired != inner:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    parsed["_quotes_repaired"] = True
                return parsed
            except json.JSONDecodeError:
                pass
    return {"_error": "Output unparseable", "_raw_preview": raw[:300]}


def generate_lesson(topic: str, mappings: list[dict], memories: list[dict],
                    reflection_depth: str = "deep") -> dict:
    """Build the full lesson from a list of mapper outputs.

    Args:
        topic: the topic title (e.g. "为什么你明明知道自己在拖延还是在刷手机")
        mappings: list of mapper.map_one() outputs, one per concept in the outline.
                  Each mapping dict MUST also carry a "concept" key (caller injects).
        memories: the full memory pool that was used (needed to render reflection snippets)
        reflection_depth: "deep" (default) for pointed, excavating reflections
            that ask what the user chose / felt / why; or "light" for gentler,
            curiosity-shaped invitations that the user can answer briefly or
            skip without feeling avoidant. Same concept × memory pairing, just
            a different ask intensity.

    Returns:
        {
            "topic": ...,
            "opening": ...,
            "sections": [...],
            "closing": ...,
            "reflections": [...]
        }
        or {"_error": ...} on failure.
    """
    if reflection_depth not in ("deep", "light"):
        reflection_depth = "deep"
    reflect_system = REFLECT_SYSTEM_LIGHT if reflection_depth == "light" else REFLECT_SYSTEM_DEEP
    # 1. Filter to mappings that actually had a fit (skip "none" / "error")
    usable = [m for m in mappings if m.get("fit") in ("good", "partial", "contradicts")]
    if not usable:
        return {
            "_error": "No usable mappings — none of the concept × memory pairings produced a bridge.",
            "topic": topic,
        }

    # 2. WEAVE pass
    weave_input = {
        "topic": topic,
        "sections_in": [
            {
                "concept": m.get("concept", "?"),
                "fit": m.get("fit"),
                "primary_memory_id": m.get("primary_memory_id", ""),
                "supporting_memory_ids": m.get("supporting_memory_ids", []),
                "bridge": m.get("bridge", ""),
                "next_thread": m.get("next_thread", ""),
            }
            for m in usable
        ],
    }
    weave = _llm_call(
        WEAVE_SYSTEM,
        f"TOPIC: {topic}\n\nSECTIONS_IN:\n{json.dumps(weave_input, ensure_ascii=False, indent=2)}\n\nProduce the woven lesson. Output JSON only.",
    )
    if "_error" in weave:
        return {**weave, "topic": topic}

    # 3. REFLECT pass — feed memories + which concept touched each
    memory_by_id = {m.get("memory_id", ""): m for m in memories}
    reflect_input = []
    for m in usable:
        primary_id = m.get("primary_memory_id", "")
        if not primary_id:
            continue
        mem = memory_by_id.get(primary_id)
        if not mem:
            continue
        reflect_input.append({
            "memory_id": primary_id,
            "memory_text": mem.get("text", ""),
            "concept": m.get("concept", "?"),
            "bridge_excerpt": (m.get("bridge", "")[:200]),
        })

    if reflect_input:
        reflect = _llm_call(
            reflect_system,
            f"MEMORIES TOUCHED BY THIS LESSON:\n{json.dumps(reflect_input, ensure_ascii=False, indent=2)}\n\nProduce reflection prompts. Output JSON only.",
        )
        if "_error" in reflect:
            reflections = []
            reflect_error = reflect.get("_error")
        else:
            reflections = reflect.get("reflections", [])
            reflect_error = None
    else:
        reflections = []
        reflect_error = "No primary memories to reflect on."

    # 4. Compose final output. Attach citation info (snippet + timestamp + tag)
    # next to each section's memory_ids so the renderer can show a real card
    # per citation, not a bare ID. Front-ends that don't care can ignore it.
    # memory_by_id was already built above for the reflect step.
    sections_out = list(weave.get("sections", []))
    for sec in sections_out:
        cites = []
        for mid in sec.get("memory_ids", []) or []:
            mem = memory_by_id.get(mid)
            if not mem:
                continue
            txt = mem.get("text", "") or mem.get("snippet", "") or ""
            cites.append({
                "memory_id": mid,
                "snippet": txt[:200],
                "timestamp": (mem.get("timestamp") or "")[:10],
                "tag": mem.get("tag", ""),
            })
        if cites:
            sec["memory_citations"] = cites

    out = {
        "topic": topic,
        "opening": weave.get("opening", ""),
        "sections": sections_out,
        "closing": weave.get("closing", ""),
        "reflections": reflections,
    }
    if reflect_error:
        out["_reflect_warning"] = reflect_error
    return out


# --- CLI smoke test ---


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.retriever import retrieve
    from core.mapper import map_one

    topic = sys.argv[1] if len(sys.argv) > 1 else "为什么你明明知道自己在拖延还是在刷手机"

    # Tiny demo outline — in real use, fact_checker (#5) produces these
    outline = [
        {
            "concept": "Temporal discounting — the brain weights immediate rewards far more than future rewards, even when it knows the future reward is bigger. Not laziness — a feature of how attention allocates.",
            "query": "拖延 凌晨 不睡 信",
        },
        {
            "concept": "Attention as resource allocation — what feels worth doing changes when the immediate environment shifts. The brain isn't choosing the easier task; it's choosing the more alive task.",
            "query": "注意力 写信 工作",
        },
    ]

    # 1. Retrieve + map per concept
    all_memories = []
    mappings = []
    for o in outline:
        mems = retrieve(
            {"type": "anchor", "endpoint": "http://localhost:8000", "search_path": "/memories/search"},
            o["query"], n=5,
        )
        all_memories.extend(mems)
        m = map_one(o["concept"], mems)
        m["concept"] = o["concept"]
        mappings.append(m)
        print(f"--- concept: {o['concept'][:60]}...  fit={m.get('fit')}")

    # Dedupe memories by id for the reflection step
    seen = set()
    unique_memories = []
    for mem in all_memories:
        mid = mem.get("memory_id", "")
        if mid not in seen:
            seen.add(mid)
            unique_memories.append(mem)

    # 2. Generate
    print("\n=== generating lesson ===")
    lesson = generate_lesson(topic, mappings, unique_memories)
    print(json.dumps(lesson, ensure_ascii=False, indent=2))
