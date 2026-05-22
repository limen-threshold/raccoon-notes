"""Top-level pipeline: topic → lesson.

Both server/http_server.py and server/mcp_server.py call run_lesson() here.
Keeping the orchestration logic in one place means the two transports stay
thin and identical in behavior.

Flow:
    topic + outline + memory_source
        ↓
    for each concept in outline:
        retrieve(memory_source, concept.query)
        map_one(concept.concept, memories)
        ↓
    generate_lesson(topic, mappings, all_unique_memories)
        ↓
    {topic, opening, sections, closing, reflections}
"""
from __future__ import annotations
from typing import Any, Optional
from . import retriever
from . import mapper
from . import generator


DEFAULT_MEMORY_SOURCE = {
    "type": "anchor",
    "endpoint": "http://localhost:8000",
    "search_path": "/limen/search",
}


def run_lesson(
    topic: str,
    outline: list[dict],
    memory_source: Optional[dict] = None,
    n_memories_per_concept: int = 5,
) -> dict:
    """Run the full pipeline for one topic.

    Args:
        topic: human-readable topic title.
        outline: list of concept dicts. Each concept dict:
            {"concept": "<full description of the knowledge point>",
             "query":   "<keywords to retrieve memories for this concept>"}
            Caller (preset topics or fact_checker for custom topics) builds this.
        memory_source: dict per retriever.retrieve schema. Defaults to local Anchor.
        n_memories_per_concept: how many memories to pull per concept.

    Returns:
        lesson dict (see generator.generate_lesson) OR
        {"_error": "...", "topic": topic} if something went wrong upstream.
    """
    src = memory_source or DEFAULT_MEMORY_SOURCE

    if not outline:
        return {"_error": "outline is empty — need at least one concept", "topic": topic}

    all_memories: list[dict] = []
    mappings: list[dict] = []

    for concept_entry in outline:
        concept_text = concept_entry.get("concept", "")
        query = concept_entry.get("query") or concept_text
        if not concept_text:
            continue

        mems = retriever.retrieve(src, query, n=n_memories_per_concept)
        all_memories.extend(mems)

        m = mapper.map_one(concept_text, mems)
        # Caller convention: inject the concept name onto the mapping for downstream
        m["concept"] = concept_text
        mappings.append(m)

    # Dedupe memories by id for the reflection step
    seen: set[str] = set()
    unique_memories: list[dict] = []
    for mem in all_memories:
        mid = mem.get("memory_id", "")
        if mid and mid not in seen:
            seen.add(mid)
            unique_memories.append(mem)

    lesson = generator.generate_lesson(topic, mappings, unique_memories)

    # Attach diagnostics so the server can return debugging info on failures
    lesson.setdefault("_diagnostics", {})
    lesson["_diagnostics"]["concepts_in_outline"] = len(outline)
    lesson["_diagnostics"]["concepts_with_fit"] = sum(
        1 for m in mappings if m.get("fit") in ("good", "partial", "contradicts")
    )
    lesson["_diagnostics"]["unique_memories_used"] = len(unique_memories)
    lesson["_diagnostics"]["memory_source_type"] = src.get("type", "?")

    return lesson
