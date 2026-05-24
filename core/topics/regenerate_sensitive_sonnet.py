"""Regenerate the 14 sensitive-topic outlines on Sonnet for side-by-side comparison.

Reads core/topics/sensitive_topics.json (curated list), pulls each title's
category/subgroup from topics_index.json, calls Sonnet with the SAME system
prompt, writes to core/topics/outlines_sonnet_compare/<slug>.json.

Use after the GLM batch to spot-check whether GLM's outline quality on
politically/clinically sensitive titles holds up against Sonnet on the same
prompt.

Usage:
    python core/topics/regenerate_sensitive_sonnet.py
"""
from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv
    for env_path in (
        Path(__file__).parent.parent.parent / ".env",
        Path.home() / "claude-home" / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

# Force Sonnet path regardless of env
os.environ["RACCOON_BATCH_MODEL"] = "sonnet"

from generate_outlines import (  # type: ignore
    OUTLINE_SYSTEM,
    _llm_call_anthropic,
    _slug,
    _escape_inner_quotes,
)


SCRIPT_DIR = Path(__file__).parent
INDEX = SCRIPT_DIR / "topics_index.json"
SENSITIVE = SCRIPT_DIR / "sensitive_topics.json"
OUT_DIR = SCRIPT_DIR / "outlines_sonnet_compare"
OUT_DIR.mkdir(exist_ok=True)


def _find_category_subgroup(idx: dict, title: str) -> tuple[str, str]:
    for cat in idx["categories"]:
        cat_name = f"{cat['icon']} {cat['name']}"
        for sg in cat["subgroups"]:
            if title in sg["titles"]:
                return cat_name, sg.get("name", "")
    return "", ""


def _parse(raw: str) -> dict:
    raw = raw.strip()
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
        candidate = raw[raw.find("{"):raw.rfind("}") + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        repaired = _escape_inner_quotes(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return {"_error": "unparseable", "_raw": raw[:400]}


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    idx = json.loads(INDEX.read_text())
    groups = json.loads(SENSITIVE.read_text())

    work = []
    for tag, titles in groups.items():
        for t in titles:
            cat, sg = _find_category_subgroup(idx, t)
            work.append({"tag": tag, "category": cat, "subgroup": sg, "title": t})

    print(f"=== {len(work)} sensitive titles → Sonnet ===\n")

    stats = {"ok": 0, "err": 0}
    for i, item in enumerate(work, 1):
        slug = _slug(item["title"])
        out_path = OUT_DIR / f"{slug}.json"
        user_content = (
            f"CATEGORY: {item['category']}\n"
            + (f"SUBGROUP: {item['subgroup']}\n" if item['subgroup'] else "")
            + f"TOPIC TITLE: {item['title']}\n\n"
            + "Produce the outline. JSON only."
        )

        t0 = time.time()
        try:
            raw = _llm_call_anthropic(OUTLINE_SYSTEM, user_content)
        except Exception as e:
            print(f"[{i}/{len(work)}] ❌ {item['title'][:40]}  ({e})")
            stats["err"] += 1
            continue
        dt = time.time() - t0

        parsed = _parse(raw)
        if "_error" in parsed or not parsed.get("concepts"):
            print(f"[{i}/{len(work)}] ❌ {item['title'][:40]}  (parse fail)")
            stats["err"] += 1
            out_path.write_text(json.dumps({
                "slug": slug, **item,
                "outline": [], "model": "claude-sonnet-4-6",
                "generated_at": datetime.utcnow().isoformat(),
                "fact_check_status": "generation_failed",
                "error": parsed.get("_error", "no concepts"),
                "raw_preview": parsed.get("_raw", "")[:300],
            }, ensure_ascii=False, indent=2))
            continue

        out = {
            "slug": slug,
            "tag": item["tag"],
            "category": item["category"],
            "subgroup": item["subgroup"],
            "title": item["title"],
            "outline": parsed["concepts"],
            "generated_at": datetime.utcnow().isoformat(),
            "model": "claude-sonnet-4-6",
            "fact_check_status": "pending",
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        stats["ok"] += 1
        print(f"[{i}/{len(work)}] ✅ {item['title'][:40]}  ({len(parsed['concepts'])} concepts, {dt:.1f}s)")
        time.sleep(0.4)

    print(f"\n=== summary: {stats['ok']} ok, {stats['err']} err ===")
    print(f"→ {OUT_DIR}")
    print("\nCompare with GLM versions:")
    print(f"  diff <(jq . {OUT_DIR}/<slug>.json) <(jq . core/topics/outlines/<slug>.json)")


if __name__ == "__main__":
    main()
