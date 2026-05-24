"""Batch-generate preset outlines for all 480 topic titles.

For each title in topics_index.json, produce an outline (list of 3-5 concepts)
that a Raccoon Notes lesson would walk through. The output goes to
`core/topics/outlines/<slug>.json` — one file per topic.

Schema per output file:
    {
        "slug": "...",
        "category": "🧠 心理学",
        "subgroup": "关于你自己" | "",
        "title": "为什么你明明知道自己在拖延还是在刷手机",
        "outline": [
            {"concept": "<full description>", "query": "<retrieval keywords>"},
            ...
        ],
        "generated_at": "2026-05-22T...",
        "model": "claude-sonnet-4-6",
        "fact_check_status": "pending"  # later: "verified" | "needs_review" | "rejected"
    }

Usage:
    python core/topics/generate_outlines.py               # generate all missing
    python core/topics/generate_outlines.py --limit 5     # smoke test
    python core/topics/generate_outlines.py --force       # regenerate even if exists
    python core/topics/generate_outlines.py --category "心理学"  # one category only

Slow on purpose — runs sequentially with a small sleep between calls. Cost
estimate: 480 × ~$0.01 = ~$5 total. Resumable: skips topics that already have
an outline file.
"""
from __future__ import annotations
import os
import sys
import json
import time
import re
import argparse
import hashlib
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


import urllib.request


SCRIPT_DIR = Path(__file__).parent
INDEX = SCRIPT_DIR / "topics_index.json"
OUTLINES_DIR = SCRIPT_DIR / "outlines"
OUTLINES_DIR.mkdir(exist_ok=True)

# Default backend: GLM 5.1 via Zhipu (cheap, Chinese-native, thinking-on by default).
# Fallback: Anthropic Sonnet 4.6. Set RACCOON_BATCH_MODEL=sonnet to use Anthropic.
MODEL = os.getenv("RACCOON_BATCH_MODEL", "glm-5.1")
GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MAX_TOKENS = 4096   # GLM 5.1 burns ~500-800 reasoning tokens before visible output


def _llm_call_glm(system: str, user_content: str) -> str:
    """Call GLM 5.1 via Zhipu. Returns raw content string. Raises on transport error."""
    key = os.getenv("ZHIPU_API_KEY")
    if not key:
        raise RuntimeError("ZHIPU_API_KEY not set")
    payload = {
        "model": "glm-5.1",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": GLM_MAX_TOKENS,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        GLM_URL,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def _llm_call_anthropic(system: str, user_content: str) -> str:
    """Fallback path. Returns raw content string."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text


OUTLINE_SYSTEM = """You write outlines for Raccoon Notes — short lessons that use a user's personal memories to explain a concept.

You will be given a clickbait-style topic title (often a "why does X happen" question). Your job: produce a 3-5 concept outline that, in order, would actually answer the question.

Each concept is ONE knowledge point — specific enough that a user's memory could illustrate it concretely. Vague concepts ("the brain is complex") are useless. Sharp concepts ("temporal discounting: brain weights immediate rewards more than future, even knowing the future is bigger") work.

For each concept ALSO produce a retrieval query — 4-12 keywords (mix Chinese + English ok) that a memory search would use to find experiences that touch this concept. Specific keywords beat generic ones. Use the user's likely vocabulary, not textbook terms.

Output — strict JSON, single object only, no preamble, no markdown fences:

{
  "concepts": [
    {
      "concept": "<one knowledge point, 1-3 sentences, in the user's register>",
      "query": "<retrieval keywords, space-separated>"
    },
    ...
  ]
}

HARD RULES:
1. 3-5 concepts per outline. Not more, not less. Each builds on the previous.
2. Each concept must be a real, falsifiable claim — not a vibe.
3. Order matters. First concept names the surprise; middle concepts unpack it; last concept lands somewhere with weight.
4. Don't moralize. Don't pad with "this is important because". Just say what the concept is.
5. Output valid JSON only.
"""


def _slug(title: str) -> str:
    """Deterministic ASCII-safe filename for a topic title."""
    # Strip emoji, punctuation, whitespace; if too aggressive, hash fallback.
    s = re.sub(r"[\W\s]+", "_", title.strip()).strip("_")
    if len(s) > 80 or not s:
        # Fall back to hash if the title is all-symbol or extremely long
        s = "topic_" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
    return s


def generate_one(title: str, category: str, subgroup: str = "") -> dict:
    """Call the configured LLM for one title. Returns parsed outline dict or {'_error': ...}."""
    user_content = (
        f"CATEGORY: {category}\n"
        + (f"SUBGROUP: {subgroup}\n" if subgroup else "")
        + f"TOPIC TITLE: {title}\n\n"
        + "Produce the outline. JSON only."
    )
    try:
        if MODEL.startswith("glm"):
            raw = _llm_call_glm(OUTLINE_SYSTEM, user_content).strip()
        else:
            raw = _llm_call_anthropic(OUTLINE_SYSTEM, user_content).strip()
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
    # lenient #1: outermost {...}
    if "{" in raw and "}" in raw:
        candidate = raw[raw.find("{"):raw.rfind("}")+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # lenient #2: escape unescaped " inside string values.
        # Sonnet in Chinese context frequently emits e.g.
        #     "concept": "...前额叶皮层负责计划和"知道"，边缘系统..."
        # The inner `"知道"` should have been `\"知道\"`. Naively s/"/\"/g would
        # break the structural quotes, so we use a heuristic: a `"` is structural
        # only if it sits at a JSON-syntax boundary (preceded by ", : [ { whitespace,
        # or followed by , : ] } whitespace). Everything else is an in-string quote
        # that needs escaping.
        repaired = _escape_inner_quotes(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return {"_error": "unparseable", "_raw": raw[:300]}


def _escape_inner_quotes(s: str) -> str:
    """Best-effort escape of unescaped `"` inside JSON string values.

    Walks char-by-char tracking whether we're inside a string. When we hit a `"`
    that's not preceded by `\` and not at a structural position, replace with `\\"`.

    Structural positions = the `"` is the start/end of a JSON string token, i.e.
    the surrounding context is JSON syntax (after `:`, `,`, `[`, `{`, or after
    whitespace following those; or before `:`, `,`, `]`, `}` after whitespace).
    """
    out = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            # preserve escape sequences verbatim
            out.append(s[i:i+2])
            i += 2
            continue
        if c == '"':
            if not in_string:
                # opening quote of a JSON string
                in_string = True
                out.append(c)
                i += 1
                continue
            # closing? look ahead for structural punctuation skipping whitespace
            j = i + 1
            while j < len(s) and s[j] in " \t\n\r":
                j += 1
            next_char = s[j] if j < len(s) else ""
            if next_char in ",:]}" or next_char == "":
                # legit closing quote
                in_string = False
                out.append(c)
                i += 1
            else:
                # quote inside the string content — escape it
                out.append('\\"')
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of topics to process")
    ap.add_argument("--force", action="store_true", help="regenerate even if outline file exists")
    ap.add_argument("--category", type=str, default=None, help="only this category (partial match)")
    ap.add_argument("--sleep", type=float, default=0.5, help="delay between calls, seconds")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    if not INDEX.exists():
        print(f"missing {INDEX} — run parse_spec.py first", file=sys.stderr)
        sys.exit(1)

    idx = json.loads(INDEX.read_text())
    categories = idx["categories"]

    # Flatten to list of {category, subgroup, title}
    work: list[dict] = []
    for cat in categories:
        cat_name = f"{cat['icon']} {cat['name']}"
        if args.category and args.category not in cat_name:
            continue
        for sg in cat["subgroups"]:
            for title in sg["titles"]:
                work.append({
                    "category": cat_name,
                    "subgroup": sg.get("name", ""),
                    "title": title,
                })

    if args.limit:
        work = work[: args.limit]

    print(f"=== {len(work)} topics queued ===\n")

    stats = {"generated": 0, "skipped": 0, "errors": 0}
    for i, item in enumerate(work, 1):
        slug = _slug(item["title"])
        out_path = OUTLINES_DIR / f"{slug}.json"
        if out_path.exists() and not args.force:
            stats["skipped"] += 1
            continue

        t0 = time.time()
        result = generate_one(item["title"], item["category"], item["subgroup"])
        dt = time.time() - t0

        if "_error" in result:
            stats["errors"] += 1
            print(f"[{i}/{len(work)}] ❌ {item['title'][:50]}  ({result['_error'][:60]})")
            # Still write an error file so we know it was attempted
            out_path.write_text(json.dumps({
                "slug": slug, **item,
                "outline": [],
                "generated_at": datetime.utcnow().isoformat(),
                "model": MODEL,
                "fact_check_status": "generation_failed",
                "error": result.get("_error", "unknown"),
            }, ensure_ascii=False, indent=2))
            continue

        concepts = result.get("concepts", [])
        if not concepts or not isinstance(concepts, list):
            stats["errors"] += 1
            print(f"[{i}/{len(work)}] ❌ {item['title'][:50]}  (no concepts)")
            continue

        out = {
            "slug": slug,
            "category": item["category"],
            "subgroup": item["subgroup"],
            "title": item["title"],
            "outline": concepts,
            "generated_at": datetime.utcnow().isoformat(),
            "model": MODEL,
            "fact_check_status": "pending",
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        stats["generated"] += 1
        print(f"[{i}/{len(work)}] ✅ {item['title'][:50]}  ({len(concepts)} concepts, {dt:.1f}s)")

        time.sleep(args.sleep)

    print(f"\n=== summary ===")
    print(f"  generated: {stats['generated']}")
    print(f"  skipped:   {stats['skipped']}")
    print(f"  errors:    {stats['errors']}")


if __name__ == "__main__":
    main()
