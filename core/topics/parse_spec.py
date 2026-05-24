"""Parse the 32-category / 480-title topic list from raccoon_notes_spec.md
into structured JSON.

Output: core/topics/topics_index.json

Schema:
    {
      "categories": [
        {
          "icon": "🧠",
          "name": "心理学",
          "subgroups": [
            {"name": "关于你自己", "titles": ["为什么你明明知道自己在拖延...", ...]},
            ...
          ]
        },
        ...
      ]
    }

Run once when spec.md changes:
    python core/topics/parse_spec.py
"""
from __future__ import annotations
import re
import json
from pathlib import Path


SPEC = Path.home() / "claude-home" / "office_limen" / "raccoon_notes_spec.md"
OUT = Path(__file__).parent / "topics_index.json"


# Bold category headers like "**🧠 心理学**"
CAT_RE = re.compile(r"\*\*([🧠💊🧬⚛️🧪🔢🌍🪐📜📖🏛️🐉🗺️💰⚖️🎨🎵🎬📚🗣️🔧🤖🏥🧩🏗️🌱🍳🐾🎮💔🧭🔑🔍])\s*([^\*]+?)\*\*")
# Sub-bullets: "- ..."
SUB_RE = re.compile(r"^- (.+)$", re.MULTILINE)
# Inline subgroup headers — appear as plain lines like "关于你自己" not preceded by - or **
# Heuristic: a line that is just plain Chinese text, no leading dash, no markdown markup, between bullets
SUBGROUP_RE = re.compile(r"^([一-鿿][^\n\-\*]{1,15})$", re.MULTILINE)


def main():
    text = SPEC.read_text()

    # Limit to topic list section
    m = re.search(r"## 话题列表.*?(?=## 分享功能)", text, re.DOTALL)
    if not m:
        raise SystemExit("could not find ## 话题列表 section")
    section = m.group(0)

    # Find all category bold headers + their positions
    cats_iter = list(CAT_RE.finditer(section))
    if not cats_iter:
        raise SystemExit("no category headers found")

    # Split section by category positions
    categories = []
    for i, cm in enumerate(cats_iter):
        icon = cm.group(1).strip()
        name = cm.group(2).strip()
        start = cm.end()
        end = cats_iter[i + 1].start() if i + 1 < len(cats_iter) else len(section)
        cat_block = section[start:end]

        # Within block: find subgroup headers + bullets
        # We track which bullets fall under which subgroup by walking line-by-line
        subgroups: list[dict] = []
        current_subgroup_name: str | None = None
        current_titles: list[str] = []
        for line in cat_block.splitlines():
            line = line.rstrip()
            if not line:
                continue
            bullet_m = re.match(r"^- (.+)$", line)
            if bullet_m:
                current_titles.append(bullet_m.group(1).strip())
                continue
            # Subgroup heuristic: short plain Chinese, no markdown
            if (re.match(r"^[一-鿿][一-鿿\s你我他她—]{1,14}$", line)
                    and not line.startswith("- ")
                    and not line.startswith("*")
                    and not line.startswith("#")):
                # Flush previous subgroup
                if current_titles:
                    subgroups.append({
                        "name": current_subgroup_name or "",
                        "titles": current_titles,
                    })
                    current_titles = []
                current_subgroup_name = line.strip()

        # Flush final
        if current_titles:
            subgroups.append({
                "name": current_subgroup_name or "",
                "titles": current_titles,
            })

        # Compute total titles for sanity log
        total = sum(len(sg["titles"]) for sg in subgroups)
        categories.append({
            "icon": icon,
            "name": name,
            "subgroups": subgroups,
            "_count": total,
        })

    grand_total = sum(c["_count"] for c in categories)

    out_doc = {"categories": categories, "total_titles": grand_total}
    OUT.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2))

    print(f"=== parsed {len(categories)} categories, {grand_total} titles total ===")
    for c in categories:
        sub_str = f", {len(c['subgroups'])} subgroups" if any(sg["name"] for sg in c["subgroups"]) else ""
        print(f"  {c['icon']} {c['name']}: {c['_count']} titles{sub_str}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
