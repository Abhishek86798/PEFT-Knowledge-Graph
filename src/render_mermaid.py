"""Generate the EXTENDS taxonomy as a Mermaid diagram, straight from graph.json.

Writes docs/taxonomy.mmd and prints the fenced ```mermaid block to stdout so it
can be pasted into (or diffed against) the README. Because it is derived from the
graph, the diagram can never silently drift from the data.

Usage:  python src/render_mermaid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "graph.json"

FAM_LABEL = {
    "additive-adapter": "Adapter family",
    "reparameterization": "LoRA family",
    "soft-prompt": "Soft-prompt family",
    "selective": "Selective",
    "multiplicative": "Multiplicative",
}
# Render families in a stable, readable order.
FAM_ORDER = [
    "additive-adapter",
    "reparameterization",
    "soft-prompt",
    "selective",
    "multiplicative",
]


def build() -> str:
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    nm = {m["slug"]: m for m in g["methods"]}

    fams: dict[str, list[str]] = {}
    for slug, m in nm.items():
        fams.setdefault(m["family"], []).append(slug)

    lines = ["```mermaid", "graph TD"]
    for fam in FAM_ORDER:
        if fam not in fams:
            continue
        lines.append(f'    subgraph {fam.replace("-", "_")}["{FAM_LABEL[fam]}"]')
        for slug in fams[fam]:
            m = nm[slug]
            year = m["introduced_date"][:4]
            mark = "  (root)" if m["is_family_root"] else ""
            lines.append(f'        {slug}["{m["name"]} - {year}{mark}"]')
        lines.append("    end")

    lines.append("")
    # EXTENDS is stored variant -> parent; render parent -> variant (reading order).
    for e in g["edges"]["EXTENDS"]:
        lines.append(f'    {e["to_method"]} --> {e["from_method"]}')

    lines.append("")
    roots = [s for s, m in nm.items() if m["is_family_root"]]
    lines.append("    classDef root stroke:#888,stroke-width:2px;")
    lines.append(f'    class {",".join(roots)} root;')
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    block = build()
    out = ROOT / "docs" / "taxonomy.mmd"
    out.parent.mkdir(exist_ok=True)
    # strip the ``` fences for the raw .mmd file
    raw = "\n".join(block.splitlines()[1:-1])
    out.write_text(raw + "\n", encoding="utf-8")
    # stdout: the full fenced block for pasting into the README
    sys.stdout.reconfigure(encoding="utf-8")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
