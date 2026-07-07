"""Step 3.3 helper: turn curated candidate papers into APPLIES edges in graph.json.

The APPLIES test (schema §3.5) -- "P runs M's mechanism UNMODIFIED and reports
>=1 quantitative result" -- cannot be decided by a script from an abstract alone;
it needs a human (or human+LLM) reading. So this tool does NOT auto-accept papers.
Instead it:

  1. loads papers_candidates.json (the ranked over-fetch buffer),
  2. drops any candidate already in graph.json as a seed/introducing paper,
  3. writes a REVIEW worksheet (papers_applies_review.json) with one row per
     candidate: its title/abstract/score + an empty `applies_methods` list and a
     `keep` flag for you to fill,
  4. on a second pass (--merge), reads the filled worksheet and writes the
     accepted papers + APPLIES edges into graph.json.

Usage:
  py tag_applies.py            # pass 1: build the review worksheet
  # ... you edit papers_applies_review.json: set keep=true and list method slugs
  py tag_applies.py --merge    # pass 2: merge accepted rows into graph.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root = parent of src/. graph.json is at root; the corpus/review JSON live
# in data/. Resolving from here keeps the tool runnable from any directory.
ROOT = Path(__file__).resolve().parent.parent
GRAPH = str(ROOT / "graph.json")
CANDIDATES = str(ROOT / "data" / "papers_candidates.json")
REVIEW = str(ROOT / "data" / "papers_applies_review.json")

# Valid method slugs, loaded from the graph so the worksheet can be checked.
def _method_slugs(g: dict) -> set[str]:
    return {m["slug"] for m in g["methods"]}


def _load(path: str) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def build_worksheet() -> None:
    g = _load(GRAPH)
    slugs = _method_slugs(g)
    known_papers = {p["arxiv_id"] for p in g["papers"]}
    # Also skip anything already recorded by title-less S2 id if present.
    try:
        candidates = _load(CANDIDATES)
    except FileNotFoundError:
        print(f"ERROR: {CANDIDATES} not found -- run fetch_papers.py first.")
        sys.exit(1)

    rows = []
    for c in candidates:
        if c.get("_role") == "seed":
            continue  # seeds are introducing papers, handled in 3.1/3.2
        # Best-effort arxiv id from externalIds; fall back to S2 paper_id.
        ext = (c.get("externalIds") or {})
        arxiv_id = ext.get("ArXiv") or c.get("paper_id")
        if arxiv_id in known_papers:
            continue
        rows.append({
            "paper_id": c.get("paper_id"),
            "arxiv_id": arxiv_id,
            "title": c.get("title"),
            "year": c.get("year"),
            "venue": c.get("venue"),
            "abstract": c.get("abstract"),
            "_score": c.get("_score"),
            # ---- fill these during review (per §3.5) ----
            "keep": None,            # true if it passes the APPLIES test
            "applies_methods": [],   # list of method slugs it runs unmodified
            "reason": "",            # one-line justification for the keep/reject call
            "confidence": None,      # 0.0-1.0; low = borderline unmodified-vs-variant
            "evidence": "",          # phrase(s) from the abstract that decided it
        })

    _dump(REVIEW, rows)
    print(f"Wrote {REVIEW} with {len(rows)} candidate rows to review.")
    print(f"Valid method slugs: {sorted(slugs)}")
    print("For each paper that runs a tracked method unmodified and reports a")
    print("result, set keep=true and list the method slug(s) in applies_methods.")
    print("Then: py tag_applies.py --merge")


def merge() -> None:
    g = _load(GRAPH)
    slugs = _method_slugs(g)
    known_papers = {p["arxiv_id"] for p in g["papers"]}
    rows = _load(REVIEW)

    added_papers = 0
    added_edges = 0
    for r in rows:
        if not r.get("keep"):
            continue
        aid = r.get("arxiv_id")
        if not aid:
            print(f"  skip (no arxiv_id): {r.get('title')!r}")
            continue
        bad = [m for m in r.get("applies_methods", []) if m not in slugs]
        if bad:
            print(f"  ERROR unknown method slug(s) {bad} for {aid}; fix and re-run")
            sys.exit(1)
        if aid not in known_papers:
            g["papers"].append({
                "arxiv_id": aid,
                "title": r.get("title"),
                "arxiv_v1_date": None,  # optional for APPLIES papers; fill if known
                "year": r.get("year"),
                "venue": r.get("venue"),
            })
            known_papers.add(aid)
            added_papers += 1
        for m in r.get("applies_methods", []):
            g["edges"]["APPLIES"].append({"from_paper": aid, "to_method": m})
            added_edges += 1

    g["_status"]["3.3_applies"] = "COMPLETE"
    _dump(GRAPH, g)
    print(f"Merged: +{added_papers} papers, +{added_edges} APPLIES edges into {GRAPH}.")
    print("Now run: py validate_graph.py")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        build_worksheet()
