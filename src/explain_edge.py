"""explain_edge.py -- "why does this edge exist?"

Every edge in this graph carries its own justification, but it's scattered across
different places depending on edge type: EXTENDS carries a reduction-test `note`
directly on the edge; COMPARED_AGAINST and EVALUATED_ON carry an `evidence_paper`
(and COMPARED_AGAINST a `note` on which results-table rows back it); INTRODUCES is
definitional (schema says exactly one per method); APPLIES has no per-edge note in
graph.json, but the tagging decision that put it there IS recorded, in
data/papers_applies_review.json, as a `reason` + `confidence` + `evidence` triple.

This tool answers "why does X --EDGE--> Y exist?" by looking up the right source
for that edge type and rendering its justification. It surfaces no new facts -- it
is a lookup, not a reasoner -- but it turns the auditability the schema already
promises ("every edge has a falsifiable test") into something a user can query in
one command instead of grepping graph.json / the review file by hand.

Usage:
  py explain_edge.py --from LoRA --to AdaLoRA --edge EXTENDS
  py explain_edge.py --from AdaLoRA --to LoRA --edge COMPARED_AGAINST
  py explain_edge.py --from Adapters --to GLUE --edge EVALUATED_ON
  py explain_edge.py --from-paper 1902.00751 --to Adapters --edge INTRODUCES
  py explain_edge.py --list-edges LoRA          # list every edge touching a method
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default stdout to the system codepage (e.g. cp1252), which
# cannot encode characters like the em-dash that appear in review-worksheet
# prose. Force UTF-8 so output never raises UnicodeEncodeError when printed or
# redirected, on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
GRAPH = str(ROOT / "graph.json")
APPLIES_REVIEW = str(ROOT / "data" / "papers_applies_review.json")

EDGE_TYPES = ["INTRODUCES", "EXTENDS", "COMPARED_AGAINST", "EVALUATED_ON", "APPLIES"]


def load_graph(path: str = GRAPH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_applies_review(path: str = APPLIES_REVIEW) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _name_maps(g: dict) -> tuple[dict, dict, dict]:
    """slug/id -> name, and the reverse name -> slug/id, for methods/papers/benchmarks."""
    method_name = {m["slug"]: m["name"] for m in g["methods"]}
    method_slug = {v: k for k, v in method_name.items()}
    # papers are keyed by arxiv id / hash id in graph.json; no separate name field
    # observed, so paper "name" is its id -- callers pass ids for papers.
    benchmark_name = {b.get("slug", b.get("id", "")): b.get("name", b.get("slug", ""))
                      for b in g.get("benchmarks", [])}
    benchmark_slug = {v: k for k, v in benchmark_name.items()}
    return method_name, method_slug, benchmark_slug


def _resolve_method(token: str, method_slug: dict, method_name: dict) -> str | None:
    if token in method_name:  # already a slug
        return token
    if token in method_slug:  # a display name
        return method_slug[token]
    # case-insensitive fallback
    low = token.lower()
    for name, slug in method_slug.items():
        if name.lower() == low:
            return slug
    return None


def _resolve_benchmark(token: str, benchmark_slug: dict) -> str | None:
    if token in benchmark_slug:
        return benchmark_slug[token]
    low = token.lower()
    for name, slug in benchmark_slug.items():
        if name.lower() == low or slug.lower() == low:
            return slug
    return token if token in benchmark_slug.values() else None


# --------------------------------------------------------------------------- #
# Per-edge-type explanation
# --------------------------------------------------------------------------- #


def explain_extends(g: dict, from_slug: str, to_slug: str) -> dict | None:
    for e in g["edges"]["EXTENDS"]:
        if e["from_method"] == from_slug and e["to_method"] == to_slug:
            return {
                "found": True,
                "edge": "EXTENDS",
                "justification": e["note"],
                "source": "curated: reduction-test note on the edge in graph.json",
                "test_applied": ("Reduction test (schema.md): A is a variant of B if "
                                 "disabling/fixing A's novel components recovers B's "
                                 "trainable mechanism unchanged."),
            }
    return None


def explain_compared_against(g: dict, from_slug: str, to_slug: str) -> dict | None:
    for e in g["edges"]["COMPARED_AGAINST"]:
        if e["from_method"] == from_slug and e["to_method"] == to_slug:
            return {
                "found": True,
                "edge": "COMPARED_AGAINST",
                "justification": e.get("note", ""),
                "evidence_paper": e.get("evidence_paper"),
                "untracked_baselines_in_same_table": e.get("_untracked_baselines", []),
                "source": "curated: verified against the evidence paper's main results table",
                "test_applied": ("Strict test (approach.md): only counts if the baseline "
                                 "appears as a ROW in a main results table -- not related "
                                 "work or discussion prose."),
            }
    return None


def explain_evaluated_on(g: dict, from_slug: str, to_bench_slug: str) -> dict | None:
    for e in g["edges"]["EVALUATED_ON"]:
        if e["from_method"] == from_slug and e["to_benchmark"] == to_bench_slug:
            return {
                "found": True,
                "edge": "EVALUATED_ON",
                "evidence_paper": e.get("evidence_paper"),
                "source": "curated: the method's evidence paper reports a result on this benchmark",
                "test_applied": ("Same 'main results table' test as COMPARED_AGAINST; three "
                                 "of these edges are backed by a main-body figure rather than "
                                 "a table -- see approach.md, 'Shaping COMPARED_AGAINST'."),
            }
    return None


def explain_introduces(g: dict, from_paper: str, to_slug: str) -> dict | None:
    for e in g["edges"]["INTRODUCES"]:
        if e["from_paper"] == from_paper and e["to_method"] == to_slug:
            return {
                "found": True,
                "edge": "INTRODUCES",
                "justification": ("Definitional: this is the paper that first defined and "
                                  "named the method. schema.md enforces exactly one "
                                  "INTRODUCES edge per method."),
                "source": "curated: identity of the method-introducing paper",
                "test_applied": "Cardinality invariant: exactly one INTRODUCES edge per method.",
            }
    return None


def explain_applies(g: dict, from_paper: str, to_slug: str, review: list[dict]) -> dict | None:
    for e in g["edges"]["APPLIES"]:
        if e["from_paper"] == from_paper and e["to_method"] == to_slug:
            entry = next((r for r in review if r.get("paper_id") == from_paper
                         or r.get("arxiv_id") == from_paper), None)
            if entry is None:
                return {
                    "found": True,
                    "edge": "APPLIES",
                    "justification": "(no matching review-worksheet entry found for this paper id)",
                    "source": "tagged from abstract at scale",
                }
            return {
                "found": True,
                "edge": "APPLIES",
                "title": entry.get("title"),
                "reason": entry.get("reason"),
                "confidence": entry.get("confidence"),
                "evidence_quote": entry.get("evidence"),
                "source": "tagged from abstract; logged in data/papers_applies_review.json",
                "test_applied": ("Disjointness test (approach.md): a paper is APPLIES only "
                                 "if it runs the method's mechanism UNMODIFIED. If it "
                                 "modifies the mechanism, it fails APPLIES and becomes a "
                                 "candidate new Method instead."),
            }
    return None


# --------------------------------------------------------------------------- #
# List every edge touching a method (discovery helper)
# --------------------------------------------------------------------------- #


def list_edges_for_method(g: dict, slug: str) -> dict:
    name = {m["slug"]: m["name"] for m in g["methods"]}
    out = {et: [] for et in EDGE_TYPES}
    for e in g["edges"]["INTRODUCES"]:
        if e["to_method"] == slug:
            out["INTRODUCES"].append(f'{e["from_paper"]} -> {name[slug]}')
    for e in g["edges"]["EXTENDS"]:
        if e["from_method"] == slug:
            out["EXTENDS"].append(f'{name[slug]} -> {name[e["to_method"]]}')
        elif e["to_method"] == slug:
            out["EXTENDS"].append(f'{name[e["from_method"]]} -> {name[slug]}')
    for e in g["edges"]["COMPARED_AGAINST"]:
        if e["from_method"] == slug:
            out["COMPARED_AGAINST"].append(f'{name[slug]} -> {name[e["to_method"]]}')
    for e in g["edges"]["EVALUATED_ON"]:
        if e["from_method"] == slug:
            out["EVALUATED_ON"].append(f'{name[slug]} -> {e["to_benchmark"]}')
    for e in g["edges"]["APPLIES"]:
        if e["to_method"] == slug:
            out["APPLIES"].append(f'{e["from_paper"]} -> {name[slug]}')
    return out


# --------------------------------------------------------------------------- #
# Orchestration + output
# --------------------------------------------------------------------------- #


def explain(g: dict, review: list[dict], edge_type: str, from_token: str, to_token: str) -> dict:
    method_name, method_slug, benchmark_slug = _name_maps(g)
    edge_type = edge_type.upper()
    if edge_type not in EDGE_TYPES:
        return {"found": False, "error": f"unknown edge type '{edge_type}', "
                                          f"expected one of {EDGE_TYPES}"}

    if edge_type == "INTRODUCES":
        to_slug = _resolve_method(to_token, method_slug, method_name)
        if to_slug is None:
            return {"found": False, "error": f"unknown method '{to_token}'"}
        r = explain_introduces(g, from_token, to_slug)
    elif edge_type == "APPLIES":
        to_slug = _resolve_method(to_token, method_slug, method_name)
        if to_slug is None:
            return {"found": False, "error": f"unknown method '{to_token}'"}
        r = explain_applies(g, from_token, to_slug, review)
    elif edge_type == "EVALUATED_ON":
        from_slug = _resolve_method(from_token, method_slug, method_name)
        to_bench = _resolve_benchmark(to_token, benchmark_slug)
        if from_slug is None:
            return {"found": False, "error": f"unknown method '{from_token}'"}
        r = explain_evaluated_on(g, from_slug, to_bench or to_token)
    else:  # EXTENDS, COMPARED_AGAINST
        from_slug = _resolve_method(from_token, method_slug, method_name)
        to_slug = _resolve_method(to_token, method_slug, method_name)
        if from_slug is None or to_slug is None:
            missing = from_token if from_slug is None else to_token
            return {"found": False, "error": f"unknown method '{missing}'"}
        r = (explain_extends(g, from_slug, to_slug) if edge_type == "EXTENDS"
             else explain_compared_against(g, from_slug, to_slug))

    if r is None:
        return {"found": False,
                "error": f"no {edge_type} edge from '{from_token}' to '{to_token}' in the graph"}
    r["query"] = {"edge": edge_type, "from": from_token, "to": to_token}
    return r


def format_human(r: dict) -> str:
    L = ["=" * 68, "EXPLAIN EDGE", "=" * 68]
    if not r.get("found"):
        L.append(f"NOT FOUND: {r.get('error')}")
        L.append("=" * 68)
        return "\n".join(L)

    q = r["query"]
    L.append(f'{q["from"]}  --{q["edge"]}-->  {q["to"]}')
    for key in ("justification", "reason", "title", "evidence_paper", "evidence_quote",
                "confidence", "untracked_baselines_in_same_table"):
        if key in r and r[key] not in (None, "", []):
            label = key.replace("_", " ")
            L.append(f"  {label:<28}: {r[key]}")
    if r.get("source"):
        L.append(f"\n  source     : {r['source']}")
    if r.get("test_applied"):
        L.append(f"  test applied: {r['test_applied']}")
    L.append("=" * 68)
    return "\n".join(L)


def format_edge_list(slug_name: str, edges: dict) -> str:
    L = ["=" * 68, f"ALL EDGES TOUCHING '{slug_name}'", "=" * 68]
    for et in EDGE_TYPES:
        items = edges[et]
        L.append(f"\n{et}  ({len(items)})")
        for it in items:
            L.append(f"   {it}")
    L.append("=" * 68)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain why a specific graph edge exists.")
    ap.add_argument("--from", dest="from_", help="source node (method name/slug, or paper id for INTRODUCES/APPLIES)")
    ap.add_argument("--to", help="target node (method name/slug, or benchmark name for EVALUATED_ON)")
    ap.add_argument("--edge", choices=[e.lower() for e in EDGE_TYPES] + EDGE_TYPES,
                    help="edge type")
    ap.add_argument("--list-edges", metavar="METHOD", help="list every edge touching this method")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    g = load_graph()

    if args.list_edges:
        method_name, method_slug, _ = _name_maps(g)
        slug = _resolve_method(args.list_edges, method_slug, method_name)
        if slug is None:
            print(f"NOT FOUND: unknown method '{args.list_edges}'")
            return 1
        edges = list_edges_for_method(g, slug)
        if args.json:
            print(json.dumps(edges, indent=2, ensure_ascii=False))
        else:
            print(format_edge_list(method_name[slug], edges))
        return 0

    if not (args.from_ and args.to and args.edge):
        ap.error("provide --from, --to, and --edge (or use --list-edges METHOD)")

    review = load_applies_review()
    result = explain(g, review, args.edge, args.from_, args.to)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_human(result))
    return 0 if result.get("found") else 1


if __name__ == "__main__":
    sys.exit(main())
