"""graph_insights.py -- the graph reasoning about *itself*.

Everything else in this project reasons about a *new* input. This tool reasons
about the knowledge state that is already there: it walks the graph looking for
structural tensions -- places where the edges, taken together, say something
non-obvious that no single edge states on its own. None of these facts is written
down anywhere; each is *derived* by combining edge types (COMPARED_AGAINST +
EVALUATED_ON + family), which is exactly what a knowledge graph buys you over a
document store.

Each detector answers a question a researcher entering the field would actually
ask, and every finding is falsifiable against graph.json -- run it and check.

Detectors
---------
1. UNGROUNDED COMPARISONS
   A method M lists N as a baseline (COMPARED_AGAINST) but M and N report results
   on *disjoint* benchmark sets (EVALUATED_ON). The comparison therefore cannot be
   a head-to-head on common ground -- it is a claim made across different
   evaluation suites. Worth knowing before you cite "M beats N."

2. UNCONTESTED METHODS
   A method that *no one* uses as a baseline (never a COMPARED_AGAINST target).
   Because COMPARED_AGAINST points from a newer method back to the prior work it
   benchmarks against, an uncontested method is one nothing newer in the corpus
   has yet measured itself against -- typically the recent frontier.

3. ISOLATED METHODS
   A method that shares no benchmark with the rest of the taxonomy, so its numbers
   sit in an evaluation island: it cannot be placed on a common axis with anything
   else without new experiments. The strongest form of tension #1.

4. APPLIES COVERAGE
   How the "who runs this in practice" signal (APPLIES edges) is distributed across
   methods. This is a *disclosed* limitation, not a hidden one: the corpus is
   LoRA-heavy, so the report quantifies exactly which methods have thin or no
   application coverage. Naming the blind spot precisely is what lets the reasoning
   engine degrade gracefully instead of returning a misleading empty result.

Usage:
  py graph_insights.py            # human-readable report
  py graph_insights.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Windows consoles default stdout to the system codepage (e.g. cp1252), which
# cannot encode some characters that appear in graph prose. Force UTF-8 so
# output never raises UnicodeEncodeError when printed or redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
GRAPH = str(ROOT / "graph.json")


def load(path: str = GRAPH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _index(g: dict):
    name = {m["slug"]: m["name"] for m in g["methods"]}
    fam = {m["slug"]: m["family"] for m in g["methods"]}
    benchmarks = defaultdict(set)
    for e in g["edges"]["EVALUATED_ON"]:
        benchmarks[e["from_method"]].add(e["to_benchmark"])
    compared = []  # (from, to, evidence_paper)
    for e in g["edges"]["COMPARED_AGAINST"]:
        compared.append((e["from_method"], e["to_method"], e.get("evidence_paper")))
    return name, fam, benchmarks, compared


# --------------------------------------------------------------------------- #
# Detector 1: comparisons with no shared benchmark
# --------------------------------------------------------------------------- #


def ungrounded_comparisons(g: dict) -> list[dict]:
    name, fam, bm, compared = _index(g)
    out = []
    for f, t, paper in compared:
        shared = bm[f] & bm[t]
        if not shared:
            out.append({
                "method": name[f],
                "baseline": name[t],
                "evidence_paper": paper,
                "method_benchmarks": sorted(bm[f]),
                "baseline_benchmarks": sorted(bm[t]),
                "why": (f"{name[f]} lists {name[t]} as a results-table baseline, but "
                        f"the two report on disjoint benchmark sets -- so the tracked "
                        f"numbers are not a head-to-head on shared ground."),
            })
    return out


# --------------------------------------------------------------------------- #
# Detector 2: never a baseline (uncontested / frontier)
# --------------------------------------------------------------------------- #


def uncontested_methods(g: dict) -> list[dict]:
    name, fam, bm, compared = _index(g)
    targets = {t for _, t, _ in compared}
    out = []
    for m in g["methods"]:
        s = m["slug"]
        if s not in targets:
            out.append({
                "method": name[s],
                "family": fam[s],
                "introduced_date": m.get("introduced_date"),
                "why": (f"No tracked method benchmarks against {name[s]}. Since a "
                        f"COMPARED_AGAINST edge runs from a newer method to the prior "
                        f"work it measures itself against, nothing in the corpus has "
                        f"yet positioned itself relative to {name[s]} -- it sits at "
                        f"the recent frontier."),
            })
    out.sort(key=lambda d: d.get("introduced_date") or "")
    return out


# --------------------------------------------------------------------------- #
# Detector 3: benchmark-isolated methods
# --------------------------------------------------------------------------- #


def isolated_methods(g: dict) -> list[dict]:
    name, fam, bm, compared = _index(g)
    slugs = list(name)
    out = []
    for s in slugs:
        neighbours = [name[o] for o in slugs if o != s and (bm[s] & bm[o])]
        if not neighbours:
            out.append({
                "method": name[s],
                "family": fam[s],
                "benchmarks": sorted(bm[s]),
                "why": (f"{name[s]} shares no benchmark with any other tracked method, "
                        f"so its results live on an evaluation island -- it cannot be "
                        f"placed on a common axis with the rest of the taxonomy without "
                        f"new experiments."),
            })
    return out


# --------------------------------------------------------------------------- #
# Detector 4: APPLIES coverage (a disclosed limitation, quantified)
# --------------------------------------------------------------------------- #


def applies_coverage(g: dict) -> dict:
    name = {m["slug"]: m["name"] for m in g["methods"]}
    counts = {s: 0 for s in name}
    for e in g["edges"]["APPLIES"]:
        counts[e["to_method"]] = counts.get(e["to_method"], 0) + 1
    total = sum(counts.values())
    per_method = [{"method": name[s], "applies_papers": counts[s]}
                  for s in sorted(name, key=lambda s: -counts[s])]
    zero = [name[s] for s in name if counts[s] == 0]
    top = max(counts, key=counts.get)
    return {
        "total_applies_edges": total,
        "per_method": per_method,
        "methods_with_zero_coverage": sorted(zero),
        "most_covered": {"method": name[top], "applies_papers": counts[top]},
        "note": (f"APPLIES coverage is concentrated: {name[top]} carries "
                 f"{counts[top]}/{total} edges and {len(zero)} of {len(name)} methods "
                 f"have none. The 'who runs this in practice' signal is therefore "
                 f"reliable for LoRA-family ideas and unavailable for the rest -- a "
                 f"known, disclosed limitation the reasoning engine reports per-query "
                 f"rather than hides."),
    }


# --------------------------------------------------------------------------- #
# Orchestration + output
# --------------------------------------------------------------------------- #


def analyze(g: dict) -> dict:
    return {
        "ungrounded_comparisons": ungrounded_comparisons(g),
        "uncontested_methods": uncontested_methods(g),
        "isolated_methods": isolated_methods(g),
        "applies_coverage": applies_coverage(g),
    }


def format_human(r: dict) -> str:
    L = ["=" * 72, "GRAPH SELF-ANALYSIS  --  structural tensions derived from the edges", "=" * 72]

    uc = r["ungrounded_comparisons"]
    L.append(f"\n1. UNGROUNDED COMPARISONS  ({len(uc)})")
    L.append("   'X benchmarks against Y' but X and Y report on disjoint benchmarks.")
    for d in uc:
        L.append(f"   - {d['method']} -> {d['baseline']}")
        L.append(f"       {d['method']} on {d['method_benchmarks']}")
        L.append(f"       {d['baseline']} on {d['baseline_benchmarks']}   (no overlap)")

    un = r["uncontested_methods"]
    L.append(f"\n2. UNCONTESTED METHODS  ({len(un)})  -- nobody benchmarks against these")
    for d in un:
        L.append(f"   - {d['method']:<14} ({d['family']}, {d['introduced_date']})")

    iso = r["isolated_methods"]
    L.append(f"\n3. BENCHMARK-ISOLATED METHODS  ({len(iso)})")
    for d in iso:
        L.append(f"   - {d['method']:<14} only on {d['benchmarks']} -- shares with nothing else")

    ac = r["applies_coverage"]
    L.append(f"\n4. APPLIES COVERAGE  ({ac['total_applies_edges']} edges; disclosed limitation)")
    for d in ac["per_method"]:
        bar = "#" * d["applies_papers"]
        flag = "  <-- no coverage" if d["applies_papers"] == 0 else ""
        L.append(f"   {d['method']:<18} {d['applies_papers']:>2} {bar}{flag}")
    L.append(f"   -> {ac['note']}")

    L.append("\n" + "-" * 72)
    L.append("Every finding is derived by combining edge types (COMPARED_AGAINST +")
    L.append("EVALUATED_ON + family), not stored on any single edge. Re-run to falsify.")
    L.append("=" * 72)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Surface structural tensions in the PEFT graph.")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()
    g = load()
    r = analyze(g)
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(format_human(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
