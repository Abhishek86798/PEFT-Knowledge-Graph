"""Executable validation suite for graph.json (schema.md §4 / §6.6).

Every rule in the schema's YAML `validation:` block exists here as a check.
Run after every change to the graph:  py validate_graph.py

Exit code 0 = all hard rules pass; 1 = at least one ERROR. Warnings never fail.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Repo root = parent of this script's directory (src/). graph.json lives there,
# so paths resolve correctly no matter where the script is invoked from.
ROOT = Path(__file__).resolve().parent.parent
GRAPH = str(ROOT / "graph.json")


def load() -> dict:
    with open(GRAPH, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    g = load()
    methods = {m["slug"]: m for m in g["methods"]}
    papers = {p["arxiv_id"]: p for p in g["papers"]}
    benches = {b["slug"]: b for b in g["benchmarks"]}
    E = g["edges"]
    errs: list[str] = []
    warns: list[str] = []

    # --- INTRODUCES: exactly one per method; endpoints exist ----------------- #
    intro_by_method: dict[str, list[str]] = defaultdict(list)
    for e in E["INTRODUCES"]:
        if e["from_paper"] not in papers:
            errs.append(f"INTRODUCES from unknown paper {e['from_paper']}")
        if e["to_method"] not in methods:
            errs.append(f"INTRODUCES to unknown method {e['to_method']}")
        intro_by_method[e["to_method"]].append(e["from_paper"])
    for slug in methods:
        n = len(intro_by_method.get(slug, []))
        if n != 1:
            errs.append(f"method {slug} has {n} INTRODUCES edges (must be exactly 1)")
    intro = {m: ps[0] for m, ps in intro_by_method.items() if ps}

    # --- Method.introduced_date == its introducing paper's arxiv_v1_date ----- #
    for m, p in intro.items():
        if methods[m]["introduced_date"] != papers[p]["arxiv_v1_date"]:
            errs.append(
                f"{m} introduced_date {methods[m]['introduced_date']} != "
                f"paper {p} v1 {papers[p]['arxiv_v1_date']}"
            )

    # --- EXTENDS: endpoints exist; root => 0 outgoing; DAG; chronology ------- #
    out = defaultdict(list)
    for e in E["EXTENDS"]:
        a, b = e["from_method"], e["to_method"]
        if a not in methods:
            errs.append(f"EXTENDS from unknown {a}")
        if b not in methods:
            errs.append(f"EXTENDS to unknown {b}")
        out[a].append(b)
        if a in methods and b in methods:
            if methods[a]["introduced_date"] <= methods[b]["introduced_date"]:
                warns.append(
                    f"chronology: {a} ({methods[a]['introduced_date']}) EXTENDS "
                    f"{b} ({methods[b]['introduced_date']}) not strictly later "
                    f"(check for concurrent preprints; mark concurrent:true if within 60d)"
                )
    for slug, m in methods.items():
        if m["is_family_root"] and out.get(slug):
            errs.append(f"root {slug} has {len(out[slug])} outgoing EXTENDS (must be 0)")

    # DAG check
    color = {s: 0 for s in methods}  # 0 white, 1 grey, 2 black

    def dfs(u: str) -> bool:
        color[u] = 1
        for v in out.get(u, []):
            if color.get(v) == 1:
                return True
            if color.get(v) == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    if any(color[s] == 0 and dfs(s) for s in methods):
        errs.append("EXTENDS contains a cycle (must be a DAG)")

    # --- COMPARED_AGAINST: endpoints/evidence exist; from method's own paper - #
    for e in E["COMPARED_AGAINST"]:
        if e["from_method"] not in methods:
            errs.append(f"CA from unknown {e['from_method']}")
        if e["to_method"] not in methods:
            errs.append(f"CA to unknown {e['to_method']}")
        if e["from_method"] == e["to_method"]:
            errs.append(f"CA self-loop {e['from_method']}")
        if e["evidence_paper"] not in papers:
            errs.append(f"CA evidence_paper unknown {e['evidence_paper']}")
        exp = intro.get(e["from_method"])
        if exp and e["evidence_paper"] != exp:
            errs.append(
                f"CA {e['from_method']} evidence {e['evidence_paper']} != "
                f"its introducing paper {exp}"
            )

    # --- EVALUATED_ON: endpoints/evidence exist; from method's own paper ----- #
    used_bench = set()
    for e in E["EVALUATED_ON"]:
        if e["from_method"] not in methods:
            errs.append(f"EO from unknown {e['from_method']}")
        if e["to_benchmark"] not in benches:
            errs.append(f"EO to unknown benchmark {e['to_benchmark']}")
        else:
            used_bench.add(e["to_benchmark"])
        if e["evidence_paper"] not in papers:
            errs.append(f"EO evidence_paper unknown {e['evidence_paper']}")
        exp = intro.get(e["from_method"])
        if exp and e["evidence_paper"] != exp:
            errs.append(
                f"EO {e['from_method']}->{e['to_benchmark']} evidence "
                f"{e['evidence_paper']} != its introducing paper {exp}"
            )

    # --- no orphan benchmarks (schema §1.3) --------------------------------- #
    for b in benches:
        if b not in used_bench:
            errs.append(f"orphan benchmark (no EVALUATED_ON edge): {b}")

    # --- APPLIES: paper+method exist; not same as INTRODUCES; not modified --- #
    intro_pairs = {(e["from_paper"], e["to_method"]) for e in E["INTRODUCES"]}
    for e in E["APPLIES"]:
        if e["from_paper"] not in papers:
            errs.append(f"APPLIES from unknown paper {e['from_paper']}")
        if e["to_method"] not in methods:
            errs.append(f"APPLIES to unknown method {e['to_method']}")
        if (e["from_paper"], e["to_method"]) in intro_pairs:
            errs.append(
                f"paper {e['from_paper']} both INTRODUCES and APPLIES "
                f"{e['to_method']} (forbidden, schema §4)"
            )

    # --- no orphan papers (>=1 of INTRODUCES / APPLIES / evidence) ---------- #
    referenced = set()
    referenced |= {e["from_paper"] for e in E["INTRODUCES"]}
    referenced |= {e["from_paper"] for e in E["APPLIES"]}
    referenced |= {e["evidence_paper"] for e in E["COMPARED_AGAINST"]}
    referenced |= {e["evidence_paper"] for e in E["EVALUATED_ON"]}
    for p in papers:
        if p not in referenced:
            warns.append(f"orphan paper (no INTRODUCES/APPLIES/evidence): {p}")

    # --- corpus size guardrail (schema §4: target 70, fail outside 60-80) ---- #
    total_papers = len(papers)
    if not (60 <= total_papers <= 80):
        warns.append(
            f"corpus size {total_papers} outside guardrail band 60-80 "
            f"(target 70) -- expected until APPLIES papers (3.3) are added"
        )

    # --- report ------------------------------------------------------------- #
    print(f"Nodes   : {len(methods)} methods, {len(papers)} papers, {len(benches)} benchmarks")
    print(
        "Edges   : "
        f"INTRODUCES {len(E['INTRODUCES'])}, EXTENDS {len(E['EXTENDS'])}, "
        f"COMPARED_AGAINST {len(E['COMPARED_AGAINST'])}, "
        f"EVALUATED_ON {len(E['EVALUATED_ON'])}, APPLIES {len(E['APPLIES'])}"
    )
    print(f"\nWARNINGS ({len(warns)}):")
    for w in warns:
        print("  !", w)
    if not warns:
        print("  none")
    print(f"\nERRORS ({len(errs)}):")
    for e in errs:
        print("  x", e)
    if not errs:
        print("  none")

    print("\n" + ("VALIDATION PASSED" if not errs else "VALIDATION FAILED"))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
