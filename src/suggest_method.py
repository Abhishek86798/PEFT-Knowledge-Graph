"""suggest_method.py -- position a new PEFT idea against the tracked graph.

Given a free-text description of a new PEFT method, this tool:
  1. Matches the described mechanism against the 13 tracked Method nodes and
     returns the closest match(es) with a justification.
  2. Walks the EXTENDS graph backward from the closest match to produce a
     suggested reading order (root -> ... -> match -> [your idea]).
  3. Lists what's already been tried in that direction (EXTENDS children +
     COMPARED_AGAINST neighbours of the match).
  4. Flags whether the idea looks near-identical to an existing method.

Matching is deliberately lexical (weighted token/signature overlap), not a
neural embedding: it is deterministic, dependency-free, and needs no API. The
mechanism knowledge base below encodes each method's schema §1.1 one-sentence
"which parameters are trained and how they are injected" statement plus a set of
signature terms that are diagnostic of that mechanism.

Usage:
  py suggest_method.py "your PEFT idea in free text ..."
  py suggest_method.py --file idea.txt
  py suggest_method.py "..." --json      # machine-readable output only
  echo "..." | py suggest_method.py      # read from stdin
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = parent of src/, so graph.json resolves from any working directory.
ROOT = Path(__file__).resolve().parent.parent
GRAPH = str(ROOT / "graph.json")

# --------------------------------------------------------------------------- #
# Mechanism knowledge base -- schema §1.1 descriptions + signature terms.
# `signature` terms are weighted higher than prose tokens: they are the terms
# that actually distinguish one mechanism from another (e.g. "bias" for BitFit,
# "kronecker" for Compacter). Keep these curated, not scraped.
# --------------------------------------------------------------------------- #

MECHANISMS: dict[str, dict] = {
    "method_adapters": {
        "mechanism": "Inserts small trainable bottleneck modules (down-project, "
                     "nonlinearity, up-project) between frozen transformer sub-layers.",
        "signature": ["adapter", "bottleneck", "module", "insert", "down-project",
                      "up-project", "serial", "residual"],
    },
    "method_pfeiffer_adapters": {
        "mechanism": "Adapter modules placed at a single (post-FFN) position per layer "
                     "and composed across tasks via a fusion layer.",
        "signature": ["adapter", "fusion", "single", "placement", "composition",
                      "multi-task", "post-ffn"],
    },
    "method_compacter": {
        "mechanism": "Adapters whose weight matrices are Kronecker products of shared "
                     "low-rank / hypercomplex factors, cutting adapter parameters.",
        "signature": ["kronecker", "hypercomplex", "phm", "shared", "low-rank",
                      "adapter", "parameterized", "factor"],
    },
    "method_prefix_tuning": {
        "mechanism": "Prepends trainable continuous key/value prefix vectors to the "
                     "attention of every transformer layer; the model stays frozen.",
        "signature": ["prefix", "key", "value", "attention", "every layer", "deep",
                      "continuous", "prepend", "virtual token"],
    },
    "method_prompt_tuning": {
        "mechanism": "Prepends trainable soft-prompt embeddings at the input layer only; "
                     "a strict input-layer special case of prefix tuning.",
        "signature": ["prompt", "soft prompt", "input", "embedding", "prepend",
                      "continuous", "input-layer", "virtual token"],
    },
    "method_ptuning": {
        "mechanism": "Continuous prompt embeddings produced by a small prompt encoder "
                     "(LSTM/MLP) inserted into the input; targets NLU.",
        "signature": ["prompt", "encoder", "lstm", "mlp", "continuous", "input",
                      "reparameterize", "nlu"],
    },
    "method_ptuning_v2": {
        "mechanism": "Deep continuous prompts applied at every layer (prefix-style) for "
                     "NLU across scales; merges P-Tuning framing with deep prefixes.",
        "signature": ["prompt", "deep", "every layer", "prefix", "nlu", "scales",
                      "continuous"],
    },
    "method_lora": {
        "mechanism": "Adds a trainable low-rank update B*A to frozen weight matrices "
                     "(W + BA); mergeable at inference, no extra latency.",
        # Signature terms must be DIAGNOSTIC of low-rank reparameterization -- not
        # generic words like "weight"/"update"/"delta" that any weight-space method
        # shares (those live in the prose, scored at prose weight, not here).
        "signature": ["low-rank", "lora", "rank", "decomposition", "mergeable",
                      "reparameterization"],
    },
    "method_adalora": {
        "mechanism": "LoRA with adaptive rank: allocates the rank budget across weight "
                     "matrices via SVD-based importance pruning of singular values.",
        "signature": ["adaptive", "rank", "budget", "svd", "singular", "importance",
                      "prune", "allocation", "low-rank"],
    },
    "method_qlora": {
        "mechanism": "LoRA adapters trained on top of a 4-bit NF4-quantized frozen base "
                     "with double quantization and paged optimizers.",
        "signature": ["quantiz", "4-bit", "nf4", "int8", "memory", "double quantization",
                      "paged", "low-rank", "lora"],
    },
    "method_vera": {
        "mechanism": "Freezes a pair of shared random low-rank matrices and trains only "
                     "small per-layer scaling vectors on top of them.",
        "signature": ["random", "shared", "frozen", "scaling vector", "low-rank",
                      "vector", "reuse", "tiny"],
    },
    "method_bitfit": {
        "mechanism": "Trains only the existing bias terms of the model; injects no new "
                     "parameters at all.",
        "signature": ["bias", "bias-only", "selective", "subset", "existing parameters",
                      "no new parameters", "sparse"],
    },
    "method_ia3": {
        "mechanism": "Learns element-wise rescaling vectors that multiply the keys, "
                     "values, and FFN activations; multiplicative, not additive.",
        "signature": ["rescal", "multiplicative", "element-wise", "scaling", "activation",
                      "key", "value", "ffn", "vector", "gate"],
    },
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "that",
    "is", "are", "be", "by", "as", "it", "we", "our", "this", "these", "at", "into",
    "only", "new", "method", "model", "models", "trainable", "train", "training",
    "parameter", "parameters", "efficient", "fine-tuning", "finetuning", "peft",
    "approach", "which", "each", "small", "large", "language",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #


@dataclass
class Graph:
    methods: dict[str, dict]
    name_of: dict[str, str]
    extends_out: dict[str, list[str]] = field(default_factory=dict)   # variant -> parents
    extends_in: dict[str, list[str]] = field(default_factory=dict)    # parent -> variants
    compared: dict[str, list[str]] = field(default_factory=dict)      # method -> baselines


def load_graph(path: str = GRAPH) -> Graph:
    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)
    methods = {m["slug"]: m for m in g["methods"]}
    name_of = {s: m["name"] for s, m in methods.items()}

    # Single source of truth: if the graph carries mechanism/signature_terms on
    # its Method nodes (the inspectable knowledge state), use those and let the
    # in-script MECHANISMS serve only as a fallback for older graphs.
    for slug, m in methods.items():
        if m.get("mechanism"):
            MECHANISMS.setdefault(slug, {})
            MECHANISMS[slug]["mechanism"] = m["mechanism"]
            if m.get("signature_terms"):
                MECHANISMS[slug]["signature"] = m["signature_terms"]
    gr = Graph(methods=methods, name_of=name_of)
    for s in methods:
        gr.extends_out[s] = []
        gr.extends_in[s] = []
        gr.compared[s] = []
    for e in g["edges"]["EXTENDS"]:
        gr.extends_out[e["from_method"]].append(e["to_method"])
        gr.extends_in[e["to_method"]].append(e["from_method"])
    for e in g["edges"]["COMPARED_AGAINST"]:
        gr.compared[e["from_method"]].append(e["to_method"])
    return gr


# --------------------------------------------------------------------------- #
# 1. Matching
# --------------------------------------------------------------------------- #

_SIG_WEIGHT = 3.0      # a signature-term hit is worth this many prose-token hits
_PROSE_WEIGHT = 1.0
_NEAR_IDENTICAL = 0.55  # normalized score above which we flag "already exists"


def score_method(query_tokens: set[str], slug: str) -> tuple[float, list[str], int]:
    """Return (normalized_score, matched_terms, signature_hit_count).

    signature_hit_count separates real mechanism overlap (diagnostic signature
    terms) from incidental prose-word overlap ("weight", "update"), so the
    novelty flag can tell "this is basically LoRA" from "shares generic words".
    """
    kb = MECHANISMS[slug]
    mech_tokens = _token_set(kb["mechanism"])
    sig_tokens = {t for term in kb["signature"] for t in _tokens(term)}

    matched = []
    raw = 0.0
    sig_hits = 0
    for t in query_tokens:
        if t in sig_tokens:
            raw += _SIG_WEIGHT
            matched.append(t)
            sig_hits += 1
        elif t in mech_tokens:
            raw += _PROSE_WEIGHT
            matched.append(t)

    # Normalize by the achievable signal of this method (so short-signature
    # methods like BitFit aren't unfairly out-scored by verbose ones).
    denom = _SIG_WEIGHT * len(sig_tokens) * 0.5 + _PROSE_WEIGHT * len(mech_tokens) * 0.5
    norm = raw / denom if denom else 0.0
    return norm, sorted(set(matched)), sig_hits


def match(query: str, gr: Graph, top_k: int = 3) -> list[dict]:
    qt = _token_set(query)
    scored = []
    for slug in MECHANISMS:
        norm, matched, sig_hits = score_method(qt, slug)
        scored.append({
            "slug": slug,
            "name": gr.name_of[slug],
            "score": round(norm, 3),
            "signature_hits": sig_hits,
            "matched_terms": matched,
            "mechanism": MECHANISMS[slug]["mechanism"],
        })
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


# --------------------------------------------------------------------------- #
# 2. Reading order (walk EXTENDS backward: root -> ... -> match)
# --------------------------------------------------------------------------- #


def reading_order(slug: str, gr: Graph) -> list[str]:
    """Walk EXTENDS parents back to a root, returning root-first name order.

    Multiple parents (P-Tuning v2) are all included; the path is de-duplicated
    while preserving a root-first reading direction.
    """
    order: list[str] = []
    seen: set[str] = set()

    def visit(s: str) -> None:
        if s in seen:
            return
        seen.add(s)
        for parent in gr.extends_out.get(s, []):
            visit(parent)
        order.append(s)  # appended after parents -> parents come first

    visit(slug)
    return [gr.name_of[s] for s in order]


# --------------------------------------------------------------------------- #
# 3. What's already been tried in this direction
# --------------------------------------------------------------------------- #


def already_tried(slug: str, gr: Graph) -> dict:
    """EXTENDS children (methods that already build on the match) + the match's
    COMPARED_AGAINST baselines (what it was benchmarked against)."""
    children = [gr.name_of[s] for s in gr.extends_in.get(slug, [])]
    # siblings: other variants that share a parent with the match
    siblings = []
    for parent in gr.extends_out.get(slug, []):
        for sib in gr.extends_in.get(parent, []):
            if sib != slug:
                siblings.append(gr.name_of[sib])
    baselines = [gr.name_of[s] for s in gr.compared.get(slug, [])]
    return {
        "extends_children": sorted(set(children)),
        "sibling_variants": sorted(set(siblings)),
        "compared_against_baselines": sorted(set(baselines)),
    }


# --------------------------------------------------------------------------- #
# 4. Near-identical flag
# --------------------------------------------------------------------------- #


def novelty_flag(matches: list[dict]) -> dict:
    """Classify novelty using BOTH the score and signature-term hits.

    A high score driven only by generic prose words (signature_hits == 0) is not
    real mechanism overlap, so it must not read as "already exists" -- that is
    the failure mode of pure lexical matching, and this guard is what keeps the
    flag honest.
    """
    top = matches[0]
    strong_signal = top["signature_hits"] >= 2

    if top["score"] >= _NEAR_IDENTICAL and strong_signal:
        return {
            "status": "LIKELY_DUPLICATE",
            "message": f"Description is very close to '{top['name']}' "
                       f"(score {top['score']}, {top['signature_hits']} signature terms). "
                       f"Verify it is not just {top['name']} re-described before "
                       f"treating it as novel.",
        }
    if top["score"] >= _NEAR_IDENTICAL * 0.6 and strong_signal:
        return {
            "status": "OVERLAPS_EXISTING",
            "message": f"Strong mechanism overlap with '{top['name']}' "
                       f"({top['signature_hits']} signature terms). Likely a variant in "
                       f"that family; position it via the reading order and check the "
                       f"already-tried list before claiming novelty.",
        }
    if top["signature_hits"] <= 1:
        return {
            "status": "APPEARS_DISTINCT",
            "message": f"Closest lexical match is '{top['name']}' but it shares only "
                       f"generic wording ({top['signature_hits']} diagnostic term(s)), "
                       f"not core mechanism. The idea may sit outside the current "
                       f"taxonomy (a candidate new family root). NOTE: matching is "
                       f"lexical -- confirm by reading '{top['name']}' before concluding.",
        }
    return {
        "status": "WEAK_OVERLAP",
        "message": f"Partial overlap with '{top['name']}'. Not a clear duplicate; "
                   f"review the closest matches and the already-tried list.",
    }


# --------------------------------------------------------------------------- #
# Orchestration + output
# --------------------------------------------------------------------------- #


def analyze(query: str, gr: Graph, top_k: int = 3) -> dict:
    matches = match(query, gr, top_k=top_k)
    top = matches[0]
    return {
        "query": query,
        "closest_matches": matches,
        "suggested_reading_order": reading_order(top["slug"], gr) + ["<your idea>"],
        "already_tried_in_this_direction": already_tried(top["slug"], gr),
        "novelty": novelty_flag(matches),
    }


def format_human(result: dict) -> str:
    lines = []
    q = result["query"]
    lines.append("=" * 68)
    lines.append("PEFT IDEA POSITIONING")
    lines.append("=" * 68)
    lines.append(f'Query: "{q[:200]}"')

    lines.append("\n1. CLOSEST TRACKED METHODS")
    for i, m in enumerate(result["closest_matches"], 1):
        bar = "#" * int(m["score"] * 20)
        lines.append(f"   {i}. {m['name']:<18} score={m['score']:<5} sig={m['signature_hits']} {bar}")
        lines.append(f"      mechanism: {m['mechanism']}")
        if m["matched_terms"]:
            lines.append(f"      matched on: {', '.join(m['matched_terms'])}")

    lines.append("\n2. SUGGESTED READING ORDER (root -> match -> your idea)")
    lines.append("   " + "  ->  ".join(result["suggested_reading_order"]))

    lines.append("\n3. ALREADY TRIED IN THIS DIRECTION")
    at = result["already_tried_in_this_direction"]
    lines.append(f"   variants extending the match : {', '.join(at['extends_children']) or '(none)'}")
    lines.append(f"   sibling variants (same parent): {', '.join(at['sibling_variants']) or '(none)'}")
    lines.append(f"   its results-table baselines  : {', '.join(at['compared_against_baselines']) or '(none)'}")

    lines.append("\n4. NOVELTY FLAG")
    nv = result["novelty"]
    lines.append(f"   [{nv['status']}] {nv['message']}")
    lines.append("=" * 68)
    return "\n".join(lines)


def _read_query(args) -> str:
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            return fh.read().strip()
    if args.text:
        return " ".join(args.text).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Position a new PEFT idea against the graph.")
    ap.add_argument("text", nargs="*", help="free-text description of the idea")
    ap.add_argument("--file", help="read the description from a file")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--top-k", type=int, default=3, help="how many matches to return")
    args = ap.parse_args()

    query = _read_query(args)
    if not query:
        ap.error("no description provided (pass text, --file, or pipe via stdin)")

    gr = load_graph()
    result = analyze(query, gr, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_human(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
