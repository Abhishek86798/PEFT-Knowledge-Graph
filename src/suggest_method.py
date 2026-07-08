"""suggest_method.py -- position a new PEFT idea against the tracked graph.

Given a free-text description of a new PEFT method, this tool:
  1. Matches the described mechanism against the 13 tracked Method nodes and
     returns the closest match(es) with a justification.
  2. Contrasts the idea against the top match at the level of *diagnostic*
     signature terms: which mechanism terms it shares vs. which define the match
     but the idea never states (the axes on which it could actually differ), and
     places it structurally in the family tree (family, roots, root-or-variant).
  3. Detects orthogonal techniques (quantization, hypercomplex parameterization)
     that ride on `combined_techniques` rather than the mechanism taxonomy, and
     names the graph's precedent for that combination.
  4. Walks the EXTENDS graph backward from the closest match to produce a
     suggested reading order (root -> ... -> match -> [your idea]).
  5. Lists what's already been tried in that direction (EXTENDS children +
     COMPARED_AGAINST neighbours of the match).
  6. Renders a novelty verdict that fuses the lexical signal with the structural
     placement and names the concrete mechanism differentiator.

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

# Windows consoles default stdout to the system codepage (e.g. cp1252), which
# cannot encode some characters that appear in mechanism prose. Force UTF-8 so
# output never raises UnicodeEncodeError when printed or redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    family_of: dict[str, str] = field(default_factory=dict)           # method -> family
    is_root: dict[str, bool] = field(default_factory=dict)            # method -> is_family_root
    combined: dict[str, list[str]] = field(default_factory=dict)      # method -> combined_techniques
    applies_count: dict[str, int] = field(default_factory=dict)       # method -> # APPLIES papers


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
    for s, m in methods.items():
        gr.extends_out[s] = []
        gr.extends_in[s] = []
        gr.compared[s] = []
        gr.family_of[s] = m.get("family", "")
        gr.is_root[s] = bool(m.get("is_family_root", False))
        gr.combined[s] = list(m.get("combined_techniques", []))
        gr.applies_count[s] = 0
    for e in g["edges"]["EXTENDS"]:
        gr.extends_out[e["from_method"]].append(e["to_method"])
        gr.extends_in[e["to_method"]].append(e["from_method"])
    for e in g["edges"]["COMPARED_AGAINST"]:
        gr.compared[e["from_method"]].append(e["to_method"])
    for e in g["edges"]["APPLIES"]:
        gr.applies_count[e["to_method"]] = gr.applies_count.get(e["to_method"], 0) + 1
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
    COMPARED_AGAINST baselines (what it was benchmarked against).

    Also reports APPLIES coverage for the matched method and states the known
    limitation honestly: the "who runs this in practice" signal is real only where
    the corpus has application papers, and the corpus is LoRA-heavy. When the match
    has zero APPLIES papers, the tool says so explicitly rather than returning an
    empty list that reads like a bug -- graceful degradation over silent absence.
    """
    children = [gr.name_of[s] for s in gr.extends_in.get(slug, [])]
    # siblings: other variants that share a parent with the match
    siblings = []
    for parent in gr.extends_out.get(slug, []):
        for sib in gr.extends_in.get(parent, []):
            if sib != slug:
                siblings.append(gr.name_of[sib])
    baselines = [gr.name_of[s] for s in gr.compared.get(slug, [])]
    n_applies = gr.applies_count.get(slug, 0)
    if n_applies == 0:
        coverage_note = (
            f"No APPLIES papers for '{gr.name_of[slug]}' in this corpus, so the "
            f"'who runs this in practice' signal is UNAVAILABLE for this match "
            f"(a known coverage limitation: the APPLIES tier is LoRA-heavy). The "
            f"EXTENDS/COMPARED_AGAINST reasoning above is unaffected -- those edges "
            f"are curated from the papers, not abstract-tagged.")
    else:
        coverage_note = (f"{n_applies} APPLIES paper(s) run '{gr.name_of[slug]}' "
                         f"unmodified in this corpus.")
    return {
        "extends_children": sorted(set(children)),
        "sibling_variants": sorted(set(siblings)),
        "compared_against_baselines": sorted(set(baselines)),
        "applies_papers_for_match": n_applies,
        "applies_coverage_note": coverage_note,
    }


# --------------------------------------------------------------------------- #
# 3b. Mechanism contrast -- WHY the idea is / isn't the match, at the level of
#     diagnostic signature terms. This is the step that turns "verify it isn't
#     X re-described" into a concrete, mechanism-level differentiator: it names
#     the diagnostic terms the idea shares with the match versus the ones that
#     define the match but the idea never mentions.
# --------------------------------------------------------------------------- #


def mechanism_contrast(query_tokens: set[str], slug: str) -> dict:
    """Split the match's signature vocabulary into shared vs. unmatched.

    `shared`     -- diagnostic terms the idea and the match both use (real overlap)
    `unmatched`  -- diagnostic terms that DEFINE the match but the idea never
                    mentions. These are the mechanism commitments the idea would
                    have to make to actually be this method -- i.e. if the idea
                    does NOT intend them, that's precisely where it differs.
    """
    sig_terms = MECHANISMS[slug]["signature"]
    shared, unmatched = [], []
    for term in sig_terms:
        term_toks = _token_set(term)
        if term_toks & query_tokens:
            shared.append(term)
        else:
            unmatched.append(term)
    return {"shared": shared, "unmatched": unmatched}


# --------------------------------------------------------------------------- #
# 3c. Family placement -- structural, not lexical. Where does the idea land in
#     the taxonomy, and does the match it lands on reduce to a family root?
# --------------------------------------------------------------------------- #


def family_placement(slug: str, gr: Graph) -> dict:
    """Report the family the idea lands in and its root, using the EXTENDS DAG.

    This is the structural half of the novelty judgment: an idea that lands on a
    variant sits inside an established family (its mechanism reduces to the root
    under the schema's reduction test); an idea that lands on a root is either
    that root or a candidate new root beside it.
    """
    family = gr.family_of.get(slug, "")
    roots = [gr.name_of[s] for s in gr.methods
             if gr.family_of.get(s) == family and gr.is_root.get(s)]
    return {
        "family": family,
        "family_roots": sorted(roots),
        "match_is_root": gr.is_root.get(slug, False),
    }


# --------------------------------------------------------------------------- #
# 3d. Orthogonal technique -- schema decision: quantization / hypercomplex-
#     parameterization ride on `combined_techniques`, NOT on the mechanism
#     taxonomy (see approach.md, "the edge I demoted"). An idea that names one
#     of these is combining an orthogonal axis with a base mechanism, and the
#     graph already records the precedent for that combination.
# --------------------------------------------------------------------------- #

# Surface vocabulary -> the combined_techniques tag the graph stores.
_TECHNIQUE_CUES: dict[str, list[str]] = {
    "quantization": ["quantiz", "4-bit", "8-bit", "int8", "nf4", "bnb", "bitsandbytes"],
    "hypercomplex-parameterization": ["hypercomplex", "kronecker", "phm", "quaternion"],
}


def orthogonal_techniques(query: str, gr: Graph) -> list[dict]:
    """Detect orthogonal techniques in the idea and name the graph's precedent."""
    q = query.lower()
    out = []
    for tag, cues in _TECHNIQUE_CUES.items():
        if any(cue in q for cue in cues):
            precedents = [gr.name_of[s] for s in gr.methods if tag in gr.combined.get(s, [])]
            out.append({"technique": tag, "precedents": sorted(precedents)})
    return out


# --------------------------------------------------------------------------- #
# 4. Near-identical flag
# --------------------------------------------------------------------------- #


def _differentiator(match_name: str, contrast: dict) -> str:
    """A concrete, mechanism-level sentence about where the idea could differ.

    Built from the match's *unmatched* signature terms -- the diagnostic
    commitments that define the match but the idea never states. If the idea
    does not intend these, that is precisely what separates it from the match.
    """
    unmatched = contrast.get("unmatched", [])
    if not unmatched:
        return (f"The idea already names every diagnostic term of '{match_name}', "
                f"so there is no mechanism axis left on which it visibly differs.")
    shown = ", ".join(unmatched[:4])
    return (f"To be more than '{match_name}' re-described, the idea must differ on "
            f"its defining mechanism -- '{match_name}' is characterised by "
            f"[{shown}], which your description does not mention. State how you "
            f"depart from those (or confirm you inherit them) to settle novelty.")


def novelty_flag(matches: list[dict], contrast: dict, placement: dict) -> dict:
    """Classify novelty using the score, signature-term hits, AND structure.

    A high score driven only by generic prose words (signature_hits == 0) is not
    real mechanism overlap, so it must not read as "already exists" -- that is
    the failure mode of pure lexical matching, and this guard is what keeps the
    flag honest. Beyond the lexical signal, the verdict now names the concrete
    mechanism axis on which the idea could differ (the match's unmatched
    diagnostic terms) and locates the idea structurally in the taxonomy.
    """
    top = matches[0]
    strong_signal = top["signature_hits"] >= 2
    diff = _differentiator(top["name"], contrast)
    fam = placement["family"]
    where = (f"Structurally it lands in the '{fam}' family "
             f"(roots: {', '.join(placement['family_roots'])}); the match is "
             f"{'a root' if placement['match_is_root'] else 'a variant that reduces to that root'}.")

    if top["score"] >= _NEAR_IDENTICAL and strong_signal:
        return {
            "status": "LIKELY_DUPLICATE",
            "message": f"Description is very close to '{top['name']}' "
                       f"(score {top['score']}, {top['signature_hits']} signature terms). "
                       f"Verify it is not just {top['name']} re-described before "
                       f"treating it as novel. {diff}",
            "differentiator": diff,
            "structural_placement": where,
        }
    if top["score"] >= _NEAR_IDENTICAL * 0.6 and strong_signal:
        return {
            "status": "OVERLAPS_EXISTING",
            "message": f"Strong mechanism overlap with '{top['name']}' "
                       f"({top['signature_hits']} signature terms). Likely a variant in "
                       f"that family; position it via the reading order and check the "
                       f"already-tried list before claiming novelty. {diff}",
            "differentiator": diff,
            "structural_placement": where,
        }
    if top["signature_hits"] <= 1:
        return {
            "status": "APPEARS_DISTINCT",
            "message": f"Closest lexical match is '{top['name']}' but it shares only "
                       f"generic wording ({top['signature_hits']} diagnostic term(s)), "
                       f"not core mechanism. The idea may sit outside the current "
                       f"taxonomy (a candidate new family root). NOTE: matching is "
                       f"lexical -- confirm by reading '{top['name']}' before concluding.",
            "differentiator": diff,
            "structural_placement": where,
        }
    return {
        "status": "WEAK_OVERLAP",
        "message": f"Partial overlap with '{top['name']}'. Not a clear duplicate; "
                   f"review the closest matches and the already-tried list. {diff}",
        "differentiator": diff,
        "structural_placement": where,
    }


# --------------------------------------------------------------------------- #
# Orchestration + output
# --------------------------------------------------------------------------- #


def analyze(query: str, gr: Graph, top_k: int = 3) -> dict:
    qt = _token_set(query)
    matches = match(query, gr, top_k=top_k)
    top = matches[0]
    contrast = mechanism_contrast(qt, top["slug"])
    placement = family_placement(top["slug"], gr)
    return {
        "query": query,
        "closest_matches": matches,
        "mechanism_contrast": contrast,
        "family_placement": placement,
        "orthogonal_techniques": orthogonal_techniques(query, gr),
        "suggested_reading_order": reading_order(top["slug"], gr) + ["<your idea>"],
        "already_tried_in_this_direction": already_tried(top["slug"], gr),
        "novelty": novelty_flag(matches, contrast, placement),
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

    top_name = result["closest_matches"][0]["name"]
    mc = result["mechanism_contrast"]
    lines.append(f"\n2. MECHANISM CONTRAST vs {top_name}")
    lines.append(f"   shares diagnostic mechanism : {', '.join(mc['shared']) or '(none)'}")
    lines.append(f"   {top_name}-defining terms NOT in your idea: "
                 f"{', '.join(mc['unmatched']) or '(none)'}")

    fp = result["family_placement"]
    lines.append(f"   family placement            : '{fp['family']}' "
                 f"(roots: {', '.join(fp['family_roots'])}); match is "
                 f"{'a ROOT' if fp['match_is_root'] else 'a VARIANT (reduces to root)'}")

    ot = result["orthogonal_techniques"]
    if ot:
        lines.append("\n3. ORTHOGONAL TECHNIQUE DETECTED (rides on combined_techniques, not the taxonomy)")
        for t in ot:
            prec = ', '.join(t["precedents"]) or "(no tracked precedent)"
            lines.append(f"   '{t['technique']}' -> already combined with a base mechanism by: {prec}")

    lines.append("\n4. SUGGESTED READING ORDER (root -> match -> your idea)")
    lines.append("   " + "  ->  ".join(result["suggested_reading_order"]))

    lines.append("\n5. ALREADY TRIED IN THIS DIRECTION")
    at = result["already_tried_in_this_direction"]
    lines.append(f"   variants extending the match : {', '.join(at['extends_children']) or '(none)'}")
    lines.append(f"   sibling variants (same parent): {', '.join(at['sibling_variants']) or '(none)'}")
    lines.append(f"   its results-table baselines  : {', '.join(at['compared_against_baselines']) or '(none)'}")
    lines.append(f"   applied in practice by       : {at['applies_papers_for_match']} paper(s) "
                 f"(APPLIES edges)")
    lines.append(f"   ! {at['applies_coverage_note']}")

    lines.append("\n6. NOVELTY VERDICT")
    nv = result["novelty"]
    lines.append(f"   [{nv['status']}]")
    lines.append(f"   {nv['structural_placement']}")
    lines.append(f"   -> {nv['differentiator']}")
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
