"""Fetch research paper data from the Semantic Scholar API for a PEFT knowledge graph.

Pipeline:
  1. Start from a hardcoded list of seed paper titles (PEFT-related).
  2. Resolve each title -> paperId via the paper search endpoint.
  3. Fetch full details for each resolved paperId.
  4. Fetch citations and references for each seed paper.
  5. Build a deduplicated candidate pool of all citing/cited papers.

The unauthenticated Semantic Scholar API is aggressively rate-limited (shared
~1 req/sec pool, frequent HTTP 429). Set the S2_API_KEY environment variable to
use an API key and raise your limits. All requests retry with exponential
backoff on transient failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

# Repo root = parent of src/. The candidate corpus is written to data/.
ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

API_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_URL = f"{API_BASE}/paper/search"
PAPER_URL = f"{API_BASE}/paper/{{paper_id}}"
BATCH_URL = f"{API_BASE}/paper/batch"  # POST {"ids":[...]} -- up to 500 ids/call
CITATIONS_URL = f"{API_BASE}/paper/{{paper_id}}/citations"
REFERENCES_URL = f"{API_BASE}/paper/{{paper_id}}/references"

# Fields requested for full paper detail (seeds).
DETAIL_FIELDS = "title,abstract,year,authors,venue,externalIds,citationCount"
# Fields requested for candidate detail via the batch endpoint (scoring needs
# title+abstract; the rest populate the output record). No externalIds/citations
# needed for scoring, but externalIds is cheap and lets us key on arXiv id later.
CANDIDATE_FIELDS = "title,abstract,year,authors,venue,externalIds"
# Fields requested for each linked (citing/cited) paper.
LINK_FIELDS = "title,externalIds"

# Number of linked papers to pull per page (API max is 1000).
LINK_PAGE_SIZE = 1000

# Batch endpoint hard limit: 500 ids per POST.
BATCH_MAX_IDS = 500

# Title-match threshold (Jaccard token overlap) below which we warn.
TITLE_MATCH_THRESHOLD = 0.6

# Polite delay between requests (seconds). Keyed API can go faster.
# Override with S2_REQUEST_DELAY to slow down for the saturated public pool
# (e.g. S2_REQUEST_DELAY=4 for an off-peak unauthenticated run).
_HAS_API_KEY = bool(os.environ.get("S2_API_KEY"))
_DEFAULT_DELAY = 1.1 if not _HAS_API_KEY else 0.1
REQUEST_DELAY = float(os.environ.get("S2_REQUEST_DELAY", _DEFAULT_DELAY))

# Retry policy for transient failures (429 / 5xx). Configurable via env so a
# run against the saturated public pool can be made more patient without edits.
# Defaults tuned for the brutal unauthenticated pool: start at 4s, double up to
# a 45s cap, and give more attempts since 429s are the norm not the exception.
MAX_RETRIES = int(os.environ.get("S2_MAX_RETRIES", 7))
BACKOFF_BASE = float(os.environ.get("S2_BACKOFF_BASE", 4.0))  # grows as BASE * 2**attempt
# Cap any single backoff/Retry-After wait so a huge server hint can't hang the run.
MAX_BACKOFF = float(os.environ.get("S2_MAX_BACKOFF", 45.0))

# --- Candidate filtering / fetching (steps 6-8) ---------------------------- #

# Keyword filter applied to title+abstract.
# Each entry is (pattern, is_regex). Most are plain case-insensitive substrings;
# "lora" uses a word boundary so it does not match "flora"/"coloration".
FILTER_KEYWORDS: list[tuple[str, bool]] = [
    ("fine-tun", False),
    ("adapter", False),
    (r"\blora\b", True),
    ("prompt tuning", False),
    ("prefix-tuning", False),
    ("parameter-efficient", False),
    ("low-rank", False),
]
_REGEX_KEYWORDS = [re.compile(p, re.IGNORECASE) for p, is_rx in FILTER_KEYWORDS if is_rx]
_PLAIN_KEYWORDS = [p for p, is_rx in FILTER_KEYWORDS if not is_rx]

# Target corpus size and the CI guardrail band (schema §4): aim for 70, band 60-80.
POOL_TARGET = 70
POOL_TARGET_MIN = 60

# Over-fetch buffer: we do NOT stop at 70. We fetch keyword-passing candidates
# until the pool reaches CANDIDATE_FETCH_TARGET (with an upper bound), so manual
# review (the APPLIES test, schema §3.5) trims a buffer instead of backfilling a
# shortfall. The final corpus is ~70; the buffer is 150-200.
CANDIDATE_FETCH_TARGET = 175  # aim for the middle of the 150-200 band
CANDIDATE_FETCH_LIMIT = 200   # hard ceiling

# --------------------------------------------------------------------------- #
# Seeds -- the 7 highest-connectivity PEFT papers (one per method family root,
# plus QLoRA), keyed by arXiv id. We resolve each seed DIRECTLY via
# GET /paper/ARXIV:{id} instead of /paper/search: the search endpoint is the
# most aggressively throttled path (it 429'd LoRA/Prompt-Tuning/QLoRA/(IA)^3 to
# death in the previous run), and we already know each seed's arXiv id.
# --------------------------------------------------------------------------- #

SEED_ARXIV: dict[str, str] = {
    "LoRA: Low-Rank Adaptation of Large Language Models": "2106.09685",
    "Parameter-Efficient Transfer Learning for NLP (Adapters)": "1902.00751",
    "Prefix-Tuning: Optimizing Continuous Prompts for Generation": "2101.00190",
    "The Power of Scale for Parameter-Efficient Prompt Tuning": "2104.08691",
    "QLoRA: Efficient Finetuning of Quantized LLMs": "2305.14314",
    "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based MLMs": "2106.10199",
    "Few-Shot PEFT is Better and Cheaper than In-Context Learning ((IA)^3)": "2205.05638",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_papers")


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


def _session() -> requests.Session:
    s = requests.Session()
    key = os.environ.get("S2_API_KEY")
    if key:
        s.headers["x-api-key"] = key
    s.headers["User-Agent"] = "peft-kg-fetcher/1.0"
    return s


SESSION = _session()


def _backoff_wait(attempt: int, resp: "requests.Response | None") -> float:
    """Seconds to wait before the next retry.

    Prefers the server's Retry-After header when present (the API's own hint is
    more accurate than our guess), otherwise exponential backoff. Capped at
    MAX_BACKOFF so a pathological hint can't stall the run.
    """
    if resp is not None:
        hint = resp.headers.get("Retry-After")
        if hint:
            try:
                return min(float(hint), MAX_BACKOFF)  # Retry-After may be integer seconds
            except ValueError:
                pass  # HTTP-date form is rare here; fall through to backoff
    return min(BACKOFF_BASE * (2 ** attempt), MAX_BACKOFF)


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """GET with polite delay + Retry-After-aware exponential backoff on 429/5xx.

    Returns the parsed JSON dict, or None if the resource is unavailable after
    retries (e.g. persistent 404 or exhausted retries).
    """
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = _backoff_wait(attempt, None)
            log.warning("Request error for %s (%s); retrying in %.1fs", url, exc, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 404:
            log.warning("404 Not Found: %s", url)
            return None

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = _backoff_wait(attempt, resp)
            log.warning(
                "HTTP %s for %s; retry %d/%d in %.1fs",
                resp.status_code, url, attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue

        # Other 4xx: not retryable.
        log.error("HTTP %s for %s: %s", resp.status_code, url, resp.text[:200])
        return None

    log.error("Exhausted %d retries for %s", MAX_RETRIES, url)
    return None


def _post(url: str, params: dict[str, Any], json_body: dict[str, Any]) -> Any | None:
    """POST with the same delay + Retry-After-aware backoff as _get.

    Used for the /paper/batch endpoint. Returns the parsed JSON (a list for
    batch), or None after exhausted retries.
    """
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.post(url, params=params, json=json_body, timeout=60)
        except requests.RequestException as exc:
            wait = _backoff_wait(attempt, None)
            log.warning("POST error for %s (%s); retrying in %.1fs", url, exc, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = _backoff_wait(attempt, resp)
            log.warning(
                "POST HTTP %s for %s; retry %d/%d in %.1fs",
                resp.status_code, url, attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue

        log.error("POST HTTP %s for %s: %s", resp.status_code, url, resp.text[:200])
        return None

    log.error("Exhausted %d retries for POST %s", MAX_RETRIES, url)
    return None


# --------------------------------------------------------------------------- #
# Title matching
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def title_similarity(a: str, b: str) -> float:
    """Jaccard token overlap between two titles. 1.0 = identical token sets."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------- #
# API operations
# --------------------------------------------------------------------------- #


def resolve_seed(seed_title: str, arxiv_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a seed directly by arXiv id via GET /paper/ARXIV:{id}.

    Bypasses /paper/search entirely (the most-throttled endpoint). Returns
    (paperId, detail) or (None, None) on failure. Since we fetch detail here
    anyway, we return it so the caller need not make a second call.
    """
    detail = _get(PAPER_URL.format(paper_id=f"ARXIV:{arxiv_id}"),
                  params={"fields": DETAIL_FIELDS})
    if not detail or not detail.get("paperId"):
        log.warning("Could not resolve seed %r (ARXIV:%s)", seed_title, arxiv_id)
        return None, None
    log.info("Resolved seed %r -> %s (ARXIV:%s)", seed_title, detail["paperId"], arxiv_id)
    return detail["paperId"], detail


def fetch_candidates_batch(paper_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch detail for many candidate papers via the batch endpoint.

    One POST per <=500 ids instead of one GET per paper -- the whole point of the
    refactor. Returns {paperId: detail}. The batch endpoint returns a list
    aligned with the input ids; null entries (unresolvable ids) are skipped.
    """
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(paper_ids), BATCH_MAX_IDS):
        chunk = paper_ids[start:start + BATCH_MAX_IDS]
        log.info("Batch-fetching %d candidate details (offset %d)...", len(chunk), start)
        data = _post(BATCH_URL, params={"fields": CANDIDATE_FIELDS}, json_body={"ids": chunk})
        if not data:
            log.warning("Batch chunk at offset %d failed; skipping", start)
            continue
        for item in data:
            if item and item.get("paperId"):
                out[item["paperId"]] = item
    return out


def _fetch_linked(url: str, container_key: str, paper_id: str) -> list[dict[str, Any]]:
    """Fetch a paginated list of linked papers (citations or references).

    `container_key` is "citingPaper" or "citedPaper" -- the nested object under
    each row that holds the actual paper.
    """
    # The API hard-rejects (HTTP 400) any page where offset + limit >= 10000.
    # Cap pagination at that ceiling and shrink the final page's limit to fit,
    # so highly-cited seeds (LoRA has 9000+ citations) stop cleanly instead of
    # erroring. 10000 linked papers is far more than the candidate pool needs.
    PAGINATION_CEILING = 10_000
    results: list[dict[str, Any]] = []
    offset = 0
    while offset < PAGINATION_CEILING:
        limit = min(LINK_PAGE_SIZE, PAGINATION_CEILING - offset)
        data = _get(
            url.format(paper_id=paper_id),
            params={"fields": LINK_FIELDS, "limit": limit, "offset": offset},
        )
        if not data:
            break
        rows = data.get("data", [])
        for row in rows:
            linked = row.get(container_key)
            if linked and linked.get("paperId"):
                results.append(linked)
        # The API returns "next" only when more pages exist.
        if "next" not in data or not rows:
            break
        offset = data["next"]
    return results


def fetch_citations(paper_id: str) -> list[dict[str, Any]]:
    """Papers that cite this paper."""
    return _fetch_linked(CITATIONS_URL, "citingPaper", paper_id)


def fetch_references(paper_id: str) -> list[dict[str, Any]]:
    """Papers that this paper references."""
    return _fetch_linked(REFERENCES_URL, "citedPaper", paper_id)


# --------------------------------------------------------------------------- #
# Candidate filtering (step 6)
# --------------------------------------------------------------------------- #


def _count_keywords(text: str) -> int:
    """Distinct filter keywords present in a single lowercased text field."""
    n = sum(1 for kw in _PLAIN_KEYWORDS if kw in text)
    n += sum(1 for rx in _REGEX_KEYWORDS if rx.search(text))
    return n


def keyword_hits(title: str | None, abstract: str | None) -> int:
    """Distinct filter keywords appearing anywhere in title+abstract (>=1 passes)."""
    return _count_keywords(f"{title or ''} {abstract or ''}".lower())


def matches_keywords(title: str | None, abstract: str | None) -> bool:
    """True if at least one filter keyword appears in title+abstract."""
    return keyword_hits(title, abstract) > 0


# Relevance scoring weights (see candidate_score).
_TITLE_HIT_WEIGHT = 10   # a keyword in the TITLE is worth much more than in the abstract
_ABSTRACT_HIT_WEIGHT = 1
_MULTI_KEYWORD_BONUS = 5  # flat bonus for matching 2+ distinct keywords overall


def candidate_score(title: str | None, abstract: str | None) -> int:
    """Relevance score for ranking the buffer (higher = more relevant).

    Per spec: title matches outrank abstract-only matches, and matching 2+
    distinct keywords outranks matching exactly 1. Concretely:
      score = 10 * (distinct keywords in title)
            +  1 * (distinct keywords in abstract only)
            +  5 * (1 if total distinct keywords >= 2 else 0)
    A title hit (>=10) always beats any number of abstract-only hits at this
    scale (there are 7 keywords), satisfying the title>abstract requirement.
    """
    t = (title or "").lower()
    a = (abstract or "").lower()
    title_hits = _count_keywords(t)
    total_hits = keyword_hits(title, abstract)
    # Abstract-only hits = keywords present in title+abstract but not in title.
    abstract_only_hits = max(total_hits - title_hits, 0)

    score = _TITLE_HIT_WEIGHT * title_hits + _ABSTRACT_HIT_WEIGHT * abstract_only_hits
    if total_hits >= 2:
        score += _MULTI_KEYWORD_BONUS
    return score


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class SeedPaper:
    seed_title: str
    paper_id: str | None = None
    detail: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)


def build_candidate_pool(seeds: Iterable[SeedPaper]) -> dict[str, dict[str, Any]]:
    """Deduplicate all citing/cited papers into a pool keyed by paperId.

    Records how each candidate was reached (which seed, via citation/reference).
    """
    pool: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if not seed.paper_id:
            continue
        for rel, papers in (("citation", seed.citations), ("reference", seed.references)):
            for p in papers:
                pid = p["paperId"]
                entry = pool.setdefault(
                    pid,
                    {
                        "paperId": pid,
                        "title": p.get("title"),
                        "externalIds": p.get("externalIds"),
                        "sources": [],
                    },
                )
                entry["sources"].append({"seed": seed.paper_id, "relation": rel})
    return pool


def rank_candidates(pool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Order pool entries most-central first.

    Centrality = number of DISTINCT seeds that reached the candidate. A paper
    cited by several PEFT seeds is more likely to be core PEFT work than one
    reached from a single seed. This is what makes the over-fetch buffer useful:
    the strongest candidates are fetched (and reviewed) first, so a truncated
    buffer still captures the most relevant papers.
    """
    def centrality(entry: dict[str, Any]) -> tuple[int, int]:
        distinct_seeds = len({s["seed"] for s in entry["sources"]})
        return (distinct_seeds, len(entry["sources"]))

    return sorted(pool.values(), key=centrality, reverse=True)


def run(
    seed_arxiv: dict[str, str],
) -> tuple[list[SeedPaper], dict[str, dict[str, Any]]]:
    """Resolve seeds by arXiv id, fetch ONLY their citations/references, then
    build the candidate pool. Candidate details are fetched later in one batch."""
    seeds: list[SeedPaper] = []

    for title, arxiv_id in seed_arxiv.items():
        seed = SeedPaper(seed_title=title)
        seed.paper_id, seed.detail = resolve_seed(title, arxiv_id)
        if not seed.paper_id:
            seeds.append(seed)
            continue

        # Citations/references are fetched for SEEDS ONLY -- candidates need only
        # scoring detail, not their own citation graphs.
        seed.citations = fetch_citations(seed.paper_id)
        seed.references = fetch_references(seed.paper_id)
        log.info(
            "%s: %d citations, %d references",
            seed.paper_id, len(seed.citations), len(seed.references),
        )
        seeds.append(seed)

    pool = build_candidate_pool(seeds)
    log.info("Candidate pool: %d unique papers", len(pool))
    return seeds, pool


def _record_from_detail(
    paper_id: str,
    detail: dict[str, Any],
    citations: list[dict[str, Any]] | None = None,
    references: list[dict[str, Any]] | None = None,
    *,
    role: str = "candidate",
    score: int | None = None,
    keyword_hit_count: int | None = None,
    distinct_seeds: int | None = None,
) -> dict[str, Any]:
    """Shape a paper-detail dict into an output record.

    authors -> list of author names; citations/references -> lists of paperIds.
    Curation aids (`_role`, `_score`, `_keyword_hits`, `_distinct_seeds`) help
    manual review pick the final ~70; they are metadata, strip before graph load.
    """
    authors = [a.get("name") for a in (detail.get("authors") or []) if a.get("name")]
    rec: dict[str, Any] = {
        "paper_id": paper_id,
        "title": detail.get("title"),
        "abstract": detail.get("abstract"),
        "year": detail.get("year"),
        "authors": authors,
        "venue": detail.get("venue"),
        "citations": [p["paperId"] for p in (citations or [])],
        "references": [p["paperId"] for p in (references or [])],
        "_role": role,  # "seed" | "candidate" -- for curation, strip before load
    }
    if score is not None:
        rec["_score"] = score
    if keyword_hit_count is not None:
        rec["_keyword_hits"] = keyword_hit_count
    if distinct_seeds is not None:
        rec["_distinct_seeds"] = distinct_seeds
    return rec


# Seeds are guaranteed-relevant anchors; give them a sentinel score above any
# candidate so they sort to the top of the buffer.
_SEED_SCORE = 10_000


def build_candidates(
    seeds: list[SeedPaper], pool: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the over-fetched candidate buffer, sorted by relevance score desc.

    Seeds are always included (with real citation/reference ids) and pinned to
    the top. Candidates are fetched most-central-first (so the fetch budget is
    spent on the papers most seeds point at) and kept if they pass the keyword
    filter, until the pool reaches CANDIDATE_FETCH_TARGET (hard ceiling
    CANDIDATE_FETCH_LIMIT). The returned list is sorted by `_score` descending
    per the scoring spec; manual curation (the APPLIES test, schema §3.5) then
    trims it down to the final ~70.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for s in seeds:
        if s.paper_id and s.detail:
            records.append(
                _record_from_detail(
                    s.paper_id, s.detail, s.citations, s.references,
                    role="seed", score=_SEED_SCORE,
                )
            )
            seen.add(s.paper_id)

    # Rank the whole pool by centrality, then take a generous top slice to
    # batch-fetch. We over-request beyond CANDIDATE_FETCH_TARGET because some
    # candidates won't pass the keyword filter and shouldn't count toward target.
    # One batch POST (<=500 ids) replaces hundreds of per-paper GETs.
    ranked = [e for e in rank_candidates(pool) if e["paperId"] not in seen]
    slice_ids = [e["paperId"] for e in ranked[:BATCH_MAX_IDS]]
    source_by_id = {e["paperId"]: e for e in ranked}
    log.info(
        "Seeds in buffer: %d; batch-fetching top %d ranked candidates...",
        len(records), len(slice_ids),
    )

    details = fetch_candidates_batch(slice_ids)

    # Keep ranking order (centrality) while filtering + scoring.
    kept = 0
    for pid in slice_ids:
        if kept >= CANDIDATE_FETCH_LIMIT:
            break
        detail = details.get(pid)
        if not detail:
            continue
        title, abstract = detail.get("title"), detail.get("abstract")
        hits = keyword_hits(title, abstract)
        if hits == 0:
            continue

        entry = source_by_id[pid]
        records.append(
            _record_from_detail(
                pid, detail,
                role="candidate",
                score=candidate_score(title, abstract),
                keyword_hit_count=hits,
                distinct_seeds=len({s["seed"] for s in entry["sources"]}),
            )
        )
        kept += 1

    # Sort by relevance score descending; seeds float to the top via sentinel.
    records.sort(key=lambda r: r.get("_score", 0), reverse=True)

    log.info("Buffer: %d seeds + %d candidates = %d total",
             len(records) - kept, kept, len(records))
    if kept < 150:
        log.warning(
            "Only %d candidates (target 150-200); widen keywords or add seeds",
            kept,
        )
    return records


def print_summary(records: list[dict[str, Any]]) -> None:
    """Step 9: print counts over the candidate buffer."""
    total = len(records)
    seeds = sum(1 for p in records if p.get("_role") == "seed")
    candidates = total - seeds
    with_abstract = sum(1 for p in records if (p.get("abstract") or "").strip())
    missing_authors = sum(1 for p in records if not p.get("authors"))
    missing_venue = sum(1 for p in records if not (p.get("venue") or "").strip())

    print("\n" + "=" * 56)
    print("SUMMARY  (candidate BUFFER, not the final corpus)")
    print("=" * 56)
    print(f"Total papers fetched      : {total}  (seeds {seeds} + candidates {candidates})")
    print(f"With non-empty abstract   : {with_abstract}")
    print(f"Missing authors           : {missing_authors}")
    print(f"Missing venue             : {missing_venue}")
    print("-" * 56)
    print(f"Output is sorted by _score desc. Curate down to ~{POOL_TARGET} using")
    print("the APPLIES test, top-scoring first. Strip _score/_role/_keyword_hits/")
    print("_distinct_seeds before loading into the graph.")
    print("=" * 56)


def main() -> None:
    seeds, pool = run(SEED_ARXIV)
    records = build_candidates(seeds, pool)

    out_path = str(ROOT / "data" / "papers_candidates.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s (%d papers, sorted by score)", out_path, len(records))

    print_summary(records)


if __name__ == "__main__":
    main()
