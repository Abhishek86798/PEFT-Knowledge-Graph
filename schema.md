# PEFT Knowledge Graph — Schema Definition

This document is normative. Every node and edge in the graph must pass the tests written here. If a candidate fact fails the test, it does not go in the graph — even if it "feels" true.

---

## 1. Entity Types

### 1.1 `Method`
A named parameter-efficient fine-tuning technique with a distinct, describable parameter-update mechanism.

**Inclusion test:** A Method node exists iff (a) it has a canonical name used by the community, and (b) one can state in one sentence *which parameters are trained and how they are injected* — e.g., "LoRA: trains low-rank matrices A,B added to frozen weight matrices as W + BA."

**Properties:**
- `name` (string, canonical, e.g. "LoRA")
- `slug` (string, primary key): sanitized lowercase identifier (e.g. `method_lora`) generated once at creation, never derived from `name` at query time. See §6.1 for why.
- `family` (enum, exactly one): `additive-adapter` | `soft-prompt` | `reparameterization` | `selective` | `multiplicative`
- `is_family_root` (bool): true iff the method *defines* its family's core mechanism rather than modifying another method's
- `introduced_date` (date, YYYY-MM-DD): the arXiv v1 submission date of the introducing paper, denormalized for cheap temporal queries. Day-level resolution is not optional in this domain: multiple foundational methods can land within months of each other in a single year (2021 alone: Prefix-Tuning in January, P-Tuning in March, Prompt-Tuning in April, LoRA and BitFit in June), and both INTRODUCES tie-breaking and EXTENDS chronology checks consume the exact date. Year is derived, never stored separately. `Paper.arxiv_v1_date` via INTRODUCES remains authoritative; a validation pass must assert the two agree.
- `combined_techniques` (list of tags, may be empty): controlled vocabulary `quantization` | `pruning` | `hypercomplex-parameterization`. Populated iff the method's paper integrates an orthogonal, pre-existing, non-PEFT technique that does not participate in the EXTENDS reduction test. Kept as a property rather than an edge because only two tracked methods use it (QLoRA, Compacter); a `Technique` entity earns its place only if this vocabulary grows beyond a handful of tags.

**Design decision — family is a property, not a node.** A `Family` entity would invite untestable `BELONGS_TO` edges. As an enum property, every Method gets exactly one family assignment, decided by mechanism, and disputes are resolved by the reduction test in §3.

### 1.2 `Paper`
A published document (arXiv or venue) that either introduces a Method or reports evaluations of tracked Methods.

**Inclusion test:** A Paper node exists iff it (a) introduces a tracked Method, (b) provides evaluation/comparison evidence backing an edge, or (c) applies a tracked Method in its own experiments (§3.5). Clause (c) is what carries the corpus from ~13 method-introducing papers to the 70-paper target: application papers, third-party benchmarking papers, and empirical surveys enter through APPLIES. Papers that merely *cite* tracked methods — including purely narrative surveys that run no experiments — remain excluded.

**Properties:** `title`, `arxiv_id`, `arxiv_v1_date` (YYYY-MM-DD; the precedence-deciding date), `year` (derived from `arxiv_v1_date`), `venue` (nullable).

**Note on multi-purpose papers.** Some papers introduce one method while also running others — e.g., (IA)³'s introducing paper (T-Few) is primarily a few-shot recipe paper that applies several existing PEFT methods alongside introducing (IA)³ itself. This is expected, not a schema violation: a paper's INTRODUCES edge is scoped to the one method it originates (§3.2), and it may separately hold APPLIES edges to any other methods it runs (§3.5) — the validation rule barring a paper from both INTRODUCES and APPLIES the *same* method (§4) still holds. The corpus-size math (§Assumptions) is edge-count-driven, not paper-role-driven: the ~13 figure counts distinct introduced methods, not a mutually exclusive bucket of "introducing papers."

### 1.3 `Benchmark`
A named evaluation target: a dataset or standardized task suite (GLUE, SuperGLUE, E2E NLG, XSum, MMLU, WikiSQL). Dataset and benchmark are intentionally unified into one entity type: the PEFT literature uses the terms interchangeably in results tables, and separating them would add a distinction the source papers themselves do not maintain, for no analytical benefit at this scope. Properties: `name`, `granularity` (enum: `suite` | `dataset` — most tracked benchmarks are suites like GLUE, but single-dataset benchmarks such as E2E NLG and WikiSQL carry `granularity: dataset`, since they have no sub-tasks to aggregate).

**Inclusion test:** A Benchmark node exists iff it has at least one EVALUATED_ON edge (§3.4) — i.e., iff a tracked method's *introducing paper* reports a number on it in a main results table. This is deliberately narrower than "any paper evaluates on it": EVALUATED_ON is scoped to introducing papers only (§3.4), so Benchmark inclusion inherits that scope rather than defining a separate, looser test that could admit benchmarks reachable only through APPLIES papers — which would leave them with zero EVALUATED_ON edges and no other edge type to justify their existence. Granularity rule: track the *suite* (GLUE), not its sub-tasks (CoLA, SST-2), unless a paper evaluates on the sub-task *alone* — in which case use `subset_evaluated` on the edge (§3.4), not a separate node.

That's it. Three entity types. No `Model` (backbone architectures), no `Metric`, no `Author` — see §5.

---

## 2. Method Taxonomy

The classification rule used for every disputed case:

> **Reduction test:** Method A is a *variant* of Method B if disabling or fixing A's novel components recovers B's trainable mechanism unchanged. If nothing reduces to an existing method, A is a family root.

| Method | Status | Family | Rationale (one line) |
|---|---|---|---|
| Adapters (Houlsby '19) | **root** | additive-adapter | Defines bottleneck-module insertion; the family's origin. |
| Pfeiffer Adapters | variant of Adapters | additive-adapter | Repositions/prunes Houlsby modules; reduces to Houlsby config. |
| Compacter | variant of Adapters | additive-adapter | Replaces adapter weights with Kronecker/hypercomplex parameterization of the same module. |
| Prefix-Tuning | **root** | soft-prompt | Defines trainable continuous vectors in every layer's KV — the deep soft-prompt mechanism. |
| Prompt-Tuning | variant of Prefix-Tuning | soft-prompt | Restricts prefixes to the input layer only; a strict simplification. Community discourse (including Lester et al.'s own framing) often treats these as siblings rather than parent-child — we classify by mechanism-subsumption, not citation framing or authorial self-positioning (see Assumptions). |
| P-Tuning | **root** | soft-prompt | Independent, contemporaneous mechanism (LSTM/MLP prompt encoder at input); does not reduce to Prefix-Tuning. |
| P-Tuning v2 | variant (dual parent) | soft-prompt | Explicitly merges P-Tuning's framing with Prefix-Tuning's deep prompts; EXTENDS both. |
| LoRA | **root** | reparameterization | Defines the low-rank ΔW decomposition. |
| AdaLoRA | variant of LoRA | reparameterization | Fix rank budget to uniform → plain LoRA. Passes reduction test cleanly. |
| **QLoRA** | **variant of LoRA** | reparameterization | Trainable mechanism is unmodified LoRA; all novelty (NF4, double quantization, paged optimizers) lives in the frozen base. Modeled as EXTENDS LoRA with `combined_techniques: [quantization]`. Not a family root. |
| VeRA | variant of LoRA | reparameterization | Borderline: doesn't reduce cleanly (freezes A,B; trains scaling vectors), but its paper explicitly modifies LoRA's decomposition — passes the EXTENDS rule (§3), fails `is_family_root`. |
| BitFit | **root** | selective | Defines subset-of-existing-parameters tuning (biases); no new parameters injected. |
| (IA)³ | **root** | multiplicative | Learned element-wise rescaling vectors on K/V/FFN activations; neither additive nor low-rank; nothing to reduce to. |

Roots: **Adapters, Prefix-Tuning, P-Tuning, LoRA, BitFit, (IA)³.** Everything else is a variant with an `EXTENDS` edge.

---

## 3. Relationship Types

Each definition is a *test*, written so that a human or an LLM tagger applying it to a paper gets the same answer.

### 3.1 `EXTENDS` — Method → Method
> **A EXTENDS B iff both hold: (1) A's paper explicitly presents A as a modification of B's parameter-update mechanism (not merely cites B for comparison or motivation), and (2) B's mechanism is structurally recoverable from A — by disabling A's novel components or as A's identified special case.**

Clause (1) alone would admit rhetorical framing ("inspired by"); clause (2) alone would admit coincidental reductions. The conjunction is what separates this graph from a citation graph: *LoRA's paper cites Adapters extensively, but LoRA does not EXTENDS Adapters* — different mechanism, no reduction. Multiple parents are allowed (P-Tuning v2). Cardinality: 0..n outgoing; roots have 0.

### 3.2 `INTRODUCES` — Paper → Method
> **Paper P INTRODUCES Method M iff P is the first publication to define M's mechanism and assign it its canonical name.**

Exactly one INTRODUCES edge per Method (arXiv v1 date breaks ties). A paper introducing a variant introduces the variant only, never the parent.

### 3.3 `COMPARED_AGAINST` — Method → Method (directional, with provenance)
> **A COMPARED_AGAINST B iff B appears as a baseline row in a main results table of A's introducing paper, evaluated on the same benchmark and backbone as A in that table.**

**Edge property:** `evidence_paper` (Paper reference) — the paper whose results table backs the edge.

A note on shape, because it is a genuine design fork: comparisons happen *inside papers*, not inside methods, so "LoRA COMPARED_AGAINST Adapters" read naively is misleading — it was the LoRA *paper* that ran the comparison. The alternative shape `Paper → COMPARED_AGAINST → Method` fixes the semantics but breaks the query the graph exists to answer: "what were LoRA's baselines" becomes a two-hop join, and the proposing method is only recoverable by leaning on INTRODUCES being exactly-one — an invariant of a different edge. We instead keep `Method → Method` and reify provenance into `evidence_paper`. This is exactly as expressive as the Paper-anchored shape, keeps the common query at one hop, and if §5 item 4 is ever relaxed, third-party comparisons slot in by pointing `evidence_paper` elsewhere.

"Main results table" excludes appendix tables and ablations. Mentions in related work, discussion, or motivation do **not** qualify — this is the strict test that keeps the edge meaningful. Directional and not symmetrized: LoRA COMPARED_AGAINST Adapters does not imply the reverse (Adapters' 2019 paper could not have compared against LoRA).

### 3.4 `EVALUATED_ON` — Method → Benchmark
> **M EVALUATED_ON B iff M's introducing paper reports at least one quantitative result for M on B in a main results table.**

**Optional edge property:** `subset_evaluated` (string, e.g. `"MMLU-Math"`, `"GLUE: CoLA, SST-2 only"`) — populated only when the paper explicitly evaluates on a proper subset of the suite. This preserves the suite-level granularity rule in §1.3 (the node is still MMLU, not MMLU-Math) while recording generative-era cherry-picking without fracturing the Benchmark node set. Absent property = full-suite or standard-split evaluation.

Deliberately scoped to the *introducing* paper only. If we admitted any paper's evaluation of any method, every popular method would connect to every popular benchmark and the edge would carry no signal. Note: the edge records *that* an evaluation exists, not the score — see §5.

### 3.5 `APPLIES` — Paper → Method
> **P APPLIES M iff P runs M's mechanism unmodified in its own experiments and reports at least one quantitative result produced with it.**

This is the corpus-breadth edge: it gives the non-introducing papers — 55 of them, whatever it takes to hit the 70 total Paper target (§4 validation) — a legitimate reason to exist in the graph: applications, third-party benchmarks, empirical surveys. It answers a question no other edge can — *who actually uses each method in practice*. Its boundaries are set by two words in the test:
- **"unmodified"** — the disjointness clause. If P modifies M's mechanism, P fails APPLIES and its contribution is a candidate new Method, judged by the reduction test in §2. APPLIES and EXTENDS can therefore never describe the same relationship.
- **"reports at least one quantitative result"** — the rigor clause, mirroring §3.3/§3.4. Citing M, discussing M, or surveying M without running it does not qualify; a narrative survey stays out (consistent with §5 item 4), while a survey that *re-benchmarks* methods enters.

Cardinality: 0..n; one paper commonly applies several methods. An introducing paper does not get an APPLIES edge to the method it introduces (INTRODUCES subsumes it), but may get APPLIES edges to *other* methods it runs — and separately, those same runs may also ground COMPARED_AGAINST edges from the introduced method (§3.3). The two edge types are not interchangeable: COMPARED_AGAINST is Method→Method (a ranking claim) while APPLIES is Paper→Method (an adoption claim); a tagger should record both where both tests are met, not substitute one for the other.

### 3.6 `combined_techniques` — demoted from edge to Method property
An earlier draft modeled this as a `COMBINES: Method → Technique-tag` edge. It was the only edge whose target was not a first-class entity — an inconsistency in the schema's shape — and only two tracked methods use it. It is therefore a list property on Method (§1.1), governed by the same test:
> **A technique tag is added iff the method's paper integrates an orthogonal, pre-existing technique that is not itself a PEFT method and does not participate in the EXTENDS reduction test.**

This still cleanly models QLoRA (EXTENDS LoRA; `combined_techniques: [quantization]`) without polluting the Method taxonomy. Promote to a real `Technique` entity only if the vocabulary outgrows a handful of tags.

---

## 4. Schema Summary (machine-readable)

```yaml
entities:
  Method:
    properties: [name, slug, family, is_family_root, introduced_date, combined_techniques]
    family_enum: [additive-adapter, soft-prompt, reparameterization, selective, multiplicative]
    technique_tag_enum: [quantization, pruning, hypercomplex-parameterization]
  Paper:
    properties: [title, arxiv_id, arxiv_v1_date, year, venue]
  Benchmark:
    properties: [name, granularity]   # granularity: suite | dataset

relationships:
  EXTENDS:            {from: Method, to: Method,    cardinality: "0..n", test: "explicit modification claim AND structural reduction"}
  INTRODUCES:         {from: Paper,  to: Method,    cardinality: "exactly 1 per Method", test: "first paper to define and name the mechanism"}
  COMPARED_AGAINST:   {from: Method, to: Method,    cardinality: "0..n, directional", properties: [evidence_paper], test: "baseline row in a main results table of evidence_paper"}
  EVALUATED_ON:       {from: Method, to: Benchmark, cardinality: "0..n", properties: [subset_evaluated?], test: "quantitative result in introducing paper's main results table"}
  APPLIES:            {from: Paper,  to: Method,    cardinality: "0..n", test: "runs the mechanism unmodified in own experiments, reports >=1 result"}

validation:
  - Method.introduced_date == Paper.arxiv_v1_date of its INTRODUCES source
  - EXTENDS edges form a DAG (no cycles)
  - chronology: if A EXTENDS B, then B.introduced_date < A.introduced_date. Resolution procedure (warning-level, not a hard failure) — if a violation fires, check both papers' arXiv v1 dates: if within 60 days, mark the edge `concurrent: true` and suppress the warning; if further apart, treat it as a tagging error and re-verify the EXTENDS direction. This procedure exists so the warning is triaged once per edge, not re-litigated every ingestion run.
  - every COMPARED_AGAINST.evidence_paper must be an existing Paper node
  - is_family_root == true implies zero outgoing EXTENDS edges
  - no Paper both INTRODUCES and APPLIES the same Method
  - every Paper node has >=1 of {INTRODUCES, APPLIES, evidence_paper backreference}  # no orphan papers
  - every Benchmark node has >=1 EVALUATED_ON edge  # no orphan benchmarks (see §1.3)
  - corpus target: 70 Papers; validation guardrail fails outside 60-80 (~13 via INTRODUCES, remainder via APPLIES). The guardrail band is intentionally wider than the single target number — 70 is what you aim for, outside 60-80 fails CI.
```

---

## 5. Explicitly NOT Modeled

1. **Metric values.** `EVALUATED_ON` is boolean, not numeric. Scores depend on backbone, parameter budget, seed, and eval script; normalizing them across 68 papers is a multi-day rabbit hole that produces numbers nobody should compare anyway. The graph answers "was it evaluated there," not "what did it score."
2. **Backbone models** (BERT, T5, GPT-3, LLaMA) as entities. Every method touches many backbones; the edges would be dense and low-signal. Backbone *is* consulted transiently during edge construction — §3.3's same-backbone condition is a filter applied while deciding whether to draw a COMPARED_AGAINST edge — but it is discarded afterward, not stored. The test depends on data read from the paper at tagging time, not on data the graph retains.
3. **Authors.** Deliberately excluded, not overlooked: author nodes answer sociology-of-science questions (who collaborates with whom), not mechanism questions (what extends what), and this graph is scoped to the latter. Author overlap is also explicitly barred from influencing family assignment (see Assumptions).
4. **Third-party numbers as *evaluation evidence*.** Non-introducing papers enter the graph via APPLIES (§3.5), but only introducing papers generate EVALUATED_ON and COMPARED_AGAINST edges. Surveys and third-party benchmarks re-report numbers under inconsistent setups; letting them mint comparison/evaluation edges would make edge provenance untraceable. So: a third-party paper can *exist* and *apply* methods, but cannot testify about how methods rank against each other.
5. **Citation edges.** Deliberately excluded — the entire point of the EXTENDS test is that citation ≠ extension. Semantic Scholar already provides the citation graph; rebuilding it adds nothing.
6. **Hyperparameters and parameter counts** (adapter dim, LoRA rank, % trainable params). Configuration-dependent, not method-identity; belongs in a spec sheet, not a graph.
7. **Temporal versioning** (LoRA-the-2021-paper vs LoRA-as-implemented-in-PEFT-lib-2026). One node per canonical mechanism; implementation drift is out of scope.
8. **Negative results and failed comparisons.** Papers rarely report them consistently; absence of an edge already carries no meaning, so modeling absence explicitly would be false precision.

The trade in every case is the same: a smaller graph where every edge has a falsifiable test beats a larger graph where edges mean "somebody mentioned this somewhere."

---

## Assumptions

- **Canonical names** are taken from each method's introducing paper; community aliases (e.g. "(IA)³" vs "IA3") resolve to one node.
- **Precedence** for INTRODUCES ties is decided by arXiv v1 submission date, not venue publication date.
- **Family assignment** is mechanism-based (the reduction test in §2), never based on citation lineage, author overlap, or self-description in abstracts.
- **Construction cost is asymmetric by edge type.** APPLIES and EVALUATED_ON edges can be tagged from abstracts and results tables (LLM-assistable at scale). EXTENDS cannot: honestly applying the reduction test requires reading the method section of both papers. The taxonomy table in §2 embodies that reading for the current ~13 methods; any new root or variant added later carries the same full-read cost, budgeted as human time, not skim time.

---

## 6. Physical Implementation Notes

The sections above are the logical schema; enforcing it in a real database requires the following, roughly in the order you'll hit them.

**6.1 Canonical IDs — never key on names or titles.** `Paper.title` changes between arXiv versions and camera-ready ("LoRA: Low-Rank Adaptation of Large Language Models" has cosmetic variants across indexes), and `Method.name` clashes and aliases ("(IA)³" / "IA3" / "IA^3"). Primary keys: `arxiv_id` for Papers; a sanitized lowercase slug for Methods (`method_lora`, `method_qlora`, `method_ptuning_v2`) generated once at node creation and never regenerated from the display name. The alias-resolution assumption (§Assumptions) is *implemented* here: aliases map to slugs in a lookup table, not by string matching at query time.

**6.2 Uniqueness constraints at the engine level.** Enforce exactly-one INTRODUCES per Method as a database constraint where the engine supports it (Neo4j: uniqueness on a materialized `introduced_by` property plus a periodic edge-count assertion, since Neo4j cannot constrain edge cardinality natively; SQL: a `UNIQUE` index on `introduces.method_slug`). Do not rely on ingestion code alone — the constraint exists to catch the ingestion bug you haven't written yet.

**6.3 EXTENDS must be a DAG — check it, don't assume it.** Cycles are logically impossible under the reduction test (A cannot reduce to B while B reduces to A) but *operationally* possible under tagging error. Run a cycle check (topological sort or Neo4j's `apoc.nodes.cycles`) as a post-ingestion validation step, and run the **chronology check alongside it**: for every A EXTENDS B, assert `B.introduced_date < A.introduced_date`. The chronology check catches a bug class the DAG check cannot — a single wrong-direction edge between otherwise unrelated nodes is acyclic but chronologically impossible. Treat chronology violations as warnings requiring manual review rather than hard failures, to accommodate the rare concurrent-preprint case.

**6.4 `combined_techniques` array indexing.** In a property graph (Neo4j), a string-array property with a membership predicate (`WHERE 'quantization' IN m.combined_techniques`) is fine at this scale and indexable from Neo4j 5+. Two caveats: (a) if the target is RDF/SPARQL, array properties don't translate — you'd decompose to one triple per tag at export time; (b) if the query "methods that EXTEND LoRA and use quantization" ever becomes hot and the array predicate doesn't index well in your engine, promote the tags to nodes then — the §3.6 demotion is explicitly reversible, and doing it lazily beats doing it speculatively.

**6.5 Idempotent ingestion.** All node and edge writes should be upserts keyed on canonical IDs (Cypher `MERGE`, SQL `ON CONFLICT`), so re-running the pipeline against Semantic Scholar never duplicates nodes. This matters more than it sounds: the corpus will be fetched multiple times as the citation expansion is tuned toward the 70-paper target.

**6.6 Run the validation block as a suite, not a comment.** Every rule in the YAML `validation:` list should exist as an executable check (a Cypher query returning violating rows, or a Python assertion pass) run after every ingestion. A validation rule that lives only in the schema document is documentation; a validation rule that fails CI is a constraint.

