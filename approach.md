# Approach — how I built this, and why

This is a narrative of the decisions I made building a PEFT knowledge graph and a
reasoning engine over it, in roughly the order I actually made them. I've kept
the forks in — the alternatives I considered and rejected — because the final
schema looks tidier than the path to it, and the path is where the reasoning
lives.

---

## Where I started: what counts as a fact worth storing

The first real decision was what the graph is *for*. I could have built a
citation graph — Semantic Scholar hands you citations for free — but a citation
graph answers "who cited whom," and I wanted "what mechanism does what, and what
builds on what." So I committed early to a rule I kept coming back to: every node
and edge has to pass a falsifiable test, and if a candidate fact fails the test
it doesn't go in even if it feels true. The whole design followed from taking
that seriously.

That immediately ruled some things out. A `Model` entity for backbones (BERT, T5,
LLaMA) — I decided against it because most PEFT methods are evaluated across many
backbones, so those edges would be dense and carry almost no signal. Same for authors: they
answer sociology-of-science questions, not mechanism questions, and I'd already
decided the graph was about mechanism. I did keep backbone *around* as a filter
during edge construction (a COMPARED_AGAINST edge only counts if the two methods
were compared on the same backbone), but I discard it afterward rather than store
it. That "consulted transiently, not retained" distinction is one I made
deliberately, not by omission.

---

## The taxonomy, and the test that decided the hard cases

The core of the graph is 13 methods, and the hard part was deciding which are
*family roots* (a genuinely distinct way of injecting trainable parameters)
versus *variants* of an existing one. Determining this required examining each
method's description beyond the abstract — abstracts alone weren't enough for
consistent classification — so I wrote a single test and applied it everywhere:

> **Reduction test:** A is a variant of B if disabling or fixing A's novel
> components recovers B's trainable mechanism unchanged. If nothing reduces to an
> existing method, A is a root.

Three forks where I could easily have gone the other way:

**QLoRA — variant of LoRA, or its own root?** This one felt like it should be a
root. It's famous, it enabled 4-bit finetuning, it *feels* like a distinct thing.
But the reduction test doesn't care how it feels — it asks what the trainable
mechanism is. And QLoRA's trainable mechanism *is* unmodified LoRA: the adapters
are identical low-rank matrices. All of QLoRA's novelty (NF4, double
quantization, paged optimizers) lives in the *frozen* base, which isn't the part
being trained. So it fails the root test. The alternative — giving it its own
family — would have hidden the fact that mechanically it's LoRA. What I did
instead: `QLoRA --EXTENDS--> LoRA`, plus a `combined_techniques: [quantization]`
property to hold the orthogonal quantization novelty. The EXTENDS edge carries
the lineage, the property carries the thing that's genuinely new, and neither
one lies about the mechanism. I let the test override my intuition here, which is
the whole point of having a test.

**Prompt-Tuning — variant of Prefix-Tuning, when the authors treat them as
siblings?** Lester et al. frame prompt-tuning as a simpler *peer* of
prefix-tuning, not a descendant, and the field mostly repeats that framing. I
went against it. Mechanically, input-layer-only soft prompts are a strict special
case of deep every-layer prefixes — fix prefix-tuning to only touch the input
layer and you *have* prompt-tuning. That's exactly the reduction test passing. I
decided mechanism-subsumption beats authorial self-positioning: how a paper
frames its own contribution is rhetoric, and I didn't want rhetoric deciding
graph structure. I flagged this one in the taxonomy itself because it's the call
a reviewer is most likely to challenge, and I'd rather meet the objection in the
document than in the viva.

**P-Tuning — root, or variant of Prefix-Tuning too?** I made this one a root even
though it's also a soft-prompt method, because its mechanism (an LSTM/MLP prompt
*encoder* generating the continuous prompts) doesn't reduce to prefix-tuning —
it's an independent, contemporaneous idea. So "soft-prompt family" ended up with
two roots (Prefix-Tuning and P-Tuning), which I was fine with: family is about
mechanism, and there genuinely are two mechanisms here.

---

## The mistake that reshaped the schema: how papers get in

Here's the decision I got wrong first and had to fix, and it's the most important
one in the build.

My initial model let papers into the graph only two ways: a paper either
INTRODUCES a method, or it provides the results-table evidence behind a
COMPARED_AGAINST / EVALUATED_ON edge. Clean, strict, every paper tied to a
mechanism claim. Then I did the arithmetic against the target corpus size of
60–80 papers and realized the problem: there are only ~13 method-introducing
papers in this space. Under my model, the corpus **capped at ~13** and there was
no honest way to reach 70. I'd built a schema that structurally couldn't hit its
own target.

The fix was a third way in: **APPLIES** (Paper→Method) — a paper that *runs* a
tracked method unmodified and reports a result. This is what carries the corpus
from 13 to 68: domain applications (medical, malware, speech), systems papers
(mLoRA, EdgeLoRA), comparative re-benchmarking studies. It also answers a
question none of my other edges could — *who actually uses each method in
practice* — which in hindsight is a genuinely useful thing to have, but I want to
be honest that I added it to fix a corpus-size problem, not because I foresaw its
value.

The word that does the work in APPLIES is "unmodified," and I chose it precisely.
If a paper *modifies* the mechanism, it fails APPLIES and becomes a candidate new
Method instead — which means APPLIES and EXTENDS can never describe the same
relationship. That disjointness is what keeps the variant-proposals
(Delta-LoRA, Echo-LoRA, and the rest) out of the APPLIES set: they propose new
mechanisms, so they aren't "applications" of LoRA.

---

## Shaping COMPARED_AGAINST: the semantically-correct option I rejected

When I modeled COMPARED_AGAINST, there was a real fork. The honest reading is
that comparisons happen *inside papers* — it was the LoRA *paper* that ran the
comparison, not LoRA-the-method — so the "correct" shape is
`Paper → COMPARED_AGAINST → Method`. I didn't use it.

The reason is the query the graph exists to answer: "what were LoRA's baselines?"
Under the Paper-anchored shape that becomes a two-hop join, and you can only
recover the *proposing* method by leaning on INTRODUCES being exactly-one — which
is an invariant of a different edge, a fragile thing to depend on. So I kept it
`Method → Method` and pushed the provenance into an `evidence_paper` property on
the edge. That's exactly as expressive as the Paper-anchored version — the paper
is still recorded — but it keeps the common query at one hop. I traded a bit of
semantic purity for a cheaper query, on purpose, because the query is the thing
the graph is *for*.

I also made COMPARED_AGAINST strict in a way I could have loosened: it only
counts if the baseline appears as a **row in a main results table**, not in
related work or discussion. That strictness is what stops it from collapsing back
into a citation graph — "mentioned somewhere" is exactly what I was trying not to
build.

That strictness caught a real error when I verified it. I went back through all 13
introducing papers and checked every COMPARED_AGAINST and EVALUATED_ON edge against
the actual results table it claims. One failed: I had recorded Prompt-Tuning as
COMPARED_AGAINST Prefix-Tuning, but Lester et al. only discuss prefix-tuning in
prose (Section 4) and a parameter-count figure — there is no quantitative baseline
row, so the edge fails my own test. I removed it (COMPARED_AGAINST went from 30 to
29). The other edges held. A separate, milder point surfaced in the same pass:
three EVALUATED_ON edges are backed by a *figure* rather than a table (Adapters on
SQuAD, Prompt-Tuning on SuperGLUE, (IA)³ on the T0 held-out tasks). My test says
"main results table"; I read that as "main results presentation" and kept them,
because dropping real main-body quantitative results on a figure-vs-table
technicality would make the graph less accurate. Those three are the edges most
open to challenge, and I'd rather name them than bury them.

---

## The edge I demoted: combined_techniques

I first modeled the quantization-in-QLoRA idea as a `COMBINES: Method → Technique`
edge, pointing at a `Technique` entity. Then I noticed something that bothered me:
it was the *only* edge in the whole schema whose target wasn't a first-class
entity with its own inclusion test — `Technique` existed only to be pointed at.
And only two methods used it (QLoRA's quantization, Compacter's hypercomplex
parameterization). So I demoted it to a plain list property on Method. If that
controlled vocabulary ever grows past a handful of tags I'd promote it back to a
real entity — I left that door open deliberately — but building a `Technique`
node type for two uses was ceremony I couldn't justify.

---

## Building the corpus, and the fetch rewrite I didn't plan for

The taxonomy, the EXTENDS edges, and the 13 papers' COMPARED_AGAINST /
EVALUATED_ON edges I verified by consulting the original papers — focusing on the
method descriptions and primary evaluation tables — because those edges need more
than an abstract to get right, and I didn't trust abstract-level tagging for them.
The ~55 APPLIES papers were the other tier: tractable to tag from abstracts, so I
fetched a candidate pool and tagged at scale. That asymmetry — evidence-heavy
edges checked against the papers, higher-volume edges tagged from abstracts — is
something I planned for.

The fetch itself I did *not* plan well, and had to rewrite mid-build. My first
`fetch_papers.py` pulled candidate details one paper at a time, one GET each.
Against the Semantic Scholar public API that fell apart: the log was wall-to-wall
HTTP 429, seed *search* was the worst-throttled path and failed to resolve LoRA
at all, and a single run ground for ~50 minutes without finishing. The 429 log is
what forced the rewrite, not foresight. Two changes fixed it: I switched candidate
fetching to the **batch endpoint** — one POST of up to 500 ids instead of 500
GETs — and I stopped using search entirely for seeds, resolving them directly by
arXiv id (`GET /paper/ARXIV:{id}`), which sidesteps the throttled search path.
That took the candidate phase from "hundreds of failing requests" to a single
call. I'd have designed it that way from the start if I'd known; I didn't, and the
rate-limit log taught me.

---

## The judgment calls with no clean answer

Two places where I want to be explicit that there *wasn't* a right answer, only a
call I made:

**SiRA and ReMix — APPLIES or not?** Both wrap LoRA in a Mixture-of-Experts and
give the result a new name ("Sparse Mixture of Low Rank Adaptation," RL-routed
mixtures). The tension: LoRA's low-rank update is run unmodified *inside* them, so
by a literal reading they "apply" LoRA — but each names itself a new method and
pitches that as its contribution, so the LoRA mechanism isn't what's being tested
in isolation. I first accepted them, then flipped both to reject. What tipped it:
consistency. I'd already rejected Echo-LoRA and Zipper-LoRA for being self-named
variants, and if I accepted SiRA I'd be answering the same "is composition a
modification?" question two different ways in the same graph. A reviewer checking
my work would find that inconsistency faster than anything else, so I made the
rule uniform — self-named composition = variant = reject. When I then reviewed the
rest of the borderline calls against that same rule, six more fell to it: ComPEFT
(compresses a trained LoRA vector post-hoc — tooling, not a run of LoRA), a
LoRA-MoE pruning framework (DMEP), a sparsity-crafting method (PESC), a
stochastic-gate method benchmarked *against* LoRA (FineGates), a weight-conditioning
pair (Pre-Diag/SORA), and an adapter-placement paper that actually introduces
long-range/recurrent adapter connections. Each names a new method, so each is a
variant, not an application. That review took the corpus from 74 to 68, still
inside the 60–80 band. I also flipped one the *other* direction ("Vanilla LoRA May
Suffice," which I'd wrongly rejected — it runs LoRA unmodified to make an empirical
point, which is textbook APPLIES). I logged a `reason`, `confidence`, and
`evidence` for every one of the 200 calls so this is auditable rather than my
say-so.

**Figures that aren't tables.** My COMPARED_AGAINST / EVALUATED_ON test says
"main results *table*." But several of these papers put their main results in a
*figure*: Prompt-Tuning's SuperGLUE comparison is Figure 1, Adapters' SQuAD result
is Figure 5, and (IA)³'s main PEFT comparison is Figure 2 with the per-dataset
numbers pushed to an appendix. A strict literal reading of my own test would
exclude those. I decided to read "main results table" as "main results
*presentation*," because the alternative — dropping real, main-body, quantitative
comparisons on a technicality about figure-vs-table — would make the graph less
accurate, not more. This is a genuine stretch of my own rule and I'd rather name
it than hide it; the (IA)³ edges (figure + appendix) are the ones most open to
challenge.

---

## How it reasons on something new

The graph would just be a database if it only answered questions about what's in
it, so the payoff is `suggest_method.py`: paste a free-text description of a new
PEFT idea and it positions the idea against the graph — closest existing
method(s) with the matched terms as justification, a reading order walked
backward along EXTENDS to the family root (`Adapters → Compacter → your idea`),
what's already been tried in that direction (EXTENDS children, siblings, the
match's baselines), and a novelty flag.

**The lexical match is only the entry point; the reasoning is graph traversal.**
Once a query lands on its closest method node, the system stops matching text and
starts walking the graph: it traverses EXTENDS *backward* to build the reading
order (root → … → match → your idea), and pulls the matched method's
COMPARED_AGAINST neighbours out of the graph to report what has already been tried
in that direction. The novelty verdict combines the lexical signal with a
structural check — whether the idea reduces to an existing mechanism or sits
outside every family. So the output is grounded in the graph's edges, not in the
text of the query.

The one decision worth explaining here is that matching is **lexical, not
neural** — weighted term overlap, no embeddings, no API. I chose that for
determinism and zero dependencies, but it has a real failure mode: generic shared
words like "weight" and "update" can make an unrelated idea look like a match. I
hit exactly this — a genuinely novel "Lie-group geodesic" idea scored as LoRA
purely on the words "weight update delta." So I split each method's vocabulary
into *signature terms* (diagnostic of the mechanism: "kronecker", "low-rank",
"bias") versus prose, and made the duplicate flag require **≥2 signature hits**,
not just a high score. After that, the Lie-group idea correctly reads
APPEARS_DISTINCT and the message tells the user outright that matching is lexical
and to confirm by reading the paper. I'd rather the tool under-claim novelty-death
than cry "duplicate" on every idea that happens to say "weight."

I put the mechanism descriptions and signature terms *in* `graph.json` on each
Method node, not hardcoded in the script, so the taxonomy is fully inspectable
without running code and there's one source of truth. That was a late change,
after I realized a reviewer opening the JSON should be able to see *why* each
method is classified the way it is, not just its name.

---

## What I'd build next

The thing I'd fix first is the lopsidedness of my APPLIES coverage. Of the 72
APPLIES edges, LoRA has 50 — and **8 of my 13 methods have zero**: Prefix-Tuning,
P-Tuning, P-Tuning v2, Compacter, Pfeiffer Adapters, AdaLoRA, VeRA, and (IA)³ are
all in the graph as methods but no application paper in my corpus runs them. That
happened because I expanded the candidate pool from citation/reference links off
7 seeds, and those seeds were LoRA-heavy — so the "who uses this in practice"
signal is real for LoRA and nearly empty for everything else. The graph currently
over-tells the LoRA story.

The fix is targeted, not just "fetch more": seed the APPLIES expansion from *each*
method rather than from a LoRA-dominated set, so a query like "who actually uses
BitFit / (IA)³ in practice" returns something. That's more valuable than
polishing the matcher, because the reasoning engine's "already tried in this
direction" output is only as good as the APPLIES coverage behind it — right now it
can answer that well for LoRA-family ideas and poorly for the soft-prompt or
multiplicative families. Broadening coverage per-method is what makes the tool
answer evenly across the taxonomy instead of just where the citations happened to
cluster.

Two smaller things I'd clean up after that. The 55 APPLIES papers are keyed by
Semantic Scholar hash ids where no arXiv id existed (many are from venues like
VLDB, not arXiv), so a pass could recover true arXiv ids for the arXiv-sourced
ones. And `combined_techniques` is built to be promoted back to a `Technique`
entity the moment the vocabulary outgrows its two current tags — that's the first
schema change I'd make if the corpus expanded.
