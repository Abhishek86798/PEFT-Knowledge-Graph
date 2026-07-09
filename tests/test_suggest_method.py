"""test_suggest_method.py -- pins the reasoning engine's behaviour on the input
classes that matter, so "operating on a new input" is demonstrably correct rather
than asserted.

Stdlib only (uses `unittest`); no network, no fixtures beyond the shipped
graph.json. Run:  py tests/test_suggest_method.py  (or: py -m unittest discover -s tests)

Each test is a *new input the graph has never seen* and encodes the intended
verdict for that class:
  - a near-duplicate of an existing variant   -> LIKELY_DUPLICATE
  - a genuinely novel mechanism                -> APPEARS_DISTINCT (novelty guard)
  - an orthogonal-technique combination        -> surfaces combined_techniques precedent
  - a family-root idea                          -> correct family + root placement
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# tests/ is a sibling of src/; add src/ to the path so `import suggest_method`
# resolves without installing the project as a package (stdlib-only, zero-dependency
# on purpose -- see approach.md).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import suggest_method as sm


class ReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gr = sm.load_graph()

    def analyze(self, text: str) -> dict:
        return sm.analyze(text, self.gr)

    # -- duplicate detection -------------------------------------------------- #

    def test_adalora_redescribed_is_flagged_duplicate(self):
        r = self.analyze("add a low-rank update to frozen weights, but choose the "
                         "rank per layer using SVD importance scores")
        self.assertEqual(r["closest_matches"][0]["name"], "AdaLoRA")
        self.assertEqual(r["novelty"]["status"], "LIKELY_DUPLICATE")
        # the verdict must name a concrete differentiator, not just flag
        self.assertIn("AdaLoRA", r["novelty"]["differentiator"])

    # -- novelty guard against pure lexical overlap --------------------------- #

    def test_lie_group_idea_reads_distinct_not_duplicate(self):
        """The guard case from approach.md: 'weight update delta' must NOT read
        as LoRA just because the words overlap."""
        r = self.analyze("represent each weight update as a learned rotation in a "
                         "Lie-group tangent space rather than an additive delta")
        self.assertEqual(r["novelty"]["status"], "APPEARS_DISTINCT")
        self.assertEqual(r["closest_matches"][0]["signature_hits"], 0)

    # -- orthogonal technique detection --------------------------------------- #

    def test_quantization_idea_surfaces_qlora_precedent(self):
        r = self.analyze("train low-rank adapters on top of a 4-bit quantized "
                         "frozen base model to save memory")
        techs = {t["technique"]: t for t in r["orthogonal_techniques"]}
        self.assertIn("quantization", techs)
        self.assertIn("QLoRA", techs["quantization"]["precedents"])

    def test_hypercomplex_idea_surfaces_compacter_precedent(self):
        r = self.analyze("build adapters whose weights are Kronecker products of "
                         "shared hypercomplex factors")
        techs = {t["technique"]: t for t in r["orthogonal_techniques"]}
        self.assertIn("hypercomplex-parameterization", techs)
        self.assertIn("Compacter", techs["hypercomplex-parameterization"]["precedents"])

    # -- structural placement uses the graph, not the query ------------------- #

    def test_bitfit_idea_placed_in_selective_family_root(self):
        r = self.analyze("train only the existing bias terms of the model and add "
                         "no new parameters")
        self.assertEqual(r["closest_matches"][0]["name"], "BitFit")
        fp = r["family_placement"]
        self.assertEqual(fp["family"], "selective")
        self.assertTrue(fp["match_is_root"])
        self.assertIn("BitFit", fp["family_roots"])

    def test_reading_order_is_root_first_and_ends_with_your_idea(self):
        r = self.analyze("add a low-rank update to frozen weights, but choose the "
                         "rank per layer using SVD importance scores")
        order = r["suggested_reading_order"]
        self.assertEqual(order[0], "LoRA")            # root first
        self.assertEqual(order[-1], "<your idea>")
        self.assertIn("AdaLoRA", order)

    # -- graceful degradation on zero-coverage methods ------------------------ #

    def test_zero_applies_match_reports_limitation_not_silence(self):
        """An (IA)^3-shaped idea (a method with zero APPLIES papers) must state
        the coverage limitation explicitly, not return an empty result that reads
        like a bug."""
        r = self.analyze("learn element-wise multiplicative rescaling vectors that "
                         "gate the keys values and ffn activations")
        at = r["already_tried_in_this_direction"]
        self.assertEqual(at["applies_papers_for_match"], 0)
        self.assertIn("UNAVAILABLE", at["applies_coverage_note"])
        # the curated-edge reasoning must be asserted as still valid
        self.assertIn("unaffected", at["applies_coverage_note"])

    def test_lora_match_reports_real_coverage(self):
        r = self.analyze("add a trainable low-rank decomposition B times A to frozen "
                         "weights, mergeable at inference")
        at = r["already_tried_in_this_direction"]
        self.assertEqual(r["closest_matches"][0]["name"], "LoRA")
        self.assertGreater(at["applies_papers_for_match"], 0)

    # -- mechanism contrast is grounded in the match's own signature terms ---- #

    def test_contrast_unmatched_are_real_signature_terms_of_match(self):
        r = self.analyze("add a low-rank update to frozen weights, but choose the "
                         "rank per layer using SVD importance scores")
        top_slug = r["closest_matches"][0]["slug"]
        match_sig = set(sm.MECHANISMS[top_slug]["signature"])
        for term in r["mechanism_contrast"]["unmatched"]:
            self.assertIn(term, match_sig,
                          f"'{term}' is reported as an unmatched signature term but "
                          f"is not in {top_slug}'s signature set")

    # -- Bug 1: family_roots must match the match's OWN EXTENDS ancestry,
    #    not every root that happens to share the family label ---------------- #

    def test_single_parent_match_reports_only_its_own_root(self):
        """Prompt-Tuning EXTENDS Prefix-Tuning only. The soft-prompt family also
        contains P-Tuning as a second, unrelated root -- family_roots must NOT
        list P-Tuning here, or it contradicts the (correct) reading order."""
        r = self.analyze("prepend trainable soft-prompt embeddings at the input "
                         "layer only, strict special case of prefix tuning")
        self.assertEqual(r["closest_matches"][0]["name"], "Prompt-Tuning")
        fp = r["family_placement"]
        self.assertEqual(fp["family_roots"], ["Prefix-Tuning"])
        self.assertNotIn("P-Tuning", fp["family_roots"])

    def test_family_roots_matches_reading_order_roots(self):
        """family_placement.family_roots and suggested_reading_order must never
        disagree about which root(s) the match descends from -- both are derived
        from the same EXTENDS ancestry walk."""
        for query, expected_name in [
            ("prepend trainable soft-prompt embeddings at the input layer only, "
             "strict special case of prefix tuning", "Prompt-Tuning"),
            ("add a low-rank update to frozen weights, but choose the rank per "
             "layer using SVD importance scores", "AdaLoRA"),
        ]:
            r = self.analyze(query)
            self.assertEqual(r["closest_matches"][0]["name"], expected_name)
            roots_in_reading_order = {
                name for name in r["suggested_reading_order"]
                if name != "<your idea>" and name != expected_name
            }
            self.assertEqual(set(r["family_placement"]["family_roots"]),
                             roots_in_reading_order,
                             f"family_roots disagrees with reading_order for {expected_name}")

    def test_dual_parent_match_reports_both_of_its_own_roots(self):
        """P-Tuning v2 genuinely EXTENDS both P-Tuning and Prefix-Tuning, so both
        must appear -- the fix must not collapse multi-parent ancestry to one root."""
        r = self.analyze("deep continuous prompts applied at every layer for NLU "
                         "across scales, merging a prompt encoder framing with "
                         "deep prefixes")
        self.assertEqual(r["closest_matches"][0]["name"], "P-Tuning v2")
        fp = r["family_placement"]
        self.assertEqual(set(fp["family_roots"]), {"P-Tuning", "Prefix-Tuning"})

    # -- Bug 2: NO_MATCH guard for zero-signal input --------------------------- #

    def test_gibberish_input_returns_no_match_not_a_false_positioning(self):
        """Zero score AND zero signature hits means retrieval found no evidence
        at all -- the tool must say so explicitly rather than confidently
        'positioning' the idea against an arbitrary tied-score method."""
        r = self.analyze("xyzzy quux plugh")
        self.assertEqual(r["status"], "NO_MATCH")
        self.assertIn("message", r)
        # must NOT continue into graph reasoning
        self.assertNotIn("family_placement", r)
        self.assertNotIn("novelty", r)
        self.assertNotIn("suggested_reading_order", r)
        self.assertNotIn("already_tried_in_this_direction", r)

    def test_no_match_still_reports_closest_matches_for_reference(self):
        r = self.analyze("xyzzy quux plugh")
        self.assertEqual(r["status"], "NO_MATCH")
        self.assertGreater(len(r["closest_matches"]), 0)
        self.assertEqual(r["closest_matches"][0]["score"], 0)
        self.assertEqual(r["closest_matches"][0]["signature_hits"], 0)

    def test_real_signal_queries_are_positioned_not_no_match(self):
        """Every query with nonzero score OR nonzero signature hits must take
        the normal reasoning path -- the guard must not over-trigger on weak
        but real signal (the Lie-group novelty-guard case in particular)."""
        for query in [
            "represent each weight update as a learned rotation in a Lie-group "
            "tangent space rather than an additive delta",  # score>0, sig=0
            "train only the existing bias terms of the model and add no new "
            "parameters",  # strong match
        ]:
            r = self.analyze(query)
            self.assertEqual(r["status"], "POSITIONED")
            self.assertIn("family_placement", r)
            self.assertIn("novelty", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
