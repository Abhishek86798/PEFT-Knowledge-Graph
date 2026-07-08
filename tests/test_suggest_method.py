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


if __name__ == "__main__":
    unittest.main(verbosity=2)
