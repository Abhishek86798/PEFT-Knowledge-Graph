"""test_explain_edge.py -- verifies "why does this edge exist" resolves the right
justification for every edge type in the schema, and fails cleanly when it can't.

Stdlib only. Run:  py tests/test_explain_edge.py  (or: py -m unittest discover -s tests)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import explain_edge as ee


class ExplainEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.g = ee.load_graph()
        cls.review = ee.load_applies_review()

    def explain(self, edge, frm, to):
        return ee.explain(self.g, self.review, edge, frm, to)

    # -- EXTENDS: reduction-test note lives directly on the edge -------------- #

    def test_extends_returns_reduction_test_note(self):
        r = self.explain("EXTENDS", "AdaLoRA", "LoRA")
        self.assertTrue(r["found"])
        self.assertIn("reduction test", r["justification"].lower())
        self.assertIn("Reduction test", r["test_applied"])

    def test_extends_dual_parent_both_directions_resolve(self):
        """P-Tuning v2 has two EXTENDS parents; both must be independently explainable."""
        r1 = self.explain("EXTENDS", "P-Tuning v2", "P-Tuning")
        r2 = self.explain("EXTENDS", "P-Tuning v2", "Prefix-Tuning")
        self.assertTrue(r1["found"])
        self.assertTrue(r2["found"])

    def test_extends_wrong_direction_not_found(self):
        """EXTENDS is directional: LoRA does not extend AdaLoRA."""
        r = self.explain("EXTENDS", "LoRA", "AdaLoRA")
        self.assertFalse(r["found"])

    # -- COMPARED_AGAINST: note + evidence_paper + untracked baselines -------- #

    def test_compared_against_returns_evidence_paper_and_note(self):
        r = self.explain("COMPARED_AGAINST", "AdaLoRA", "LoRA")
        self.assertTrue(r["found"])
        self.assertIsNotNone(r["evidence_paper"])
        self.assertIn("main results table", r["test_applied"])

    def test_compared_against_nonexistent_pair_not_found(self):
        r = self.explain("COMPARED_AGAINST", "BitFit", "QLoRA")
        self.assertFalse(r["found"])
        self.assertIn("no COMPARED_AGAINST edge", r["error"])

    # -- EVALUATED_ON: evidence_paper, resolves benchmark by name or slug ----- #

    def test_evaluated_on_resolves_benchmark_by_display_name(self):
        r = self.explain("EVALUATED_ON", "LoRA", "GLUE")
        self.assertTrue(r["found"])
        self.assertIsNotNone(r["evidence_paper"])

    def test_evaluated_on_resolves_benchmark_by_slug(self):
        r = self.explain("EVALUATED_ON", "LoRA", "glue")
        self.assertTrue(r["found"])

    def test_evaluated_on_missing_benchmark_not_found(self):
        r = self.explain("EVALUATED_ON", "BitFit", "mmlu")  # BitFit only reports GLUE
        self.assertFalse(r["found"])

    # -- INTRODUCES: definitional, cardinality-backed ------------------------- #

    def test_introduces_returns_definitional_justification(self):
        r = self.explain("INTRODUCES", "1902.00751", "Adapters")
        self.assertTrue(r["found"])
        self.assertIn("Definitional", r["justification"])

    def test_introduces_wrong_paper_not_found(self):
        r = self.explain("INTRODUCES", "0000.00000", "Adapters")
        self.assertFalse(r["found"])

    # -- APPLIES: reason/confidence/evidence pulled from the review worksheet - #

    def test_applies_returns_reason_confidence_evidence(self):
        r = self.explain("APPLIES", "9aef980ac6e6a1cbc470362a042b75cfb50e2e48", "LoRA")
        self.assertTrue(r["found"])
        self.assertIsNotNone(r.get("reason"))
        self.assertIsNotNone(r.get("confidence"))
        self.assertIn("Vanilla LoRA", r["reason"])

    def test_applies_reason_is_verbatim_from_review_worksheet(self):
        """The explanation must not paraphrase -- it must be the literal recorded
        reason, so the audit trail is exact."""
        paper_id = "9aef980ac6e6a1cbc470362a042b75cfb50e2e48"
        entry = next(e for e in self.review
                     if e.get("paper_id") == paper_id or e.get("arxiv_id") == paper_id)
        r = self.explain("APPLIES", paper_id, "LoRA")
        self.assertEqual(r["reason"], entry["reason"])
        self.assertEqual(r["confidence"], entry["confidence"])

    def test_applies_nonexistent_paper_not_found(self):
        r = self.explain("APPLIES", "not-a-real-paper-id", "LoRA")
        self.assertFalse(r["found"])

    # -- method-name resolution is case-insensitive and slug-or-name agnostic - #

    def test_method_resolution_accepts_slug_or_display_name(self):
        r_name = self.explain("EXTENDS", "AdaLoRA", "LoRA")
        r_slug = self.explain("EXTENDS", "method_adalora", "method_lora")
        self.assertTrue(r_name["found"])
        self.assertTrue(r_slug["found"])
        self.assertEqual(r_name["justification"], r_slug["justification"])

    def test_method_resolution_is_case_insensitive(self):
        r = self.explain("EXTENDS", "adalora", "lora")
        self.assertTrue(r["found"])

    def test_unknown_method_reports_clean_error(self):
        r = self.explain("EXTENDS", "NotARealMethod", "LoRA")
        self.assertFalse(r["found"])
        self.assertIn("unknown method", r["error"])

    def test_unknown_edge_type_reports_clean_error(self):
        r = self.explain("NOT_A_REAL_EDGE", "LoRA", "AdaLoRA")
        self.assertFalse(r["found"])
        self.assertIn("unknown edge type", r["error"])

    # -- --list-edges discovery helper ---------------------------------------- #

    def test_list_edges_counts_match_graph(self):
        edges = ee.list_edges_for_method(self.g, "method_lora")
        applies_total = sum(1 for e in self.g["edges"]["APPLIES"] if e["to_method"] == "method_lora")
        extends_total = sum(1 for e in self.g["edges"]["EXTENDS"]
                            if e["from_method"] == "method_lora" or e["to_method"] == "method_lora")
        self.assertEqual(len(edges["APPLIES"]), applies_total)
        self.assertEqual(len(edges["EXTENDS"]), extends_total)
        self.assertEqual(len(edges["INTRODUCES"]), 1)  # exactly one, per schema invariant

    def test_list_edges_unknown_method_returns_none(self):
        name_map = {m["slug"]: m["name"] for m in self.g["methods"]}
        slug_map = {v: k for k, v in name_map.items()}
        resolved = ee._resolve_method("TotallyMadeUp", slug_map, name_map)
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
