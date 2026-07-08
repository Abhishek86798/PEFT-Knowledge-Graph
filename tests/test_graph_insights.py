"""test_graph_insights.py -- pins the graph self-analysis detectors against the
shipped graph.json. Every assertion here is a claim about the *current* knowledge
state; if an edge changes, a test should change with it (that is the point --
the findings are falsifiable, not asserted).

Stdlib only. Run:  py tests/test_graph_insights.py  (or: py -m unittest discover -s tests)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import graph_insights as gi


class InsightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.g = gi.load()
        cls.r = gi.analyze(cls.g)

    # -- detector correctness (definitional invariants) ----------------------- #

    def test_ungrounded_comparisons_have_truly_disjoint_benchmarks(self):
        """Every reported ungrounded comparison must actually have zero benchmark
        overlap -- otherwise the detector is lying."""
        name, fam, bm, compared = gi._index(self.g)
        rev = {v: k for k, v in name.items()}
        for d in self.r["ungrounded_comparisons"]:
            f, t = rev[d["method"]], rev[d["baseline"]]
            self.assertEqual(bm[f] & bm[t], set(),
                             f"{d['method']} vs {d['baseline']} are reported as "
                             f"ungrounded but share a benchmark")

    def test_uncontested_methods_are_never_a_baseline(self):
        name, fam, bm, compared = gi._index(self.g)
        targets = {t for _, t, _ in compared}
        rev = {v: k for k, v in name.items()}
        for d in self.r["uncontested_methods"]:
            self.assertNotIn(rev[d["method"]], targets,
                             f"{d['method']} is reported uncontested but IS a baseline")

    def test_isolated_methods_share_no_benchmark(self):
        name, fam, bm, compared = gi._index(self.g)
        slugs = list(name)
        rev = {v: k for k, v in name.items()}
        for d in self.r["isolated_methods"]:
            s = rev[d["method"]]
            for o in slugs:
                if o != s:
                    self.assertEqual(bm[s] & bm[o], set(),
                                     f"{d['method']} reported isolated but shares a "
                                     f"benchmark with {name[o]}")

    # -- APPLIES coverage is quantified and consistent ------------------------ #

    def test_applies_coverage_totals_match_edges(self):
        ac = self.r["applies_coverage"]
        self.assertEqual(ac["total_applies_edges"], len(self.g["edges"]["APPLIES"]))
        self.assertEqual(sum(d["applies_papers"] for d in ac["per_method"]),
                         ac["total_applies_edges"])

    def test_applies_coverage_names_zero_methods(self):
        ac = self.r["applies_coverage"]
        # a real, disclosed property of the current corpus
        self.assertIn("(IA)^3", ac["methods_with_zero_coverage"])
        self.assertEqual(ac["most_covered"]["method"], "LoRA")

    # -- the (IA)^3 showpiece: three detectors converge ----------------------- #

    def test_ia3_is_flagged_by_all_three_detectors(self):
        """(IA)^3 is the strongest tension in the graph: isolated, uncontested,
        and every one of its comparisons is ungrounded. If curation ever grounds
        it, these assertions should be revisited -- deliberately."""
        methods_ungrounded = {d["method"] for d in self.r["ungrounded_comparisons"]}
        uncontested = {d["method"] for d in self.r["uncontested_methods"]}
        isolated = {d["method"] for d in self.r["isolated_methods"]}
        self.assertIn("(IA)^3", methods_ungrounded)
        self.assertIn("(IA)^3", uncontested)
        self.assertIn("(IA)^3", isolated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
