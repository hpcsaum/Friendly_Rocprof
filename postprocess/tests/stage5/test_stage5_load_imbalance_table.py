import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

imb = load_module_by_path("stage5_load_imbalance_table", "stage5", "stage5_load_imbalance_table.py")

from stage5_table_render import render_table  # noqa: E402  (needs sys.path insert above first)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

flat = load_module_by_path("stage4_rocprofsys_sample_flat", "stage4", "stage4_rocprofsys_sample_flat.py")

v3 = load_module_by_path("stage4_rocprofv3", "stage4", "stage4_rocprofv3.py")


class ComputeLoadImbalanceTests(unittest.TestCase):
    def test_matches_independently_computed_statistics_on_real_fixture(self):
        per_file, _ = flat.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"))
        selected, _ = imb.compute_load_imbalance(per_file, show_all=True)
        by_label = {e["label"]: e for e in selected}

        cs_values = [ft["compute_stencil"] for ft in per_file]
        self.assertAlmostEqual(by_label["compute_stencil"]["avg"], statistics.mean(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["std_dev"], statistics.pstdev(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["min"], min(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["max"], max(cs_values))

        main_values = [ft["main"] for ft in per_file]
        self.assertAlmostEqual(by_label["main"]["std_dev"], statistics.pstdev(main_values))

        # compute_stencil varies more (9.4 vs 9.5) than main (10.9 vs 10.924161)
        # in absolute terms -- confirm it sorts first when ranked by std_dev.
        self.assertGreater(by_label["compute_stencil"]["std_dev"], by_label["main"]["std_dev"])
        sorted_labels = [e["label"] for e in selected]
        self.assertEqual(sorted_labels.index("compute_stencil"), 0)

    def test_missing_rank_scores_zero_not_omitted(self):
        per_file_totals = [{"only_on_rank0": 10.0}, {}]
        selected, _ = imb.compute_load_imbalance(per_file_totals, show_all=True)
        entry = next(e for e in selected if e["label"] == "only_on_rank0")
        self.assertAlmostEqual(entry["avg"], 5.0)
        self.assertAlmostEqual(entry["std_dev"], 5.0)
        self.assertAlmostEqual(entry["min"], 0.0)
        self.assertAlmostEqual(entry["max"], 10.0)

    def test_top_n_selects_highest_std_dev(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        selected, desc = imb.compute_load_imbalance(per_file_totals, top=1)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("top 1 of 3", desc)

    def test_threshold_is_coefficient_of_variation(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        # b: avg=10, std_dev=5 -> cv=50%; a and c: std_dev=0 -> cv=0%
        selected, desc = imb.compute_load_imbalance(per_file_totals, threshold=10.0)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("coefficient of variation", desc)

    def test_show_all(self):
        per_file_totals = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
        selected, desc = imb.compute_load_imbalance(per_file_totals, show_all=True)
        self.assertEqual(len(selected), 1)
        self.assertIn("all 1 entries", desc)

    def test_tie_break_is_deterministic_by_label(self):
        # Two labels with identical avg/std_dev/min/max -- output order must not depend on
        # dict/set iteration order (the original nondeterministic-tie-order bug).
        per_file_totals = [{"zeta": 5.0, "alpha": 5.0}, {"zeta": 5.0, "alpha": 5.0}]
        selected, _ = imb.compute_load_imbalance(per_file_totals, show_all=True)
        self.assertEqual([e["label"] for e in selected], ["alpha", "zeta"])

    def test_matches_independently_computed_statistics_on_real_gpu_fixture(self):
        # Same shared compute_load_imbalance() used by both CPU and GPU tools -- exercised
        # here against a real rocprofv3 per-rank fixture.
        per_file, _ = v3.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        selected, _ = imb.compute_load_imbalance(per_file, show_all=True)
        by_label = {e["label"]: e for e in selected}

        jacobi_values = [ft["JacobiIterationKernel"] for ft in per_file]
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg"], statistics.mean(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["std_dev"], statistics.pstdev(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["min"], min(jacobi_values))
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["max"], max(jacobi_values))


class LoadImbalanceColumnsTests(unittest.TestCase):
    def test_default_item_label_is_function(self):
        entries = [{"label": "foo", "avg": 1.0, "std_dev": 0.5, "min": 0.5, "max": 1.5, "cv_pct": 50.0}]
        table = render_table(imb.load_imbalance_columns(), entries)
        self.assertIn("avg(s)", table)
        self.assertIn("std_dev", table)
        self.assertIn("min(s)", table)
        self.assertIn("max(s)", table)
        self.assertIn("function", table)
        self.assertIn("foo", table)

    def test_item_label_override_for_gpu(self):
        table = render_table(imb.load_imbalance_columns(item_label="kernel"), [])
        # empty entries short-circuits to "(none found)" before the header would show --
        # confirm the header text itself with one real entry instead.
        entries = [{"label": "my_kernel", "avg": 1.0, "std_dev": 0.5, "min": 0.5, "max": 1.5, "cv_pct": 50.0}]
        table = render_table(imb.load_imbalance_columns(item_label="kernel"), entries)
        self.assertIn("kernel", table)

    def test_empty_entries(self):
        self.assertIn("none found", render_table(imb.load_imbalance_columns(), []))


class ImbalanceNoteTests(unittest.TestCase):
    def test_function_self(self):
        note = imb.imbalance_note("function", "self")
        self.assertEqual(
            note,
            "  - Each function's own self time on each rank, compared across ranks -- a rank "
            "that never called a function counts as 0.0 for that rank, not omitted.\n",
        )

    def test_function_inclusive(self):
        note = imb.imbalance_note("function", "inclusive")
        self.assertIn("own inclusive time", note)

    def test_kernel_total(self):
        note = imb.imbalance_note("kernel", "total")
        self.assertIn("Each kernel's own total time", note)
        self.assertIn("never launched a kernel", note)


if __name__ == "__main__":
    unittest.main()
