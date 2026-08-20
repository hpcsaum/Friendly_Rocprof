import importlib.util
import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "stage1", "stage1_rocprofsys_sample.py")

spec = importlib.util.spec_from_file_location("stage1_rocprofsys_sample", MODULE_PATH)
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1_rocprofsys_sample"] = stage1
spec.loader.exec_module(stage1)


class CleanLabelTests(unittest.TestCase):
    def test_single_rank_no_indent(self):
        self.assertEqual(stage1.clean_label("00>>>main"), "main")

    def test_single_rank_with_indent(self):
        self.assertEqual(stage1.clean_label("00>>>|_compute_stencil"), "compute_stencil")

    def test_mpi_rank_prefix_no_indent(self):
        self.assertEqual(stage1.clean_label("00|00>>>main"), "main")

    def test_mpi_rank_prefix_with_indent(self):
        self.assertEqual(stage1.clean_label("00|00>>>|_compute_stencil"), "compute_stencil")


class ParseTableFileTests(unittest.TestCase):
    def test_flat_single_rank_file(self):
        path = os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")
        rows = stage1.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "apply_boundary", "hipLaunchKernel", "hipMemcpy"})
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 9.812345)

    def test_hierarchical_mpi_rank_file(self):
        path = os.path.join(FIXTURES, "mpi_2rank", "wall_clock-2001.txt")
        rows = stage1.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "hipMemcpy"})

    def test_non_timing_file_returns_none(self):
        path = os.path.join(FIXTURES, "no_timing_data", "available.txt")
        self.assertIsNone(stage1.parse_table_file(path))


if __name__ == "__main__":
    unittest.main()
