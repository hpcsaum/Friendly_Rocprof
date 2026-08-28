"""Tests for stage1_rocprofsys_sample.py's timemory pipe-delimited text-table parser.

CleanLabelTests     -- clean_label() stripping rank prefix and hierarchy indent, independently
ParseTableFileTests -- parse_table_file() header detection, row extraction, non-table rejection
"""

import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

stage1 = load_module_by_path("stage1_rocprofsys_sample", "stage1", "stage1_rocprofsys_sample.py")


class CleanLabelTests(unittest.TestCase):
    # (rank prefix x indent) 2x2 matrix -- clean_label() must strip both independently of the other.
    CASES = [
        ("single_rank_no_indent", "00>>>main", "main"),
        ("single_rank_with_indent", "00>>>|_compute_stencil", "compute_stencil"),
        ("mpi_rank_prefix_no_indent", "00|00>>>main", "main"),
        ("mpi_rank_prefix_with_indent", "00|00>>>|_compute_stencil", "compute_stencil"),
    ]

    def test_strips_rank_prefix_and_indent(self):
        for name, raw, expected in self.CASES:
            with self.subTest(case=name):
                self.assertEqual(stage1.clean_label(raw), expected)


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
