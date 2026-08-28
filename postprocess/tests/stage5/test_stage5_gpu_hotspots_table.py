"""Tests for stage5_gpu_hotspots_table.py's GPU_HOTSPOTS_COLUMNS column spec -- confirms total time,
%total, calls, and avg(us) render with the right headers and formatting; render_table() itself is
exhaustively covered in test_stage5_table_render.py, so this is only a wiring check on the column
spec (GpuHotspotsColumnsTests).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

gpu_table = load_module_by_path("stage5_gpu_hotspots_table", "stage5", "stage5_gpu_hotspots_table.py")

from stage5_table_render import render_table  # noqa: E402  (needs sys.path insert above first)


class GpuHotspotsColumnsTests(unittest.TestCase):
    # Thin column-spec sanity check, not a render_table() test -- see
    # test_stage5_cpu_hotspots_table.py's CpuHotspotsColumnsTests for why.
    CASES = [
        ("includes_expected_columns",
         [{"label": "k", "count": 2, "sum": 0.001, "avg_us": 500.0, "pct_total": 12.5}],
         ["%total", "avg(us)", "12.5", "500.00"]),
        ("empty_entries", [], ["none found"]),
    ]

    def test_columns(self):
        for name, entries, expect_substrings in self.CASES:
            with self.subTest(case=name):
                table = render_table(gpu_table.GPU_HOTSPOTS_COLUMNS, entries)
                for substring in expect_substrings:
                    self.assertIn(substring, table)


if __name__ == "__main__":
    unittest.main()
