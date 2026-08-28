"""Tests for stage5_cpu_hotspots_table.py's CPU_HOTSPOTS_COLUMNS column spec -- confirms self/total
time, %total, %self, and calls all render with the right headers and formatting; render_table()
itself is exhaustively covered in test_stage5_table_render.py, so this is only a wiring check on the
column spec (CpuHotspotsColumnsTests).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

cpu_table = load_module_by_path("stage5_cpu_hotspots_table", "stage5", "stage5_cpu_hotspots_table.py")

from stage5_table_render import render_table  # noqa: E402  (needs sys.path insert above first)


class CpuHotspotsColumnsTests(unittest.TestCase):
    # (case, entries, substrings render_table(CPU_HOTSPOTS_COLUMNS, entries) must include) --
    # this is a thin column-spec sanity check, not a render_table() test (that's exhaustively
    # covered in test_stage5_table_render.py); it only confirms CPU_HOTSPOTS_COLUMNS itself has
    # the right headers/keys/formatting wired up.
    CASES = [
        ("includes_self_and_pct_total_columns",
         [{"label": "a", "count": 1, "sum": 4.0, "self_sum": 1.0, "pct_self": 25.0, "pct_total": 25.0}],
         ["self(s)", "%total", "25.0", "1.000000", "4.000000"]),  # self_sum, sum (total(s))
        ("pct_total_none_renders_as_na",
         [{"label": "a", "count": 1, "sum": 1.0, "self_sum": 0.1, "pct_self": 10.0, "pct_total": None}],
         ["n/a"]),
        ("empty_entries", [], ["none found"]),
    ]

    def test_columns(self):
        for name, entries, expect_substrings in self.CASES:
            with self.subTest(case=name):
                table = render_table(cpu_table.CPU_HOTSPOTS_COLUMNS, entries)
                for substring in expect_substrings:
                    self.assertIn(substring, table)


if __name__ == "__main__":
    unittest.main()
