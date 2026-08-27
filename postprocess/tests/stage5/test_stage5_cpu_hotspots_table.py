import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

cpu_table = load_module_by_path("stage5_cpu_hotspots_table", "stage5", "stage5_cpu_hotspots_table.py")

from stage5_table_render import render_table  # noqa: E402  (needs sys.path insert above first)


class CpuHotspotsColumnsTests(unittest.TestCase):
    def test_includes_self_and_pct_total_columns(self):
        entries = [{"label": "a", "count": 1, "sum": 4.0, "self_sum": 1.0, "pct_self": 25.0, "pct_total": 25.0}]
        table = render_table(cpu_table.CPU_HOTSPOTS_COLUMNS, entries)
        self.assertIn("self(s)", table)
        self.assertIn("%total", table)
        self.assertIn("25.0", table)
        self.assertIn("1.000000", table)  # self_sum
        self.assertIn("4.000000", table)  # sum (total(s))

    def test_pct_total_none_renders_as_na(self):
        entries = [{"label": "a", "count": 1, "sum": 1.0, "self_sum": 0.1, "pct_self": 10.0, "pct_total": None}]
        table = render_table(cpu_table.CPU_HOTSPOTS_COLUMNS, entries)
        self.assertIn("n/a", table)

    def test_empty_entries(self):
        self.assertIn("none found", render_table(cpu_table.CPU_HOTSPOTS_COLUMNS, []))


if __name__ == "__main__":
    unittest.main()
