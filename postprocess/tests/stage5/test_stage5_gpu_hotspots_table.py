import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

gpu_table = load_module_by_path("stage5_gpu_hotspots_table", "stage5", "stage5_gpu_hotspots_table.py")

from stage5_table_render import render_table  # noqa: E402  (needs sys.path insert above first)


class GpuHotspotsColumnsTests(unittest.TestCase):
    def test_includes_expected_columns(self):
        entries = [{"label": "k", "count": 2, "sum": 0.001, "avg_us": 500.0, "pct_total": 12.5}]
        table = render_table(gpu_table.GPU_HOTSPOTS_COLUMNS, entries)
        self.assertIn("%total", table)
        self.assertIn("avg(us)", table)
        self.assertIn("12.5", table)
        self.assertIn("500.00", table)

    def test_empty(self):
        self.assertIn("none found", render_table(gpu_table.GPU_HOTSPOTS_COLUMNS, []))


if __name__ == "__main__":
    unittest.main()
