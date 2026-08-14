import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location(
    "stage5_gpu_hotspots_table", os.path.join(POSTPROCESS_DIR, "stage5_gpu_hotspots_table.py")
)
gpu_table = importlib.util.module_from_spec(spec)
sys.modules["stage5_gpu_hotspots_table"] = gpu_table
spec.loader.exec_module(gpu_table)

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
