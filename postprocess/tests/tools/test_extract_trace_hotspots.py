import importlib.util
import glob
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "tools", "extract_trace_hotspots.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("extract_trace_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_trace_hotspots"] = hotspots
spec.loader.exec_module(hotspots)

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")


def _clear_cache():
    for f in glob.glob(os.path.join(TWO_RANK_DIR, "*.agg.json")):
        os.remove(f)


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        _clear_cache()

    def test_end_to_end_fused_table_and_load_imbalance(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(TWO_RANK_DIR, dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("jacobi_sweep", report)
            self.assertIn("jacobi_kernel.kd", report)  # GPU domain row present in the fused table
            self.assertIn("Load imbalance across 2 ranks", report)
            self.assertIn("MPI ranks: 2", report)

    def test_show_all_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(TWO_RANK_DIR, dest, show_all=True)
            self.assertIn("showing all", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        _clear_cache()

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            hotspots.main([TWO_RANK_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("jacobi_sweep", report)


if __name__ == "__main__":
    unittest.main()
