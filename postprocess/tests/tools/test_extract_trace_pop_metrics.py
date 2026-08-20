import importlib.util
import glob
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "tools", "extract_trace_pop_metrics.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("extract_trace_pop_metrics", MODULE_PATH)
pop_metrics = importlib.util.module_from_spec(spec)
sys.modules["extract_trace_pop_metrics"] = pop_metrics
spec.loader.exec_module(pop_metrics)

TWO_RANK_DIR = os.path.join(FIXTURES, "trace_cli_two_rank")


def _clear_cache():
    for f in glob.glob(os.path.join(TWO_RANK_DIR, "*.agg.json")):
        os.remove(f)


class WriteReportTests(unittest.TestCase):
    def tearDown(self):
        _clear_cache()

    def test_single_run_reports_gpu_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_metrics.write_report([TWO_RANK_DIR], dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("GPU-Util", report)  # GPU visibility is always part of the trace
            self.assertIn("pool: CPU+GPU, single unified trace source", report)
            self.assertIn("MPI ranks: 2", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        _clear_cache()

    def test_main_requires_scaling_flag_with_scaled_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            with self.assertRaises(SystemExit):
                pop_metrics.main([TWO_RANK_DIR, TWO_RANK_DIR, "-o", dest])

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            pop_metrics.main([TWO_RANK_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("=== Metrics ===", report)


if __name__ == "__main__":
    unittest.main()
