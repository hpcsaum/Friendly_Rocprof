import importlib.util
import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "stage1_rocprofv3.py")

spec = importlib.util.spec_from_file_location("stage1_rocprofv3", MODULE_PATH)
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1_rocprofv3"] = stage1
spec.loader.exec_module(stage1)


class ParseKernelStatsCsvTests(unittest.TestCase):
    def test_single_rank_file(self):
        path = os.path.join(FIXTURES, "rocprofv3_single_rank", "myhost", "1234_kernel_stats.csv")
        rows = stage1.parse_kernel_stats_csv(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"JacobiIterationKernel", "BoundaryKernel", "__hipRegisterFatBinary"})
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["JacobiIterationKernel"]["count"], 1000)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["total_ns"], 537449866)

    def test_scientific_notation_value_parses(self):
        path = os.path.join(FIXTURES, "rocprofv3_single_rank", "myhost", "1234_kernel_stats.csv")
        rows = stage1.parse_kernel_stats_csv(path)
        by_label = {r["label"]: r for r in rows}
        # TotalDurationNs=9000 here is plain, but this row's Percentage/StdDev are in
        # scientific notation in the fixture -- confirms DictReader/our code don't choke on it.
        self.assertAlmostEqual(by_label["__hipRegisterFatBinary"]["total_ns"], 9000)

    def test_non_matching_csv_returns_none(self):
        path = os.path.join(FIXTURES, "rocprofv3_no_data", "myhost", "1_agent_info.csv")
        self.assertIsNone(stage1.parse_kernel_stats_csv(path))


if __name__ == "__main__":
    unittest.main()
