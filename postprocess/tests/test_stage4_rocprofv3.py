import importlib.util
import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4_rocprofv3.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("stage4_rocprofv3", MODULE_PATH)
v3 = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofv3"] = v3
spec.loader.exec_module(v3)


class AggregateTests(unittest.TestCase):
    def test_single_rank(self):
        entries, scanned, total_ns = v3.aggregate(os.path.join(FIXTURES, "rocprofv3_single_rank"))
        self.assertEqual(len(scanned), 1)
        by_label = {e["label"]: e for e in entries}
        expected_total_ns = 537449866 + 58000000 + 9000
        self.assertAlmostEqual(total_ns, expected_total_ns)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["sum"], 537449866 / 1e9)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["pct_total"], 537449866 / expected_total_ns * 100)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg_us"], 537449866 / 1000 / 1000)

    def test_mpi_ranks_aggregate_by_kernel_name(self):
        entries, scanned, total_ns = v3.aggregate(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in entries}
        jacobi_ns = 268724933 + 268000000
        boundary_ns = 16000000 + 17000000
        expected_total_ns = jacobi_ns + boundary_ns
        self.assertAlmostEqual(total_ns, expected_total_ns)
        self.assertEqual(by_label["JacobiIterationKernel"]["count"], 1000)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["sum"], jacobi_ns / 1e9)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["pct_total"], jacobi_ns / expected_total_ns * 100)
        self.assertAlmostEqual(by_label["JacobiIterationKernel"]["avg_us"], jacobi_ns / 1000 / 1000)

    def test_no_kernel_stats_csv_found(self):
        entries, scanned, total_ns = v3.aggregate(os.path.join(FIXTURES, "rocprofv3_no_data"))
        self.assertEqual(entries, [])
        self.assertEqual(scanned, [])
        self.assertEqual(total_ns, 0)


class AggregatePerRankTests(unittest.TestCase):
    def test_single_rank_returns_one_dict(self):
        per_file, scanned = v3.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_single_rank"))
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(per_file), 1)
        self.assertAlmostEqual(per_file[0]["JacobiIterationKernel"], 537449866 / 1e9)

    def test_mpi_2rank_keeps_ranks_separate(self):
        per_file, scanned = v3.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        values = sorted(ft["JacobiIterationKernel"] for ft in per_file)
        self.assertAlmostEqual(values[0], 268000000 / 1e9)
        self.assertAlmostEqual(values[1], 268724933 / 1e9)

    def test_no_kernel_stats_csv_found(self):
        per_file, scanned = v3.aggregate_per_rank(os.path.join(FIXTURES, "rocprofv3_no_data"))
        self.assertEqual(per_file, [])
        self.assertEqual(scanned, [])


if __name__ == "__main__":
    unittest.main()
