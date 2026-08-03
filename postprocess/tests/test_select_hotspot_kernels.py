import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "select_hotspot_kernels.py")

# select_hotspot_kernels.py does a plain top-level "import extract_GPU_hotspots", relying on
# its own directory being on sys.path -- true automatically when run directly, but not when
# loaded here by explicit file path, so replicate that manually (same technique as
# test_select_hotspot_functions.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("select_hotspot_kernels", MODULE_PATH)
selector = importlib.util.module_from_spec(spec)
sys.modules["select_hotspot_kernels"] = selector
spec.loader.exec_module(selector)

import extract_GPU_hotspots as gpu_tool  # noqa: E402
import extract_hotspots as combined_tool  # noqa: E402

GPU_SINGLE_RANK = os.path.join(FIXTURES, "rocprofv3_single_rank")
GPU_MPI_2RANK = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
GPU_NO_DATA = os.path.join(FIXTURES, "rocprofv3_no_data")
CPU_MPI_2RANK = os.path.join(FIXTURES, "mpi_2rank")


class LabelsFromOutputDirTests(unittest.TestCase):
    def test_default_excludes_single_call_kernel(self):
        # rocprofv3_single_rank's __hipRegisterFatBinary row has Calls=1 -- no second
        # dispatch for the launcher's default -d 2 to target, so it's excluded by default.
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, show_all=True)
        self.assertIn("JacobiIterationKernel", labels)
        self.assertIn("BoundaryKernel", labels)
        self.assertNotIn("__hipRegisterFatBinary", labels)

    def test_all_dispatches_includes_single_call_kernel(self):
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, show_all=True, require_multiple_calls=False)
        self.assertIn("__hipRegisterFatBinary", labels)

    def test_underlying_aggregate_still_reports_the_single_call_kernel(self):
        # confirms the exclusion happens in select_hotspot_kernels, not because the data
        # is somehow missing from extract_GPU_hotspots.aggregate() itself.
        entries, _scanned, _total_ns = gpu_tool.aggregate(GPU_SINGLE_RANK)
        by_label = {e["label"]: e for e in entries}
        self.assertEqual(by_label["__hipRegisterFatBinary"]["count"], 1)

    def test_mpi_2rank_both_kernels_eligible(self):
        # per-rank Calls=500 each, aggregated to 1000 -- well above the count>=2 bar.
        labels = selector.labels_from_output_dir(GPU_MPI_2RANK, show_all=True)
        self.assertIn("JacobiIterationKernel", labels)
        self.assertIn("BoundaryKernel", labels)

    def test_top_n_selects_highest_only(self):
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, top=1)
        self.assertEqual(labels, ["JacobiIterationKernel"])

    def test_threshold_filters(self):
        # BoundaryKernel is ~9.6% of total in this fixture -- excluded at a 50% threshold.
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, threshold=50.0)
        self.assertEqual(labels, ["JacobiIterationKernel"])

    def test_no_timing_data_raises(self):
        with self.assertRaises(SystemExit):
            selector.labels_from_output_dir(GPU_NO_DATA)

    def test_sorted_and_deduped(self):
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, show_all=True)
        self.assertEqual(labels, sorted(set(labels)))


class LabelsFromReportTests(unittest.TestCase):
    def test_solo_gpu_report_default_excludes_single_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            gpu_tool.write_report(GPU_SINGLE_RANK, dest, show_all=True)
            labels = selector.labels_from_report(dest)
        self.assertIn("JacobiIterationKernel", labels)
        self.assertIn("BoundaryKernel", labels)
        self.assertNotIn("__hipRegisterFatBinary", labels)

    def test_solo_gpu_report_all_dispatches_includes_single_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            gpu_tool.write_report(GPU_SINGLE_RANK, dest, show_all=True)
            labels = selector.labels_from_report(dest, require_multiple_calls=False)
        self.assertIn("__hipRegisterFatBinary", labels)

    def test_combined_report_reads_table_3(self):
        # Proves the same parser covers tool 3's combined report format too -- table 3's
        # header ("=== 3. GPU kernel hotspots (rocprofv3 run) -- showing ... ===") shares the
        # "GPU kernel hotspots" substring with tool 2's own report header, and appears before
        # table 6 ("GPU kernel load imbalance"), so the first-match search finds the right one.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            combined_tool.write_report(CPU_MPI_2RANK, GPU_MPI_2RANK, dest, show_all=True)
            labels = selector.labels_from_report(dest)
        self.assertIn("JacobiIterationKernel", labels)
        self.assertIn("BoundaryKernel", labels)

    def test_maxsplit_regression_multiword_kernel_signature(self):
        # gpu_tool.format_table()'s columns: # total(s) %total calls avg(us) kernel -- one
        # fewer column than the CPU extractor's table (no self-time metric), so this must be
        # maxsplit=5/parts[5], not the CPU version's maxsplit=6/parts[6]. Verify explicitly
        # rather than trusting the arithmetic, same spirit as the CPU extractor's own
        # regression test for its column count.
        report = (
            "rocprofv3 GPU kernel hotspots report\n\n"
            "GPU kernel hotspots -- showing top 1 of 1 entries\n"
            "    #      total(s)   %total       calls    avg(us)  kernel\n"
            "    1     1.000000     10.0          50      20.00  "
            "void MyKernel<int, float>(int*, float const*) const\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with open(dest, "w") as f:
                f.write(report)
            labels = selector.labels_from_report(dest)
        self.assertEqual(labels, ["void MyKernel<int, float>(int*, float const*) const"])

    def test_malformed_text_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "not_a_report.txt")
            with open(dest, "w") as f:
                f.write("this is not a hotspots report at all\n")
            with self.assertRaises(SystemExit):
                selector.labels_from_report(dest)

    def test_missing_file_raises(self):
        with self.assertRaises((SystemExit, OSError)):
            selector.labels_from_report("/nonexistent/hotspots.txt")


class MainCLITests(unittest.TestCase):
    def test_requires_one_source(self):
        with self.assertRaises(SystemExit):
            selector.main([])

    def test_mutually_exclusive_report_and_output_dir(self):
        with self.assertRaises(SystemExit):
            selector.main(["--report", "a", "--output-dir", "b"])

    def test_output_dir_must_exist(self):
        with self.assertRaises(SystemExit):
            selector.main(["--output-dir", "/nonexistent/dir"])


if __name__ == "__main__":
    unittest.main()
