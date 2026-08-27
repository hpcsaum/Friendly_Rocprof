import contextlib
import glob
import io
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# select_hotspot_kernels.py does a plain top-level "import extract_GPU_hotspots", relying on
# its own directory being on sys.path -- true automatically when run directly, but not when
# loaded here by explicit file path, so replicate that manually (same technique as
# test_select_instrumented_functions.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

selector = load_module_by_path("select_hotspot_kernels", "tools", "select_hotspot_kernels.py")

import extract_GPU_hotspots as gpu_tool  # noqa: E402
import extract_hotspots as combined_tool  # noqa: E402
from stage5_table_render import wrap_trailing_label  # noqa: E402
import stage6_time_range_config as trc  # noqa: E402

GPU_SINGLE_RANK = os.path.join(FIXTURES, "rocprofv3_single_rank")
GPU_MPI_2RANK = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
GPU_NO_DATA = os.path.join(FIXTURES, "rocprofv3_no_data")
CPU_MPI_2RANK = os.path.join(FIXTURES, "mpi_2rank")
TRACE_KERNEL_SELECTION_DIR = os.path.join(FIXTURES, "trace_gpu_kernel_selection")


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

    def test_pre_wrapped_kernel_name_reconstructs_across_physical_lines(self):
        # A kernel name long enough that extract_GPU_hotspots.py's real render_table() call
        # would hard-wrap it across 2+ physical lines -- confirms labels_from_report() (via
        # iter_table_rows()) rejoins it back into one label, not just its first physical line.
        long_kernel = "void MyKernel<" + "T" * 100 + ">(int, float const*)"
        prefix = "    1     1.000000     10.0          50      20.00  "
        row_text = wrap_trailing_label(prefix, long_kernel, width=90)
        report = (
            "rocprofv3 GPU kernel hotspots report\n\n"
            "GPU kernel hotspots -- showing top 1 of 1 entries\n"
            "    #      total(s)   %total       calls    avg(us)  kernel\n"
            + row_text + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with open(dest, "w") as f:
                f.write(report)
            labels = selector.labels_from_report(dest)
        self.assertEqual(labels, [long_kernel])

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


def _clear_kernel_selection_cache():
    for f in glob.glob(os.path.join(TRACE_KERNEL_SELECTION_DIR, "*.agg.json")):
        os.remove(f)


class LabelsFromTraceDirTests(unittest.TestCase):
    def tearDown(self):
        _clear_kernel_selection_cache()
        trc.configure(None)

    def test_only_kernel_labels_are_returned(self):
        # cpu_heavy_function's self_sum (60.0) dwarfs every kernel's, and hipLaunchKernel is a
        # launch call, not a dispatch -- neither may leak into the result.
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, show_all=True,
                                                  require_multiple_calls=False)
        self.assertEqual(set(labels), {"kernel_a.kd", "kernel_b.kd", "kernel_c.kd"})

    def test_default_excludes_single_dispatch_kernel(self):
        # kernel_b.kd is dispatched only once total -- excluded by default.
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, show_all=True)
        self.assertIn("kernel_a.kd", labels)
        self.assertIn("kernel_c.kd", labels)
        self.assertNotIn("kernel_b.kd", labels)

    def test_all_dispatches_includes_single_dispatch_kernel(self):
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, show_all=True,
                                                  require_multiple_calls=False)
        self.assertIn("kernel_b.kd", labels)

    def test_top_n_selects_highest_only(self):
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, top=1)
        self.assertEqual(labels, ["kernel_a.kd"])

    def test_threshold_filters(self):
        # kernel_c.kd is ~7.4% of total kernel time -- excluded at a 10% threshold.
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, threshold=10.0,
                                                  require_multiple_calls=False)
        self.assertEqual(labels, ["kernel_a.kd", "kernel_b.kd"])

    def test_no_trace_data_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                selector.labels_from_trace_dir(tmp)

    def test_nonexistent_dir_raises(self):
        with self.assertRaises(SystemExit):
            selector.labels_from_trace_dir("/nonexistent/trace-dir")


class MainCLITests(unittest.TestCase):
    def tearDown(self):
        _clear_kernel_selection_cache()
        trc.configure(None)

    def test_requires_one_source(self):
        with self.assertRaises(SystemExit):
            selector.main([])

    def test_mutually_exclusive_report_and_output_dir(self):
        with self.assertRaises(SystemExit):
            selector.main(["--report", "a", "--output-dir", "b"])

    def test_mutually_exclusive_trace_dir_and_report(self):
        with self.assertRaises(SystemExit):
            selector.main(["--report", "a", "--trace-dir", "b"])

    def test_output_dir_must_exist(self):
        with self.assertRaises(SystemExit):
            selector.main(["--output-dir", "/nonexistent/dir"])

    def test_trace_dir_source_prints_kernel_labels(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--trace-dir", TRACE_KERNEL_SELECTION_DIR, "--all-dispatches"])
        printed = set(buf.getvalue().splitlines())
        self.assertEqual(printed, {"kernel_a.kd", "kernel_b.kd", "kernel_c.kd"})

    def test_time_range_flag_is_accepted_with_trace_dir(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--trace-dir", TRACE_KERNEL_SELECTION_DIR, "--time-range", "0:65",
                            "--all-dispatches"])
        printed = set(buf.getvalue().splitlines())
        # kernel_c.kd's second dispatch starts at 64.5s and only partially overlaps [0, 65] --
        # still contributes some in-window time, so it should still show up; the main point of
        # this test is that --time-range is accepted at all on the --trace-dir path.
        self.assertIn("kernel_a.kd", printed)


if __name__ == "__main__":
    unittest.main()
