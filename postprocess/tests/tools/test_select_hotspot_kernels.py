"""Tests for select_hotspot_kernels.py, which resolves GPU hotspot kernel names into
rocprof-compute "-k" input from a hotspots report, a rocprofv3 output directory, or a
rocprof-sys trace directory.

SelectionKwargsTests       -- shared selection semantics (require_multiple_calls, top, threshold,
                               no-data errors) exercised once per applicable loader via subTest
LabelsFromOutputDirTests   -- labels_from_output_dir()-specific behavior: single-call exclusion
                               source, MPI aggregation, sorted/deduped output
LabelsFromReportTests      -- labels_from_report(): combined-report table selection, column-count
                               regression, malformed/missing-file errors
LabelsFromTraceDirTests    -- labels_from_trace_dir()'s discover/aggregate/select wiring end to end
MainCLITests               -- CLI entry point: source requirement, mutually-exclusive flags,
                               --report/--output-dir/--trace-dir output, --time-range acceptance
"""

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
from _tools_test_helpers import assert_help_leads_with_explanation  # noqa: E402

selector = load_module_by_path("select_hotspot_kernels", "tools", "select_hotspot_kernels.py")

import extract_GPU_hotspots as gpu_tool  # noqa: E402
import extract_hotspots as combined_tool  # noqa: E402
import stage6_time_range_config as trc  # noqa: E402

GPU_SINGLE_RANK = os.path.join(FIXTURES, "rocprofv3_single_rank")
GPU_MPI_2RANK = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
GPU_NO_DATA = os.path.join(FIXTURES, "rocprofv3_no_data")
CPU_MPI_2RANK = os.path.join(FIXTURES, "mpi_2rank")
TRACE_KERNEL_SELECTION_DIR = os.path.join(FIXTURES, "trace_gpu_kernel_selection")


def _gpu_report_labels(source_dir, **kwargs):
    """labels_from_output_dir()/labels_from_trace_dir() both take (source_dir, **selection
    kwargs) directly; labels_from_report() instead reads an already-written report file, and has
    no show_all of its own (write_report() decides what's IN the report text; require_multiple_calls
    is the only read-time filter labels_from_report() itself applies). This wraps report writing
    +reading behind the same (source_dir, **kwargs) shape as the other two loaders, so all three
    can share one subTest table below -- show_all is accepted and ignored (always written True)
    rather than forcing every call site to know which loader does and doesn't have it."""
    kwargs.pop("show_all", None)
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "hotspots.txt")
        gpu_tool.write_report(source_dir, dest, show_all=True)
        return selector.labels_from_report(dest, **kwargs)


class SelectionKwargsTests(unittest.TestCase):
    """labels_from_output_dir()/labels_from_report()/labels_from_trace_dir() share identical
    selection semantics (require_multiple_calls default-excludes a kernel with only one
    dispatch/call; top/threshold rank and filter the rest) -- each shape below is exercised once
    per applicable loader via subTest, rather than once per loader as separate test methods.
    labels_from_report() has no top/threshold of its own (see _gpu_report_labels()), so it's
    absent from those two shapes' case lists."""

    def test_default_excludes_the_single_dispatch_kernel(self):
        # GPU_SINGLE_RANK's __hipRegisterFatBinary has Calls=1 -- no second dispatch for the
        # default require_multiple_calls=True bar to clear. TRACE_KERNEL_SELECTION_DIR's
        # kernel_b.kd is dispatched only once total, same rule.
        cases = [
            ("output_dir", selector.labels_from_output_dir, GPU_SINGLE_RANK,
             {"JacobiIterationKernel", "BoundaryKernel"}, {"__hipRegisterFatBinary"}),
            ("report", _gpu_report_labels, GPU_SINGLE_RANK,
             {"JacobiIterationKernel", "BoundaryKernel"}, {"__hipRegisterFatBinary"}),
            ("trace_dir", selector.labels_from_trace_dir, TRACE_KERNEL_SELECTION_DIR,
             {"kernel_a.kd", "kernel_c.kd"}, {"kernel_b.kd"}),
        ]
        for name, loader, source, expect_in, expect_out in cases:
            with self.subTest(source=name):
                labels = loader(source, show_all=True)
                for label in expect_in:
                    self.assertIn(label, labels)
                for label in expect_out:
                    self.assertNotIn(label, labels)

    def test_all_dispatches_includes_the_single_dispatch_kernel(self):
        cases = [
            ("output_dir", selector.labels_from_output_dir, GPU_SINGLE_RANK, "__hipRegisterFatBinary"),
            ("report", _gpu_report_labels, GPU_SINGLE_RANK, "__hipRegisterFatBinary"),
            ("trace_dir", selector.labels_from_trace_dir, TRACE_KERNEL_SELECTION_DIR, "kernel_b.kd"),
        ]
        for name, loader, source, expect_label in cases:
            with self.subTest(source=name):
                labels = loader(source, show_all=True, require_multiple_calls=False)
                self.assertIn(expect_label, labels)

    def test_top_n_selects_highest_only(self):
        cases = [
            ("output_dir", selector.labels_from_output_dir, GPU_SINGLE_RANK, ["JacobiIterationKernel"]),
            ("trace_dir", selector.labels_from_trace_dir, TRACE_KERNEL_SELECTION_DIR, ["kernel_a.kd"]),
        ]
        for name, loader, source, expected in cases:
            with self.subTest(source=name):
                self.assertEqual(loader(source, top=1), expected)

    def test_threshold_filters(self):
        # BoundaryKernel is ~9.6% of GPU_SINGLE_RANK's total -- excluded at a 50% threshold.
        # kernel_c.kd is ~7.4% of TRACE_KERNEL_SELECTION_DIR's total kernel time -- excluded at 10%.
        cases = [
            ("output_dir", selector.labels_from_output_dir, GPU_SINGLE_RANK,
             {"threshold": 50.0}, ["JacobiIterationKernel"]),
            ("trace_dir", selector.labels_from_trace_dir, TRACE_KERNEL_SELECTION_DIR,
             {"threshold": 10.0, "require_multiple_calls": False}, ["kernel_a.kd", "kernel_b.kd"]),
        ]
        for name, loader, source, kwargs, expected in cases:
            with self.subTest(source=name):
                self.assertEqual(loader(source, **kwargs), expected)

    def test_no_data_raises(self):
        with tempfile.TemporaryDirectory() as empty_trace_dir:
            cases = [
                ("output_dir_no_timing_data", lambda: selector.labels_from_output_dir(GPU_NO_DATA)),
                ("trace_dir_empty", lambda: selector.labels_from_trace_dir(empty_trace_dir)),
                ("trace_dir_nonexistent", lambda: selector.labels_from_trace_dir("/nonexistent/trace-dir")),
            ]
            for name, call in cases:
                with self.subTest(source=name):
                    with self.assertRaises(SystemExit):
                        call()


class LabelsFromOutputDirTests(unittest.TestCase):
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

    def test_sorted_and_deduped(self):
        labels = selector.labels_from_output_dir(GPU_SINGLE_RANK, show_all=True)
        self.assertEqual(labels, sorted(set(labels)))


class LabelsFromReportTests(unittest.TestCase):
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


def _clear_kernel_selection_cache():
    for f in glob.glob(os.path.join(TRACE_KERNEL_SELECTION_DIR, "*.agg.json")):
        os.remove(f)


class LabelsFromTraceDirTests(unittest.TestCase):
    def tearDown(self):
        _clear_kernel_selection_cache()
        trc.configure(None)

    def test_loader_wires_discovery_aggregation_and_selection_together(self):
        # The gpu_kernel-only domain filter itself is exhaustively covered directly against
        # aggregate_gpu_kernels() in test_stage4_rocprofsys_trace_flat.py -- this confirms
        # labels_from_trace_dir()'s own composition (discover_ranks -> aggregate_gpu_kernels ->
        # select_entries) reaches the same result end to end: cpu_heavy_function's huge self_sum
        # and the hipLaunchKernel launch call both correctly stay out of the returned labels.
        labels = selector.labels_from_trace_dir(TRACE_KERNEL_SELECTION_DIR, show_all=True,
                                                  require_multiple_calls=False)
        self.assertEqual(set(labels), {"kernel_a.kd", "kernel_b.kd", "kernel_c.kd"})


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

    def test_output_dir_source_prints_kernel_labels(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--output-dir", GPU_MPI_2RANK, "--all-dispatches"])
        printed = set(buf.getvalue().splitlines())
        self.assertEqual(printed, {"BoundaryKernel", "JacobiIterationKernel"})

    def test_report_source_prints_kernel_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "hotspots.txt")
            gpu_tool.write_report(GPU_MPI_2RANK, report_path, show_all=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                selector.main(["--report", report_path, "--all-dispatches"])
        printed = set(buf.getvalue().splitlines())
        self.assertEqual(printed, {"BoundaryKernel", "JacobiIterationKernel"})

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


class HelpTextTests(unittest.TestCase):
    def test_help_leads_with_explanation(self):
        assert_help_leads_with_explanation(self, selector, "rocprofiler-compute/en/latest")


if __name__ == "__main__":
    unittest.main()
