"""Tests for select_instrumented_functions.py, which resolves CPU hotspot function names into
rocprof-sys-instrument "-R" input: escaping, selection from an output dir or report, the
--gpu-output-dir/--ancestor-depth selection-widening additions, and the separate
--check-instrumented lost-function-warning mode.

EscapeForInstrumentRegexTests   -- escape_for_instrument_regex()'s metachar escaping
LabelsFromOutputDirTests        -- labels_from_output_dir(): threshold/top/unfiltered selection,
                                    MPI aggregation, dated-subdirectory discovery
LabelsFromReportTests           -- labels_from_report(): CPU-only and combined-report parsing,
                                    column-count regression, malformed/missing-file errors
FindLostFunctionsTests          -- find_lost_functions()'s substring match against instrumented.json,
                                    missing/corrupt file handling
MainResolveModeTests            -- resolve-mode CLI: mutually-exclusive flags, printed label/regex
                                    pairs, --extra-noise-config
MainCheckInstrumentedModeTests  -- --check-instrumented mode: conflicting flags, always-zero exit,
                                    missing instrumented file handling
FlatTreeForAncestorsTests       -- flat_tree_for_ancestors()'s output-dir vs report-mode tree sourcing
MainAncestorExpansionTests      -- --ancestor-depth expansion end to end, including report-mode's
                                    sibling calltree.txt requirement
MainGpuOutputDirTests           -- --gpu-output-dir kernel-owner resolution and dedup against an
                                    already-selected owner
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# select_instrumented_functions.py does a plain top-level "import extract_CPU_hotspots",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same technique as test_extract_hotspots.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

selector = load_module_by_path("select_instrumented_functions", "tools", "select_instrumented_functions.py")

import extract_calltree as calltree_tool  # noqa: E402
import extract_CPU_hotspots as cpu_tool  # noqa: E402
import extract_GPU_hotspots as gpu_tool  # noqa: E402
import extract_hotspots as combined_tool  # noqa: E402
import stage6_noise_config  # noqa: E402

SINGLE_RANK = os.path.join(FIXTURES, "single_rank")
MPI_2RANK = os.path.join(FIXTURES, "mpi_2rank")
NO_TIMING_DATA = os.path.join(FIXTURES, "no_timing_data")
GPU_MPI_2RANK = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
MPI_2RANK_DATED_SUBDIR = os.path.join(FIXTURES, "mpi_2rank_dated_subdir")


class EscapeForInstrumentRegexTests(unittest.TestCase):
    def test_escapes_each_metachar(self):
        for ch in ".^$*+?()[]{}|\\":
            self.assertEqual(selector.escape_for_instrument_regex(ch), "\\" + ch)

    def test_leaves_non_metachars_untouched(self):
        for s in ("<", ">", "~", "::", "MyClass::compute"):
            self.assertEqual(selector.escape_for_instrument_regex(s), s)

    def test_plain_identifier_unchanged(self):
        self.assertEqual(selector.escape_for_instrument_regex("compute_stencil"), "compute_stencil")

    def test_mixed_string(self):
        self.assertEqual(selector.escape_for_instrument_regex("foo(int)"), "foo\\(int\\)")


class LabelsFromOutputDirTests(unittest.TestCase):
    def test_default_threshold_excludes_gpu_api_entries(self):
        labels = selector.labels_from_output_dir(SINGLE_RANK, threshold=1.0)
        self.assertIn("compute_stencil", labels)
        self.assertIn("apply_boundary", labels)
        self.assertIn("main", labels)
        self.assertNotIn("hipMemcpy", labels)
        self.assertNotIn("hipLaunchKernel", labels)

    def test_top_n_selects_highest_only(self):
        # By self-time (the default), compute_stencil beats main -- main's
        # self_sum (2.43) is real but smaller than compute_stencil's (9.32) in
        # this fixture.
        labels = selector.labels_from_output_dir(SINGLE_RANK, top=1)
        self.assertEqual(labels, ["compute_stencil"])

    def test_top_n_unfiltered_selects_main_by_inclusive_time(self):
        labels = selector.labels_from_output_dir(SINGLE_RANK, top=1, unfiltered=True)
        self.assertEqual(labels, ["main"])

    def test_mpi_2rank_labels(self):
        # main's self-time share of runtime here is ~0.1%, below the 1% default
        # threshold -- correctly excluded as a pass-through wrapper, unlike its
        # huge inclusive share which used to pull it in.
        labels = selector.labels_from_output_dir(MPI_2RANK, threshold=1.0)
        self.assertIn("compute_stencil", labels)
        self.assertNotIn("main", labels)
        self.assertNotIn("hipMemcpy", labels)

    def test_mpi_2rank_labels_unfiltered_includes_main(self):
        labels = selector.labels_from_output_dir(MPI_2RANK, threshold=1.0, unfiltered=True)
        self.assertIn("main", labels)

    def test_no_timing_data_raises(self):
        with self.assertRaises(SystemExit):
            selector.labels_from_output_dir(NO_TIMING_DATA)

    def test_sorted_and_deduped(self):
        labels = selector.labels_from_output_dir(SINGLE_RANK, show_all=True)
        self.assertEqual(labels, sorted(set(labels)))

    def test_finds_files_nested_in_a_dated_subdirectory(self):
        # Reproduces the reported bug: rocprof-sys's default time-stamped output
        # subdirectory must not make this tool 4 helper miss the hotspot functions.
        labels = selector.labels_from_output_dir(MPI_2RANK_DATED_SUBDIR, threshold=1.0)
        self.assertIn("compute_stencil", labels)


class LabelsFromReportTests(unittest.TestCase):
    def test_solo_cpu_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            cpu_tool.write_report(SINGLE_RANK, dest)
            labels = selector.labels_from_report(dest)
        self.assertIn("compute_stencil", labels)
        self.assertIn("apply_boundary", labels)
        self.assertNotIn("hipLaunchKernel", labels)
        self.assertNotIn("hipMemcpy", labels)

    def test_combined_report_excludes_gpu_domain_and_gpu_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            combined_tool.write_report(MPI_2RANK, GPU_MPI_2RANK, dest)
            labels = selector.labels_from_report(dest)
        self.assertIn("compute_stencil", labels)
        self.assertIn("main", labels)
        self.assertNotIn("hipMemcpy", labels)
        self.assertNotIn("JacobiIterationKernel", labels)

    def test_function_name_with_spaces_survives_the_new_column_count(self):
        # cpu_tool.format_table() has 6 numeric/count columns ahead of the function
        # name (self(s), %total, total(s), calls, %self) -- labels_from_report's
        # maxsplit must match that exactly, or a multi-word C++ signature gets
        # truncated/misparsed.
        report = (
            "rocprof-sys hotspots report (CPU-side only)\n\n"
            "CPU compute hotspots (candidates for GPU offload) -- showing top 1 of 1 entries\n"
            "    #      self(s)   %total      total(s)       calls    %self  function\n"
            "    1     1.000000     10.0      2.000000          50    50.0  "
            "MyNamespace::Foo(int, double) const\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with open(dest, "w") as f:
                f.write(report)
            labels = selector.labels_from_report(dest)
        self.assertEqual(labels, ["MyNamespace::Foo(int, double) const"])

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


class FindLostFunctionsTests(unittest.TestCase):
    def _write_instrumented_json(self, tmp, entries):
        path = os.path.join(tmp, "instrumented.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    def test_all_requested_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_instrumented_json(tmp, [
                {"function": "main", "signature": {"name": "main"}},
                {"function": "compute_stencil(int, double)", "signature": {"name": "compute_stencil(int, double)"}},
            ])
            lost = selector.find_lost_functions(path, ["main", "compute_stencil"])
        self.assertEqual(lost, [])

    def test_some_missing_returns_exactly_those(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_instrumented_json(tmp, [
                {"function": "main", "signature": {"name": "main"}},
            ])
            lost = selector.find_lost_functions(path, ["main", "compute_stencil", "apply_boundary"])
        self.assertEqual(lost, ["compute_stencil", "apply_boundary"])

    def test_substring_match_against_signature_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_instrumented_json(tmp, [
                {"function": "unrelated", "signature": {"name": "MyClass::compute(int, double)"}},
            ])
            lost = selector.find_lost_functions(path, ["MyClass::compute"])
        self.assertEqual(lost, [])

    def test_missing_file_raises_dedicated_exception(self):
        with self.assertRaises(selector.InstrumentedFileError):
            selector.find_lost_functions("/nonexistent/instrumented.json", ["main"])

    def test_corrupt_file_raises_dedicated_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "instrumented.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            with self.assertRaises(selector.InstrumentedFileError):
                selector.find_lost_functions(path, ["main"])


class MainResolveModeTests(unittest.TestCase):
    def test_mutually_exclusive_report_and_output_dir(self):
        with self.assertRaises(SystemExit):
            selector.main(["--report", "a", "--output-dir", "b"])

    def test_requires_one_source(self):
        with self.assertRaises(SystemExit):
            selector.main([])

    def test_prints_label_regex_pairs(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--output-dir", SINGLE_RANK, "--top", "1"])
        self.assertIn("compute_stencil\tcompute_stencil", buf.getvalue())

    def test_unfiltered_prints_main_by_inclusive_time(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--output-dir", SINGLE_RANK, "--top", "1", "--unfiltered"])
        self.assertIn("main\tmain", buf.getvalue())

    def test_output_dir_must_exist(self):
        with self.assertRaises(SystemExit):
            selector.main(["--output-dir", "/nonexistent/dir"])

    def test_extra_noise_config_flag_excludes_a_configured_row(self):
        self.addCleanup(stage6_noise_config.configure, None)
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["apply_boundary"]}}, f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                selector.main(["--output-dir", SINGLE_RANK, "--all", "--extra-noise-config", config_path])
        self.assertNotIn("apply_boundary", buf.getvalue())
        self.assertIn("compute_stencil", buf.getvalue())

    def test_extra_noise_config_conflicts_with_report(self):
        with self.assertRaises(SystemExit):
            selector.main(["--report", "a.txt", "--extra-noise-config", "b.json"])


class MainCheckInstrumentedModeTests(unittest.TestCase):
    def test_conflicts_with_report(self):
        with self.assertRaises(SystemExit):
            selector.main(["--check-instrumented", "x.json", "--report", "y.txt"])

    def test_always_exits_zero_and_warns_on_lost_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "instrumented.json")
            with open(path, "w") as f:
                json.dump([{"function": "main", "signature": {"name": "main"}}], f)

            stdin = io.StringIO("main\ncompute_stencil\n")
            stderr = io.StringIO()
            with unittest.mock.patch("sys.stdin", stdin), contextlib.redirect_stderr(stderr):
                result = selector.main(["--check-instrumented", path])
        self.assertIsNone(result)
        self.assertIn("compute_stencil", stderr.getvalue())
        self.assertNotIn("warning: hotspot function 'main'", stderr.getvalue())

    def test_missing_instrumented_file_skips_without_raising(self):
        stdin = io.StringIO("main\n")
        stderr = io.StringIO()
        with unittest.mock.patch("sys.stdin", stdin), contextlib.redirect_stderr(stderr):
            result = selector.main(["--check-instrumented", "/nonexistent/instrumented.json"])
        self.assertIsNone(result)
        self.assertIn("skipping lost-function check", stderr.getvalue())

    def test_conflicts_with_gpu_output_dir(self):
        with self.assertRaises(SystemExit):
            selector.main(["--check-instrumented", "x.json", "--gpu-output-dir", "y"])

    def test_conflicts_with_ancestor_depth(self):
        with self.assertRaises(SystemExit):
            selector.main(["--check-instrumented", "x.json", "--ancestor-depth", "2"])


class FlatTreeForAncestorsTests(unittest.TestCase):
    def test_output_dir_mode_builds_the_real_merged_tree(self):
        flat = selector.flat_tree_for_ancestors(None, MPI_2RANK)
        by_label = {r["label"]: r for r in flat}
        self.assertIn("compute_stencil", by_label)
        self.assertIs(by_label["compute_stencil"]["parent"], by_label["main"])

    def test_report_mode_reads_sibling_calltree_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            hotspots = os.path.join(tmp, "hotspots.txt")
            calltree = os.path.join(tmp, "calltree.txt")
            cpu_tool.write_report(MPI_2RANK, hotspots)
            calltree_tool.write_report(MPI_2RANK, None, calltree)

            flat = selector.flat_tree_for_ancestors(hotspots, None)
        by_label = {r["label"]: r for r in flat}
        self.assertIn("compute_stencil", by_label)
        self.assertIs(by_label["compute_stencil"]["parent"], by_label["main"])

    def test_report_mode_missing_calltree_raises_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            hotspots = os.path.join(tmp, "hotspots.txt")
            with open(hotspots, "w") as f:
                f.write("placeholder\n")
            with self.assertRaisesRegex(SystemExit, "calltree.txt"):
                selector.flat_tree_for_ancestors(hotspots, None)


class MainAncestorExpansionTests(unittest.TestCase):
    def test_default_depth_pulls_in_the_immediate_caller(self):
        buf, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(stderr):
            selector.main(["--output-dir", MPI_2RANK, "--top", "1"])
        self.assertIn("compute_stencil\tcompute_stencil", buf.getvalue())
        self.assertIn("main\tmain", buf.getvalue())
        self.assertIn("ancestor function(s) added for tree connectivity (--ancestor-depth 1)", stderr.getvalue())

    def test_ancestor_depth_zero_disables_expansion(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            selector.main(["--output-dir", MPI_2RANK, "--top", "1", "--ancestor-depth", "0"])
        self.assertIn("compute_stencil\tcompute_stencil", buf.getvalue())
        self.assertNotIn("main\tmain", buf.getvalue())

    def test_report_mode_uses_sibling_calltree_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            hotspots = os.path.join(tmp, "hotspots.txt")
            calltree = os.path.join(tmp, "calltree.txt")
            cpu_tool.write_report(MPI_2RANK, hotspots)
            calltree_tool.write_report(MPI_2RANK, None, calltree)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                selector.main(["--report", hotspots])
        self.assertIn("main\tmain", buf.getvalue())

    def test_report_mode_missing_calltree_raises_with_default_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            hotspots = os.path.join(tmp, "hotspots.txt")
            cpu_tool.write_report(MPI_2RANK, hotspots)
            with self.assertRaisesRegex(SystemExit, "calltree.txt"):
                selector.main(["--report", hotspots])

    def test_report_mode_missing_calltree_is_fine_with_depth_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            hotspots = os.path.join(tmp, "hotspots.txt")
            cpu_tool.write_report(MPI_2RANK, hotspots)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                selector.main(["--report", hotspots, "--ancestor-depth", "0"])
        self.assertIn("compute_stencil\tcompute_stencil", buf.getvalue())


class MainGpuOutputDirTests(unittest.TestCase):
    def _make_gpu_dir(self, tmp, kernel_name):
        gpu_dir = os.path.join(tmp, "rocprofv3", "myhost")
        os.makedirs(gpu_dir)
        with open(os.path.join(gpu_dir, "1_kernel_stats.csv"), "w") as f:
            f.write('"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"\n')
            f.write(f'"{kernel_name}",1000,900000000,900000.0,90.0,800000,1000000,5000.0\n')
        return os.path.dirname(gpu_dir)

    def test_decoded_kernel_owner_is_added_to_the_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu_dir = self._make_gpu_dir(tmp, "jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form")
            buf, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(stderr):
                selector.main(["--output-dir", MPI_2RANK, "--top", "1", "--gpu-output-dir", gpu_dir])
        self.assertIn("jacobi_sweep$pressure_solver_mod_", buf.getvalue())
        self.assertIn("GPU-kernel-owner function(s) added from --gpu-output-dir", stderr.getvalue())

    def test_unrecognized_kernel_names_add_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu_dir = self._make_gpu_dir(tmp, "JacobiIterationKernel")
            buf, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(stderr):
                selector.main(["--output-dir", MPI_2RANK, "--top", "1", "--gpu-output-dir", gpu_dir])
        self.assertNotIn("GPU-kernel-owner function(s) added", stderr.getvalue())

    def test_owner_already_selected_is_not_double_counted(self):
        # compute_stencil's own kernel-owner decode would just be itself, if it happened to
        # already be the hotspot pulled in -- confirms the "- selected" dedup in main() rather
        # than asserting the exact resolved label, which is decode-scheme-specific.
        with tempfile.TemporaryDirectory() as tmp:
            gpu_dir = self._make_gpu_dir(tmp, "compute_stencil$ck_L1_1_cce$noloop$form")
            buf, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(stderr):
                selector.main(["--output-dir", MPI_2RANK, "--top", "1", "--gpu-output-dir", gpu_dir])
        self.assertNotIn("GPU-kernel-owner function(s) added", stderr.getvalue())

    def test_gpu_output_dir_requires_an_existing_directory(self):
        with self.assertRaises(SystemExit):
            selector.main(["--output-dir", MPI_2RANK, "--top", "1", "--gpu-output-dir", "/nonexistent/dir"])


if __name__ == "__main__":
    unittest.main()
