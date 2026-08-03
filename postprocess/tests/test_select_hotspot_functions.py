import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "select_hotspot_functions.py")

# select_hotspot_functions.py does a plain top-level "import extract_CPU_hotspots",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same technique as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("select_hotspot_functions", MODULE_PATH)
selector = importlib.util.module_from_spec(spec)
sys.modules["select_hotspot_functions"] = selector
spec.loader.exec_module(selector)

import extract_CPU_hotspots as cpu_tool  # noqa: E402
import extract_GPU_hotspots as gpu_tool  # noqa: E402
import extract_hotspots as combined_tool  # noqa: E402

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
        labels = selector.labels_from_output_dir(SINGLE_RANK, top=1)
        self.assertEqual(labels, ["main"])

    def test_mpi_2rank_labels(self):
        labels = selector.labels_from_output_dir(MPI_2RANK, threshold=1.0)
        self.assertIn("compute_stencil", labels)
        self.assertIn("main", labels)
        self.assertNotIn("hipMemcpy", labels)

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
        self.assertIn("main", labels)


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
        self.assertIn("main\tmain", buf.getvalue())

    def test_output_dir_must_exist(self):
        with self.assertRaises(SystemExit):
            selector.main(["--output-dir", "/nonexistent/dir"])


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


if __name__ == "__main__":
    unittest.main()
