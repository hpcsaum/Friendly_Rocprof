import importlib.util
import json
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_calltree.py")

# extract_calltree.py does a plain top-level "from stage1_run_dirs import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_calltree", MODULE_PATH)
ct_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_calltree"] = ct_tool
spec.loader.exec_module(ct_tool)

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

FILTERS_DIR = os.path.join(FIXTURES, "calltree_sampling_filters")
MULTI_RANK_DIR = os.path.join(FIXTURES, "calltree_sampling_multi_rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class HeaderProseTests(unittest.TestCase):
    def test_header_states_ranks_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            report = ct_tool.write_report(MULTI_RANK_DIR, None, dest)
        self.assertIn("MPI ranks: 3", report)


class ShowAllInternalsTests(unittest.TestCase):
    def test_equivalent_to_all_four_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "calltree.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--show-all-internals"])
            with open(dest) as f:
                report = f.read()
        for label in ("hipLaunchKernel", "__libc_start_main", "MPIR_Allreduce_cdesc", "posix_memalign"):
            self.assertIn(label, report)


class MainCliTests(unittest.TestCase):
    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            ct_tool.main(["/no/such/directory"])

    def test_empty_input_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            ct_tool.main([EMPTY_DIR])

    def test_end_to_end_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            ct_tool.main([FILTERS_DIR, "-o", dest, "--max-depth", "1"])
            self.assertTrue(os.path.isfile(dest))

    def test_explicit_two_directories(self):
        # FILTERS_DIR has no GPU data of its own -- passing MULTI_RANK_DIR's GPU-less directory
        # as an explicit second arg still exercises the two-directory code path end to end.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            ct_tool.main([FILTERS_DIR, FILTERS_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
            self.assertIn("CPU run directory:", report)
            self.assertIn("GPU run directory:", report)

    def test_extra_noise_config_flag_excludes_a_configured_row(self):
        self.addCleanup(stage6_noise_config.configure, None)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["foo_normal_call"]}}, f)
            ct_tool.main([FILTERS_DIR, "-o", dest, "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("foo_normal_call", report)


if __name__ == "__main__":
    unittest.main()
