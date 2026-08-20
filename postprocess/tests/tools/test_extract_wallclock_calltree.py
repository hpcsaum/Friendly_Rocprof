import importlib.util
import json
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "tools", "extract_wallclock_calltree.py")

# extract_wallclock_calltree.py does a plain top-level "from stage1_run_dirs import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("extract_wallclock_calltree", MODULE_PATH)
ct_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_wallclock_calltree"] = ct_tool
spec.loader.exec_module(ct_tool)

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


class ResolveTwoDirsTests(unittest.TestCase):
    # extract_wallclock_calltree.py imports resolve_two_dirs() from stage1_run_dirs.py (which has
    # its own thorough direct tests) -- this just confirms the import wires through correctly.
    def test_detects_paired_subdirs(self):
        cpu_dir, gpu_dir = ct_tool.resolve_two_dirs(KERNEL_ANCHOR_DIR, None)
        self.assertEqual(cpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprof-sys"))
        self.assertEqual(gpu_dir, os.path.join(KERNEL_ANCHOR_DIR, "rocprofv3"))

    def test_falls_back_to_run_dir_itself_when_flat(self):
        cpu_dir, gpu_dir = ct_tool.resolve_two_dirs(MPI_2RANK_DIR, None)
        self.assertEqual(cpu_dir, MPI_2RANK_DIR)
        self.assertIsNone(gpu_dir)


class HeaderProseTests(unittest.TestCase):
    def test_header_states_ranks_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "wallclock_calltree.txt")
            report = ct_tool.write_report(MPI_2RANK_DIR, None, dest)
        self.assertIn("MPI ranks: 2", report)


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
            ct_tool.main([MPI_2RANK_DIR, "-o", dest, "--max-depth", "1"])
            self.assertTrue(os.path.isfile(dest))

    def test_explicit_two_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            ct_tool.main([MPI_2RANK_DIR, MPI_2RANK_DIR, "-o", dest])
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
                json.dump({"add": {"other": ["compute_stencil"]}}, f)
            ct_tool.main([MPI_2RANK_DIR, "-o", dest, "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("compute_stencil", report)


if __name__ == "__main__":
    unittest.main()
