import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "stage1_run_dirs.py")

spec = importlib.util.spec_from_file_location("stage1_run_dirs", MODULE_PATH)
srd = importlib.util.module_from_spec(spec)
sys.modules["stage1_run_dirs"] = srd
spec.loader.exec_module(srd)

COMBINED_DIR = os.path.join(FIXTURES, "pop_combined_2rank")
FLAT_LAYOUT_DIR = os.path.join(FIXTURES, "mpi_2rank")  # no rocprof-sys/ subdir, files directly in the dir


class ResolveRunDirsTests(unittest.TestCase):
    def test_detects_paired_subdirs(self):
        cpu_dir, gpu_dir = srd.resolve_run_dirs(COMBINED_DIR)
        self.assertEqual(cpu_dir, os.path.join(COMBINED_DIR, "rocprof-sys"))
        self.assertEqual(gpu_dir, os.path.join(COMBINED_DIR, "rocprofv3"))

    def test_falls_back_to_run_dir_itself_when_no_rocprof_sys_subdir(self):
        # mpi_2rank's wall_clock-*.txt files sit directly in the fixture dir --
        # the tool-1-alone layout (no rocprof-sys/ nesting).
        cpu_dir, gpu_dir = srd.resolve_run_dirs(FLAT_LAYOUT_DIR)
        self.assertEqual(cpu_dir, FLAT_LAYOUT_DIR)
        self.assertIsNone(gpu_dir)

    def test_cpu_only_subdir_with_no_gpu_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "rocprof-sys"))
            cpu_dir, gpu_dir = srd.resolve_run_dirs(tmp)
            self.assertEqual(cpu_dir, os.path.join(tmp, "rocprof-sys"))
            self.assertIsNone(gpu_dir)


if __name__ == "__main__":
    unittest.main()
