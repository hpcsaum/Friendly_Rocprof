import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_pop_metrics.py")

# extract_pop_metrics.py does a plain top-level "from stage5_pop_metrics_table import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_pop_metrics", MODULE_PATH)
pop_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_pop_metrics"] = pop_tool
spec.loader.exec_module(pop_tool)

REF_DIR = os.path.join(FIXTURES, "pop_ref_2rank")
SCALED_DIR = os.path.join(FIXTURES, "pop_scaled_4rank")
COMBINED_DIR = os.path.join(FIXTURES, "pop_combined_2rank")


class WriteReportTests(unittest.TestCase):
    def test_single_run_report_has_no_scaling_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR], dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("=== Metrics ===", report)
            self.assertIn(pop_tool.run_label(REF_DIR), report)
            header_line = next(line for line in report.splitlines() if "ranks" in line and "LB" in line)
            self.assertNotIn("CompE", header_line)  # no scaling columns in the table itself
            self.assertNotIn("(reference)", report)
            self.assertIn("need a scaling study", report)

    def test_multi_run_report_has_one_merged_table_with_scaling_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR, SCALED_DIR], dest, scaling="strong")
            self.assertIn("strong scaling", report)
            self.assertIn(f"{pop_tool.run_label(REF_DIR)} (reference)", report)
            self.assertIn(pop_tool.run_label(SCALED_DIR), report)
            # exactly one table -- not a separate "Per-run"/"Scaling comparison" pair
            self.assertEqual(report.count("=== Metrics"), 1)
            header_line = next(line for line in report.splitlines() if "ranks" in line and "LB" in line)
            self.assertIn("CompE", header_line)
            self.assertIn("GE", header_line)

    def test_not_computed_and_caveats_sections_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "pop_metrics.txt")
            report = pop_tool.write_report([REF_DIR], dest)
            self.assertIn("Serialisation Efficiency", report)
            self.assertIn("PAPI hardware counters", report)
            self.assertIn("MPICH/Cray-MPICH", report)

    def test_gpu_columns_shown_only_when_gpu_data_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            with_gpu = pop_tool.write_report([COMBINED_DIR], os.path.join(tmp, "a.txt"))
            without_gpu = pop_tool.write_report([REF_DIR], os.path.join(tmp, "b.txt"))

        header_with = next(line for line in with_gpu.splitlines() if "ranks" in line and "LB" in line)
        header_without = next(line for line in without_gpu.splitlines() if "ranks" in line and "LB" in line)
        self.assertIn("GPU-Off", header_with)
        self.assertIn("GPU-Util", header_with)
        self.assertIn("GPU-LB", header_with)
        self.assertNotIn("GPU-Off", header_without)
        self.assertNotIn("GPU-Util", header_without)
        self.assertNotIn("GPU-LB", header_without)

    def test_gpu_eff_shown_only_when_both_runs_have_gpu_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            both_gpu = pop_tool.write_report(
                [COMBINED_DIR, COMBINED_DIR], os.path.join(tmp, "a.txt"), scaling="strong"
            )
            mixed = pop_tool.write_report(
                [REF_DIR, COMBINED_DIR], os.path.join(tmp, "b.txt"), scaling="strong"
            )

        header_both = next(line for line in both_gpu.splitlines() if "ranks" in line and "LB" in line)
        header_mixed = next(line for line in mixed.splitlines() if "ranks" in line and "LB" in line)
        self.assertIn("GPU-Eff", header_both)
        self.assertNotIn("GPU-Eff", header_mixed)  # reference (REF_DIR) has no GPU data at all

    def test_column_order_matches_grouping(self):
        # Non-scaling metrics first (GPU-Util before GPU-Off per the user's
        # preferred order), all scaling metrics grouped at the end.
        with tempfile.TemporaryDirectory() as tmp:
            report = pop_tool.write_report(
                [COMBINED_DIR, COMBINED_DIR], os.path.join(tmp, "out.txt"), scaling="strong"
            )
        header = next(line for line in report.splitlines() if "ranks" in line and "LB" in line)
        columns = ["LB", "CommE", "PE", "GPU-Util", "GPU-Off", "GPU-LB", "CompE", "GE", "GPU-Eff"]
        positions = [header.index(c) for c in columns]
        self.assertEqual(positions, sorted(positions))


class MainCliTests(unittest.TestCase):
    def test_scaling_flag_required_with_scaled_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                pop_tool.main([REF_DIR, SCALED_DIR, "-o", os.path.join(tmp, "out.txt")])

    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            pop_tool.main(["/no/such/directory"])

    def test_single_directory_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            pop_tool.main([REF_DIR, "-o", dest])
            self.assertTrue(os.path.isfile(dest))

    def test_multi_directory_with_scaling_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            pop_tool.main([REF_DIR, SCALED_DIR, "--scaling", "weak", "-o", dest])
            self.assertTrue(os.path.isfile(dest))


if __name__ == "__main__":
    unittest.main()
