import importlib.util
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_hotspots.py")

# extract_hotspots.py does a plain top-level "import extract_CPU_hotspots"/
# "import extract_GPU_hotspots", relying on its own directory being on sys.path --
# true automatically when it's run directly (`python3 extract_hotspots.py`),
# but not when loaded here by explicit file path, so replicate that manually.
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_hotspots", MODULE_PATH)
combined = importlib.util.module_from_spec(spec)
sys.modules["extract_hotspots"] = combined
spec.loader.exec_module(combined)

CPU_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_DIR = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
CPU_DIR_SINGLE = os.path.join(FIXTURES, "single_rank")
GPU_DIR_SINGLE = os.path.join(FIXTURES, "rocprofv3_single_rank")
CPU_DIR_EMPTY = os.path.join(FIXTURES, "no_timing_data")
GPU_DIR_EMPTY = os.path.join(FIXTURES, "rocprofv3_no_data")


class BuildCombinedViewTests(unittest.TestCase):
    def test_subtraction_arithmetic_matches_documented_formula(self):
        fused, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = combined.build_combined_view(CPU_DIR, GPU_DIR)

        # Independently recompute expected numbers straight from the sibling
        # modules' own aggregate() on the same fixtures, rather than hand-typing
        # decimals -- this is the actual documented formula, not a guess.
        import extract_CPU_hotspots as cpu_tool
        import extract_GPU_hotspots as gpu_tool

        exp_cpu_entries, exp_gpu_api_entries, exp_cpu_scanned, exp_cpu_total_raw = cpu_tool.aggregate(CPU_DIR)
        exp_gpu_entries, exp_gpu_scanned, exp_gpu_total_ns = gpu_tool.aggregate(GPU_DIR)
        exp_overhead = sum(e["sum"] for e in exp_gpu_api_entries)
        exp_cpu_pure = max(0.0, exp_cpu_total_raw - exp_overhead)
        exp_gpu_total_sec = exp_gpu_total_ns / 1e9
        exp_combined_total = exp_cpu_pure + exp_gpu_total_sec

        self.assertAlmostEqual(info["cpu_total_raw"], exp_cpu_total_raw)
        self.assertAlmostEqual(info["gpu_api_overhead_sec"], exp_overhead)
        self.assertGreater(info["gpu_api_overhead_sec"], 0)  # fixture's hipMemcpy bucket is non-zero
        self.assertAlmostEqual(info["cpu_pure_total_sec"], exp_cpu_pure)
        self.assertAlmostEqual(info["gpu_total_sec"], exp_gpu_total_sec)
        self.assertAlmostEqual(info["combined_total_sec"], exp_combined_total)
        # the subtraction must have actually removed something, not be a no-op
        self.assertLess(info["cpu_pure_total_sec"], info["cpu_total_raw"])

    def test_fused_pct_total_differs_from_each_sides_own_standalone_pct(self):
        fused, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = combined.build_combined_view(CPU_DIR, GPU_DIR)
        fused_by_label = {(e["label"], e["domain"]): e for e in fused}
        cpu_by_label = {e["label"]: e for e in cpu_entries}
        gpu_by_label = {e["label"]: e for e in gpu_entries}

        # same absolute "sum" as the standalone tools...
        self.assertAlmostEqual(fused_by_label[("compute_stencil", "CPU")]["sum"], cpu_by_label["compute_stencil"]["sum"])
        self.assertAlmostEqual(fused_by_label[("JacobiIterationKernel", "GPU")]["sum"], gpu_by_label["JacobiIterationKernel"]["sum"])
        # ...but a DIFFERENT pct_total than each side's own standalone number,
        # proving genuine recombination happened (not the rejected no-op rescale).
        self.assertNotAlmostEqual(
            fused_by_label[("compute_stencil", "CPU")]["pct_total"],
            cpu_by_label["compute_stencil"]["pct_total"],
        )
        self.assertNotAlmostEqual(
            fused_by_label[("JacobiIterationKernel", "GPU")]["pct_total"],
            gpu_by_label["JacobiIterationKernel"]["pct_total"],
        )

    def test_fused_list_excludes_gpu_api_overhead_bucket(self):
        fused, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = combined.build_combined_view(CPU_DIR, GPU_DIR)
        fused_labels = {e["label"] for e in fused}
        self.assertNotIn("hipMemcpy", fused_labels)
        self.assertIn("hipMemcpy", {e["label"] for e in cpu_gpu_api_entries})

    def test_mismatched_pairing_combines_without_error_gigo(self):
        # Deliberately mismatched: single_rank (CPU) with the 2-rank GPU fixture.
        # No cross-validation should happen -- this must not raise.
        fused, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = combined.build_combined_view(CPU_DIR_SINGLE, GPU_DIR)
        self.assertTrue(fused)
        self.assertGreater(info["combined_total_sec"], 0)


class WriteReportTests(unittest.TestCase):
    def test_four_tables_present_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)
            self.assertTrue(os.path.isfile(dest))
            i1 = report.index("=== 1. Combined hotspots")
            i2 = report.index("=== 2. CPU compute hotspots")
            i3 = report.index("=== 3. GPU kernel hotspots")
            i4 = report.index("=== 4. GPU API / launch overhead")
            self.assertTrue(i1 < i2 < i3 < i4)

    def test_table2_matches_standalone_cpu_tool_output(self):
        import extract_CPU_hotspots as cpu_tool
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        cpu_entries, cpu_gpu_api_entries, cpu_scanned, cpu_total_raw = cpu_tool.aggregate(CPU_DIR)
        selected, _ = cpu_tool.select_entries(cpu_entries, cpu_total_raw)
        standalone_table = cpu_tool.format_table(selected)
        self.assertIn(standalone_table.strip(), report)

    def test_table3_matches_standalone_gpu_tool_output(self):
        import extract_GPU_hotspots as gpu_tool
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        gpu_entries, gpu_scanned, gpu_total_ns = gpu_tool.aggregate(GPU_DIR)
        selected, _ = gpu_tool.select_entries(gpu_entries, gpu_total_ns / 1e9)
        standalone_table = gpu_tool.format_table(selected)
        self.assertIn(standalone_table.strip(), report)

    def test_table4_matches_standalone_gpu_api_bucket(self):
        import extract_CPU_hotspots as cpu_tool
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        cpu_entries, cpu_gpu_api_entries, cpu_scanned, cpu_total_raw = cpu_tool.aggregate(CPU_DIR)
        selected, _ = cpu_tool.select_entries(cpu_gpu_api_entries, cpu_total_raw)
        standalone_table = cpu_tool.format_table(selected)
        self.assertIn(standalone_table.strip(), report)

    def test_header_labels_both_runs_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)
            self.assertIn("CPU run directory (rocprof-sys):", report)
            self.assertIn("GPU run directory (rocprofv3):", report)
            self.assertIn("not checked against each other", report)
            self.assertIn("executable: jacobi_mpi", report)  # from mpi_2rank's metadata.json
            self.assertIn("executable: jacobi_hip", report)  # from rocprofv3_mpi_2rank's config.json

    def test_mismatched_pairing_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR_SINGLE, GPU_DIR, dest)
            self.assertIn("=== 1. Combined hotspots", report)

    def test_raises_when_cpu_side_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                combined.write_report(CPU_DIR_EMPTY, GPU_DIR, dest)

    def test_raises_when_gpu_side_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                combined.write_report(CPU_DIR, GPU_DIR_EMPTY, dest)

    def test_selection_modes_apply_to_all_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest, show_all=True)
            self.assertIn("showing all", report)


if __name__ == "__main__":
    unittest.main()
