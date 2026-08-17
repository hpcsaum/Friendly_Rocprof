import importlib.util
import json
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

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

CPU_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_DIR = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
CPU_DIR_SINGLE = os.path.join(FIXTURES, "single_rank")
GPU_DIR_SINGLE = os.path.join(FIXTURES, "rocprofv3_single_rank")
CPU_DIR_EMPTY = os.path.join(FIXTURES, "no_timing_data")
GPU_DIR_EMPTY = os.path.join(FIXTURES, "rocprofv3_no_data")
CPU_DIR_DATED_SUBDIR = os.path.join(FIXTURES, "mpi_2rank_dated_subdir")
COMBINED_DIR = os.path.join(FIXTURES, "pop_combined_2rank")


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
        import stage4_rocprofsys_flat
        from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
        from stage5_table_render import render_table, select_entries
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        cpu_entries, _cpu_gpu_api_entries, _cpu_scanned, cpu_total_raw = stage4_rocprofsys_flat.aggregate(CPU_DIR)

        def _prepare(entries):
            for e in entries:
                e["pct_total"] = (e["self_sum"] / cpu_total_raw * 100.0) if cpu_total_raw > 0 else None

        selected, _ = select_entries(
            cpu_entries, rank_field="self_sum", threshold_field="pct_total", prepare=_prepare,
        )
        standalone_table = render_table(CPU_HOTSPOTS_COLUMNS, selected)
        self.assertIn(standalone_table.strip(), report)

    def test_table3_matches_standalone_gpu_tool_output(self):
        import stage4_rocprofv3
        from stage5_gpu_hotspots_table import GPU_HOTSPOTS_COLUMNS
        from stage5_table_render import render_table, select_entries
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        gpu_entries, _gpu_scanned, _gpu_total_ns = stage4_rocprofv3.aggregate(GPU_DIR)
        selected, _ = select_entries(gpu_entries, rank_field="sum", threshold_field="pct_total")
        standalone_table = render_table(GPU_HOTSPOTS_COLUMNS, selected)
        self.assertIn(standalone_table.strip(), report)

    def test_table4_matches_standalone_gpu_api_bucket(self):
        import stage4_rocprofsys_flat
        from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
        from stage5_table_render import render_table, select_entries
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        _cpu_entries, cpu_gpu_api_entries, _cpu_scanned, cpu_total_raw = stage4_rocprofsys_flat.aggregate(CPU_DIR)

        def _prepare(entries):
            for e in entries:
                e["pct_total"] = (e["self_sum"] / cpu_total_raw * 100.0) if cpu_total_raw > 0 else None

        selected, _ = select_entries(
            cpu_gpu_api_entries, rank_field="self_sum", threshold_field="pct_total", prepare=_prepare,
        )
        standalone_table = render_table(CPU_HOTSPOTS_COLUMNS, selected)
        self.assertIn(standalone_table.strip(), report)

    def test_header_shows_one_shared_metadata_block_cpu_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)
            self.assertIn("CPU run directory:", report)
            self.assertIn("GPU run directory:", report)
            self.assertIn("not checked against each other", report)
            # exactly one metadata block, not two -- CPU's own metadata.json wins over
            # rocprofv3_mpi_2rank's config.json when both are available
            self.assertEqual(report.count("executable:"), 1)
            self.assertIn("executable: jacobi_mpi", report)

    def test_header_falls_back_to_gpu_metadata_when_cpu_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            # single_rank has no metadata.json at all -- GPU side's config.json should be used instead
            report = combined.write_report(CPU_DIR_SINGLE, GPU_DIR, dest)
            self.assertIn("executable: jacobi_hip", report)

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

    def test_load_imbalance_tables_5_and_6_present_after_table_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)
            i4 = report.index("=== 4. GPU API / launch overhead")
            i5 = report.index("=== 5. CPU load imbalance")
            i6 = report.index("=== 6. GPU kernel load imbalance")
            self.assertTrue(i4 < i5 < i6)
            self.assertIn("compute_stencil", report[i5:i6])
            self.assertIn("JacobiIterationKernel", report[i6:])

    def test_load_imbalance_tables_match_standalone_tool_output(self):
        import stage4_rocprofsys_flat
        import stage4_rocprofv3
        from stage5_load_imbalance_table import compute_load_imbalance, load_imbalance_columns
        from stage5_table_render import render_table
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR, GPU_DIR, dest)

        cpu_per_rank, _ = stage4_rocprofsys_flat.aggregate_per_rank(CPU_DIR)
        cpu_selected, _ = compute_load_imbalance(cpu_per_rank)
        self.assertIn(render_table(load_imbalance_columns(), cpu_selected).strip(), report)

        gpu_per_rank, _ = stage4_rocprofv3.aggregate_per_rank(GPU_DIR)
        gpu_selected, _ = compute_load_imbalance(gpu_per_rank)
        self.assertIn(render_table(load_imbalance_columns(item_label="kernel"), gpu_selected).strip(), report)

    def test_load_imbalance_tables_skipped_on_single_rank_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR_SINGLE, GPU_DIR_SINGLE, dest)
            self.assertIn("CPU load imbalance across ranks (rocprof-sys run) -- skipped", report)
            self.assertIn("GPU kernel load imbalance across ranks (rocprofv3 run) -- skipped", report)

    def test_cpu_side_nested_in_dated_subdirectory_still_succeeds(self):
        # Reproduces the reported bug: rocprof-sys's default time-stamped output
        # subdirectory must not make the combined tool fail either.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = combined.write_report(CPU_DIR_DATED_SUBDIR, GPU_DIR, dest)
            self.assertIn("=== 1. Combined hotspots", report)
            self.assertIn("compute_stencil", report)


class MainCliTests(unittest.TestCase):
    def test_two_explicit_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            combined.main([CPU_DIR, GPU_DIR, "-o", dest])
            self.assertTrue(os.path.isfile(dest))

    def test_single_combined_directory_auto_resolves_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            combined.main([COMBINED_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
            self.assertIn(f"CPU run directory: {os.path.abspath(os.path.join(COMBINED_DIR, 'rocprof-sys'))}", report)
            self.assertIn(f"GPU run directory: {os.path.abspath(os.path.join(COMBINED_DIR, 'rocprofv3'))}", report)

    def test_single_directory_with_no_gpu_subdir_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            combined.main([CPU_DIR])

    def test_extra_noise_config_flag_excludes_a_configured_row(self):
        self.addCleanup(stage6_noise_config.configure, None)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["compute_stencil"]}}, f)
            combined.main([CPU_DIR, GPU_DIR, "-o", dest, "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("compute_stencil", report)


if __name__ == "__main__":
    unittest.main()
