import importlib.util
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_CPU_hotspots.py")

# extract_CPU_hotspots.py does a plain top-level "from stage1_rocprofsys import ...",
# relying on its own directory being on sys.path -- true automatically when run
# directly, but not when loaded here by explicit file path, so replicate that
# manually (same as test_extract_hotspots.py).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_CPU_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_CPU_hotspots"] = hotspots
spec.loader.exec_module(hotspots)

import stage4_rocprofsys_flat as flat  # noqa: E402  (needs sys.path insert above first)
import stage6_noise_config  # noqa: E402


class MetadataGuessingTests(unittest.TestCase):
    def test_guesses_from_mpi_fixture_metadata_json(self):
        data = hotspots.load_json_file(os.path.join(FIXTURES, "mpi_2rank"), hotspots.METADATA_FILENAME)
        self.assertEqual(hotspots.guess_executable(data, hotspots.EXECUTABLE_KEYS), "jacobi_mpi")
        self.assertEqual(
            hotspots.guess_run_datetime(data, hotspots.RUN_DATETIME_KEYS), "2026-07-21T07:40:00",
        )
        self.assertEqual(hotspots.guess_total_runtime(data, hotspots.TOTAL_RUNTIME_KEYS), "21.824161 sec")
        # world_size is nested under "settings" -- exercises the one-level-deep search
        self.assertEqual(hotspots.guess_num_ranks(data, hotspots.PID_SUFFIX_RE, [], keys=hotspots.NUM_RANKS_KEYS), 2)

    def test_missing_metadata_json_leaves_fields_blank(self):
        data = hotspots.load_json_file(os.path.join(FIXTURES, "single_rank"), hotspots.METADATA_FILENAME)
        self.assertEqual(data, {})
        self.assertIsNone(hotspots.guess_executable(data, hotspots.EXECUTABLE_KEYS))
        self.assertIsNone(hotspots.guess_total_runtime(data, hotspots.TOTAL_RUNTIME_KEYS))

    def test_num_ranks_falls_back_to_distinct_pids_in_filenames(self):
        scanned = ["/x/wall_clock-1001.txt", "/x/wall_clock-1002.txt", "/x/roctracer-1001.txt"]
        self.assertEqual(hotspots.guess_num_ranks({}, hotspots.PID_SUFFIX_RE, scanned, keys=hotspots.NUM_RANKS_KEYS), 2)

    def test_run_datetime_falls_back_to_output_dir_timestamp_pattern(self):
        output_dir = "/some/rocprof-sys-app-output/2025-01-21_07.40"
        self.assertEqual(
            hotspots.guess_run_datetime(
                {}, hotspots.RUN_DATETIME_KEYS, output_dir=output_dir, dir_pattern=hotspots.TIME_OUTPUT_DIR_RE,
            ),
            "2025-01-21_07.40",
        )

    def test_gather_run_info_end_to_end_with_metadata(self):
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "mpi_2rank"), [])
        self.assertEqual(info["executable"], "jacobi_mpi")
        self.assertEqual(info["run_datetime"], "2026-07-21T07:40:00")
        self.assertEqual(info["total_runtime"], "21.824161 sec")
        self.assertEqual(info["num_ranks"], 2)

    def test_gather_run_info_end_to_end_without_metadata(self):
        scanned = [os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")]
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "single_rank"), scanned)
        self.assertIsNone(info["executable"])
        self.assertIsNone(info["run_datetime"])
        self.assertIsNone(info["total_runtime"])
        self.assertEqual(info["num_ranks"], 1)  # one distinct pid (1234) among scanned files


class WriteReportTests(unittest.TestCase):
    def test_end_to_end_on_mpi_fixture_default_top20(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("compute_stencil", report)
            self.assertIn("showing top 20 of", report)
            self.assertIn("true GPU kernel execution time is not present", report)
            self.assertIn("scripts/profile_GPU_hotspots.sh", report)
            self.assertIn("executable: jacobi_mpi", report)
            self.assertIn("run date/time: 2026-07-21T07:40:00", report)
            self.assertIn("runtime: 21.824161 sec", report)
            self.assertIn("MPI ranks: 2", report)

    def test_end_to_end_on_single_rank_fixture_blank_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "single_rank"), dest)
            self.assertIn("executable: \n", report)
            self.assertIn("run date/time: \n", report)
            self.assertIn("runtime: \n", report)
            self.assertIn("MPI ranks: 1\n", report)

    def test_threshold_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, threshold=50.0)
            self.assertIn(">= 50% of total measured time", report)
            self.assertNotIn("apply_boundary", report)  # not present in this fixture anyway, sanity check

    def test_show_all_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, show_all=True)
            self.assertIn("showing all", report)

    def test_raises_when_no_timing_data_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            with self.assertRaises(SystemExit):
                hotspots.write_report(os.path.join(FIXTURES, "no_timing_data"), dest)

    def test_load_imbalance_table_present_after_hotspots_tables_on_mpi_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest)
            i_hotspots = report.index("CPU compute hotspots")
            i_gpu_api = report.index("GPU API / launch overhead")
            i_imbalance = report.index("CPU load imbalance across 2 ranks")
            self.assertTrue(i_hotspots < i_gpu_api < i_imbalance)
            self.assertIn("compute_stencil", report[i_imbalance:])

    def test_load_imbalance_table_skipped_on_single_rank_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "single_rank"), dest)
            self.assertIn("CPU load imbalance across ranks -- skipped: only 1 rank/file found", report)

    def test_default_ranks_by_self_time_compute_stencil_beats_main(self):
        # main has huge inclusive time but near-zero self time in this fixture --
        # the exact pollution this feature targets.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, show_all=True)
            self.assertIn("Ranked by self time", report)
            hotspots_section = report[report.index("=== 1."):report.index("=== 2.")]
            self.assertLess(hotspots_section.index("compute_stencil"), hotspots_section.index("main"))

    def test_unfiltered_ranks_by_inclusive_time_main_beats_compute_stencil(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, show_all=True, unfiltered=True)
            self.assertIn("Ranked by inclusive (total) time", report)
            hotspots_section = report[report.index("=== 1."):report.index("=== 2.")]
            self.assertLess(hotspots_section.index("main"), hotspots_section.index("compute_stencil"))


class NestedDatedSubdirectoryTests(unittest.TestCase):
    """rocprof-sys's default ROCPROFSYS_TIME_OUTPUT behavior nests every per-process file one
    level deeper, inside an auto-generated timestamped subdirectory (e.g. "2026-08-03_09.24/") --
    reported as a real bug against this fixture's real-world equivalent. Every scan in this
    module must find files there, not just directly under output_dir."""

    DIR = os.path.join(FIXTURES, "mpi_2rank_dated_subdir")

    def test_load_metadata_finds_nested_metadata_json(self):
        data = hotspots.load_json_file(self.DIR, hotspots.METADATA_FILENAME)
        self.assertEqual(hotspots.guess_executable(data, hotspots.EXECUTABLE_KEYS), "jacobi_mpi")
        self.assertEqual(hotspots.guess_num_ranks(data, hotspots.PID_SUFFIX_RE, [], keys=hotspots.NUM_RANKS_KEYS), 2)

    def test_guess_run_datetime_falls_back_to_nested_scanned_file_dirname(self):
        _cpu, _gpu, scanned, _total = flat.aggregate(self.DIR)
        data = hotspots.load_json_file(self.DIR, hotspots.METADATA_FILENAME)  # no start_time field in this fixture
        self.assertEqual(
            hotspots.guess_run_datetime(
                data, hotspots.RUN_DATETIME_KEYS, output_dir=self.DIR, scanned_files=scanned,
                dir_pattern=hotspots.TIME_OUTPUT_DIR_RE,
            ),
            "2026-08-03_09.24",
        )

    def test_write_report_succeeds_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(self.DIR, dest)
            self.assertIn("compute_stencil", report)
            self.assertIn("run date/time: 2026-08-03_09.24", report)
            self.assertIn("MPI ranks: 2", report)


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_extra_noise_config_flag_excludes_a_configured_row(self):
        # apply_boundary is a real, otherwise-untagged row in single_rank -- configuring it as
        # "other" and confirming it disappears from the written report proves --extra-noise-config
        # actually reaches stage3's tagging, end to end through main().
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["apply_boundary"]}}, f)
            hotspots.main([
                os.path.join(FIXTURES, "single_rank"), "-o", dest,
                "--extra-noise-config", config_path,
            ])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("apply_boundary", report)
        self.assertIn("compute_stencil", report)

    def test_friendly_rocprof_noise_config_env_var_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["apply_boundary"]}}, f)
            with unittest.mock.patch.dict(os.environ, {"FRIENDLY_ROCPROF_NOISE_CONFIG": config_path}):
                hotspots.main([os.path.join(FIXTURES, "single_rank"), "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("apply_boundary", report)


if __name__ == "__main__":
    unittest.main()
