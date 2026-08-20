import importlib.util
import json
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_sample_flat.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_sample_flat", MODULE_PATH)
flat = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_sample_flat"] = flat
spec.loader.exec_module(flat)

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)


class AggregateTests(unittest.TestCase):
    def test_single_rank_bucketing_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = flat.aggregate(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(len(scanned), 1)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil", "apply_boundary"})
        self.assertEqual(gpu_labels, {"hipLaunchKernel", "hipMemcpy"})
        # single file -> total_runtime is just that file's largest SUM ("main")
        self.assertAlmostEqual(total_runtime, 13.360265)
        by_label = {e["label"]: e for e in cpu}
        # self_sum = sum * (% SELF / 100); pct_total (aggregate()'s own default) is self-based
        self.assertAlmostEqual(by_label["compute_stencil"]["self_sum"], 9.812345 * 0.95)
        self.assertAlmostEqual(by_label["compute_stencil"]["pct_total"], 9.812345 * 0.95 / 13.360265 * 100)

    def test_rocprofsys_wrapper_noise_excluded_from_both_buckets(self):
        # rocprofsys_wrapper_noise/wall_clock-9001.txt: main(10.0), real
        # compute_stencil(8.0), plus gotcha_wrap(1.0)/lookup_hashtable(1.0)
        # wrapper-noise rows -- neither should appear in cpu_entries OR
        # gpu_entries; root_sum/total_runtime (from main, 10.0) is unaffected.
        cpu, gpu, scanned, total_runtime = flat.aggregate(
            os.path.join(FIXTURES, "rocprofsys_wrapper_noise")
        )
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil"})
        self.assertEqual(gpu_labels, set())
        self.assertAlmostEqual(total_runtime, 10.0)

    def test_mpi_spawned_thread_noise_excluded_from_both_buckets(self):
        # mpi_spawned_thread_noise/wall_clock-5001.txt: main -> MPI_Init ->
        # pthread_create -> start_thread(thread 1, 9.0s, 100% self) -- real
        # Cray MPICH shape (background thread spawned directly under
        # MPI_Init, not a GPU-API call). start_thread must be dropped
        # entirely; main/MPI_Init/pthread_create (real, or at least not
        # thread-root-under-MPI) stay; root_sum (10.0, from main) unaffected.
        cpu, gpu, scanned, total_runtime = flat.aggregate(
            os.path.join(FIXTURES, "mpi_spawned_thread_noise")
        )
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertNotIn("start_thread", cpu_labels)
        self.assertNotIn("start_thread", gpu_labels)
        self.assertEqual(cpu_labels, {"main", "MPI_Init", "pthread_create"})
        self.assertAlmostEqual(total_runtime, 10.0)

    def test_mpi_ranks_aggregate_by_function_name_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = flat.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in cpu}
        # 9.5 (rank0) + 9.4 (rank1)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 18.9)
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        gpu_by_label = {e["label"]: e for e in gpu}
        self.assertAlmostEqual(gpu_by_label["hipMemcpy"]["sum"], 0.0395)
        # total_runtime = rank0's main (10.924161) + rank1's main (10.900000)
        self.assertAlmostEqual(total_runtime, 21.824161)
        expected_self_sum = 9.5 * 0.94 + 9.4 * 0.935
        self.assertAlmostEqual(by_label["compute_stencil"]["self_sum"], expected_self_sum)
        self.assertAlmostEqual(by_label["compute_stencil"]["pct_total"], expected_self_sum / 21.824161 * 100)

    def test_main_has_near_zero_self_time_despite_huge_inclusive_time(self):
        # main's % SELF is 0.1 in this fixture (it just calls compute_stencil) --
        # this is the exact "pass-through wrapper" pollution this feature targets.
        cpu, _gpu, _scanned, _total = flat.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["main"]["sum"], 21.824161)  # huge inclusive time
        self.assertLess(by_label["main"]["self_sum"], 0.03)  # but almost no self time
        self.assertLess(by_label["main"]["self_sum"], by_label["compute_stencil"]["self_sum"])

    def test_directory_with_no_timing_files(self):
        cpu, gpu, scanned, total_runtime = flat.aggregate(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(scanned, [])
        self.assertEqual(cpu, [])
        self.assertEqual(gpu, [])
        self.assertEqual(total_runtime, 0)


class OtherTagConfigTests(unittest.TestCase):
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_other_tagged_row_excluded_same_as_a_built_in_noise_tag(self):
        # single_rank fixture: main/compute_stencil/apply_boundary are real CPU rows -- configure
        # a custom "other" pattern matching apply_boundary and confirm scan_ranks()'s drop
        # condition treats it exactly like wrapper_noise/compiler_runtime_noise already are.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_config.json")
            with open(path, "w") as f:
                json.dump({"add": {"other": ["apply_boundary"]}}, f)
            stage6_noise_config.configure(path)

            cpu, _gpu, _scanned, _total = flat.aggregate(os.path.join(FIXTURES, "single_rank"))
        cpu_labels = {e["label"] for e in cpu}
        self.assertNotIn("apply_boundary", cpu_labels)
        self.assertIn("compute_stencil", cpu_labels)  # unrelated row unaffected


class ScanRanksMultiMetricFileTests(unittest.TestCase):
    """rocprof-sys's default (sampling-enabled) config writes THREE per-rank
    text tables (wall_clock, sampling_wall_clock, sampling_cpu_clock) -- these
    exercise the fix that stops each metric-type file from being counted as
    its own rank, confirmed against a real HPC-generated directory that hit
    this exact bug (see DEVELOPMENT_HISTORY.md)."""

    DIR = os.path.join(FIXTURES, "multi_metric_rank")

    def test_aggregate_counts_two_ranks_not_six_files(self):
        cpu, _gpu, scanned, total_runtime = flat.aggregate(self.DIR)
        # 2 ranks x 2 included files each (wall_clock, sampling_wall_clock) --
        # sampling_cpu_clock is excluded outright, so never "scanned".
        self.assertEqual(len(scanned), 4)
        # total_runtime = rank0's own max (main=10.0) + rank1's own max (main=10.5),
        # NOT the sum of all 4 files' own maxes (which would double-count main).
        self.assertAlmostEqual(total_runtime, 20.5)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["main"]["sum"], 20.5)

    def test_wall_clock_wins_over_sampling_wall_clock_for_shared_label(self):
        cpu, _gpu, _scanned, _total = flat.aggregate(self.DIR)
        by_label = {e["label"]: e for e in cpu}
        # shared_func is in both wall_clock (2.0/2.1) and sampling_wall_clock
        # (2.5/2.6) per rank -- wall_clock's values must win outright, not sum.
        self.assertAlmostEqual(by_label["shared_func"]["sum"], 2.0 + 2.1)

    def test_sampling_wall_clock_used_when_label_only_there(self):
        cpu, _gpu, _scanned, _total = flat.aggregate(self.DIR)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["compute_A"]["sum"], 5.0 + 5.2)

    def test_total_runtime_not_corrupted_by_multi_thread_same_label_rows(self):
        # Each rank's wall_clock file has TWO raw rows labeled "worker_loop"
        # (mirrors real rocprof-sys output, where e.g. "start_thread" gets one
        # row per worker thread). Merged by label, worker_loop's summed value
        # (12.5/12.8 per rank) exceeds "main"'s own value (10.0/10.5) -- but
        # total_runtime must still be driven by main (the true root scope, by
        # raw per-row SUM before merging), not the merged worker_loop total.
        cpu, _gpu, _scanned, total_runtime = flat.aggregate(self.DIR)
        self.assertAlmostEqual(total_runtime, 20.5)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["worker_loop"]["sum"], 6.0 + 6.5 + 6.2 + 6.6)

    def test_sampling_cpu_clock_data_never_appears(self):
        cpu, gpu, _scanned, _total = flat.aggregate(self.DIR)
        all_labels = {e["label"] for e in cpu} | {e["label"] for e in gpu}
        self.assertNotIn("cpu_only_ghost", all_labels)

    def test_aggregate_per_rank_returns_two_dicts_not_six(self):
        per_rank, rank_keys = flat.aggregate_per_rank(self.DIR)
        self.assertEqual(len(rank_keys), 2)
        self.assertEqual(len(per_rank), 2)
        for ft in per_rank:
            self.assertNotIn("cpu_only_ghost", ft)
        shared_values = sorted(ft["shared_func"] for ft in per_rank)
        self.assertAlmostEqual(shared_values[0], 2.0)
        self.assertAlmostEqual(shared_values[1], 2.1)
        compute_a_values = sorted(ft["compute_A"] for ft in per_rank)
        self.assertAlmostEqual(compute_a_values[0], 5.0)
        self.assertAlmostEqual(compute_a_values[1], 5.2)


class RocrClassificationTests(unittest.TestCase):
    DIR = os.path.join(FIXTURES, "rocr_runtime_internals")

    def test_rocr_functions_land_in_gpu_bucket_not_cpu(self):
        cpu, gpu, _scanned, _total = flat.aggregate(self.DIR)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil"})
        self.assertIn("rocr::core::BusyWaitSignal::WaitAcquire(hsa_signal_condition_t, long)", gpu_labels)


class WrapperContaminatedBranchFixtureTests(unittest.TestCase):
    DIR = os.path.join(FIXTURES, "wrapper_contaminated_branch")

    def test_contaminated_branch_excluded_end_to_end(self):
        # wrapper_contaminated_branch/wall_clock-7001.txt: main -> [noise
        # branch topped by a generic std::pair<..._Rb_tree...>-style label,
        # with ANOTHER generic std::_Rb_tree<...>::_M_erase layer in between
        # it and get_library (two non-matching layers, matching the real
        # multi-layer chain confirmed via test_apps HPC data), MPI_Init
        # (clean), run_simulation (clean)]. The WHOLE noise branch --
        # including the intermediate generic layer, which doesn't match
        # anything on its own either -- must vanish, not just its top label.
        cpu, gpu, _scanned, _total = flat.aggregate(self.DIR)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        noise_labels = {
            "std::pair<std::_Rb_tree_iterator<int>, bool> noise_top",
            "std::_Rb_tree<unsigned long, std::pair<unsigned long const, std::set<unsigned long>>>::_M_erase",
            "get_library",
        }
        for label in noise_labels:
            self.assertNotIn(label, cpu_labels)
            self.assertNotIn(label, gpu_labels)
        self.assertIn("main", cpu_labels)
        self.assertIn("MPI_Init", cpu_labels)
        self.assertIn("run_simulation", cpu_labels)


class GpuSpawnedThreadFixtureTests(unittest.TestCase):
    DIR = os.path.join(FIXTURES, "gpu_spawned_thread")

    def test_hip_spawned_thread_lands_in_gpu_bucket(self):
        cpu, gpu, _scanned, _total = flat.aggregate(self.DIR)
        gpu_labels = {e["label"] for e in gpu}
        self.assertIn("start_thread", gpu_labels)
        by_label = {e["label"]: e for e in gpu}
        self.assertAlmostEqual(by_label["start_thread"]["self_sum"], 18.0)

    def test_app_spawned_thread_stays_in_cpu_bucket(self):
        # Both start_thread rows share the exact same label -- they must NOT be
        # merged together across buckets; the app-spawned one (12.0s, no GPU
        # ancestor) has to be told apart from the HIP-spawned one (18.0s).
        cpu, gpu, _scanned, _total = flat.aggregate(self.DIR)
        cpu_labels = {e["label"] for e in cpu}
        self.assertIn("start_thread", cpu_labels)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["start_thread"]["self_sum"], 12.0)


class AggregatePerRankTests(unittest.TestCase):
    def test_single_rank_returns_one_dict(self):
        per_file, scanned = flat.aggregate_per_rank(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(per_file), 1)
        self.assertEqual(set(per_file[0]), {"main", "compute_stencil", "apply_boundary"})
        self.assertNotIn("hipLaunchKernel", per_file[0])
        self.assertNotIn("hipMemcpy", per_file[0])

    def test_mpi_2rank_keeps_ranks_separate(self):
        # default (self-time) values: compute_stencil is 9.5*0.94 / 9.4*0.935, not the raw sum
        per_file, scanned = flat.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(per_file), 2)
        values = sorted(ft["compute_stencil"] for ft in per_file)
        self.assertAlmostEqual(values[0], 9.4 * 0.935)
        self.assertAlmostEqual(values[1], 9.5 * 0.94)
        # GPU-API bucket excluded per-rank the same way aggregate() excludes it globally
        for ft in per_file:
            self.assertNotIn("hipMemcpy", ft)

    def test_mpi_2rank_unfiltered_uses_inclusive_time(self):
        per_file, _scanned = flat.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"), unfiltered=True)
        values = sorted(ft["compute_stencil"] for ft in per_file)
        self.assertAlmostEqual(values[0], 9.4)
        self.assertAlmostEqual(values[1], 9.5)

    def test_no_timing_data_returns_empty(self):
        per_file, scanned = flat.aggregate_per_rank(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(per_file, [])
        self.assertEqual(scanned, [])


class NestedDatedSubdirectoryTests(unittest.TestCase):
    """rocprof-sys's default ROCPROFSYS_TIME_OUTPUT behavior nests every per-process file one
    level deeper, inside an auto-generated timestamped subdirectory (e.g. "2026-08-03_09.24/") --
    reported as a real bug against this fixture's real-world equivalent. Every scan in this
    module must find files there, not just directly under output_dir."""

    DIR = os.path.join(FIXTURES, "mpi_2rank_dated_subdir")

    def test_aggregate_finds_files_one_level_down(self):
        cpu, gpu, scanned, total_runtime = flat.aggregate(self.DIR)
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 18.9)
        self.assertAlmostEqual(total_runtime, 21.824161)

    def test_aggregate_per_rank_finds_files_one_level_down(self):
        per_file, scanned = flat.aggregate_per_rank(self.DIR)
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(per_file), 2)


if __name__ == "__main__":
    unittest.main()
