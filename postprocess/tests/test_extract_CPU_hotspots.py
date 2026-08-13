import importlib.util
import os
import shutil
import statistics
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "extract_CPU_hotspots.py")

spec = importlib.util.spec_from_file_location("extract_CPU_hotspots", MODULE_PATH)
hotspots = importlib.util.module_from_spec(spec)
sys.modules["extract_CPU_hotspots"] = hotspots
spec.loader.exec_module(hotspots)


class CleanLabelTests(unittest.TestCase):
    def test_single_rank_no_indent(self):
        self.assertEqual(hotspots.clean_label("00>>>main"), "main")

    def test_single_rank_with_indent(self):
        self.assertEqual(hotspots.clean_label("00>>>|_compute_stencil"), "compute_stencil")

    def test_mpi_rank_prefix_no_indent(self):
        self.assertEqual(hotspots.clean_label("00|00>>>main"), "main")

    def test_mpi_rank_prefix_with_indent(self):
        self.assertEqual(hotspots.clean_label("00|00>>>|_compute_stencil"), "compute_stencil")


class ParseTableFileTests(unittest.TestCase):
    def test_flat_single_rank_file(self):
        path = os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")
        rows = hotspots.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "apply_boundary", "hipLaunchKernel", "hipMemcpy"})
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["compute_stencil"]["count"], 1000)
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 9.812345)

    def test_hierarchical_mpi_rank_file(self):
        path = os.path.join(FIXTURES, "mpi_2rank", "wall_clock-2001.txt")
        rows = hotspots.parse_table_file(path)
        self.assertIsNotNone(rows)
        labels = {r["label"] for r in rows}
        self.assertEqual(labels, {"main", "compute_stencil", "hipMemcpy"})

    def test_non_timing_file_returns_none(self):
        path = os.path.join(FIXTURES, "no_timing_data", "available.txt")
        self.assertIsNone(hotspots.parse_table_file(path))


class IsGpuEntryTests(unittest.TestCase):
    def test_hip_prefix_is_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("hipLaunchKernel", "wall_clock-1.txt"))

    def test_hsa_prefix_is_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("hsa_amd_memory_pool_allocate", "wall_clock-1.txt"))

    def test_plain_function_is_cpu(self):
        self.assertFalse(hotspots.is_gpu_entry("compute_stencil", "wall_clock-1.txt"))

    def test_roctracer_filename_forces_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry("some_wrapped_call", "roctracer-1.txt"))

    def test_rocr_prefix_is_gpu(self):
        self.assertTrue(hotspots.is_gpu_entry(
            "rocr::core::BusyWaitSignal::WaitAcquire(hsa_signal_condition_t, long)", "wall_clock-1.txt"
        ))
        self.assertTrue(hotspots.is_gpu_entry("rocr::os::ThreadTrampoline(void*)", "wall_clock-1.txt"))

    def test_gpu_kernel_jit_compilation_noise_is_gpu(self):
        # clang/LLVM frontend + comgr compiling GPU machine code on first
        # kernel launch -- confirmed misclassified as CPU compute in real
        # test_apps HPC data's hotspots.txt before this fix.
        self.assertTrue(hotspots.is_gpu_entry(
            "clang::CodeGen::mergeDefaultFunctionDefinition(...)", "wall_clock-1.txt"
        ))
        self.assertTrue(hotspots.is_gpu_entry("amd_comgr_iterate_map_metadata", "wall_clock-1.txt"))
        # Return-type-prefixed form -- exercises substring, not startswith,
        # matching (real observed label from test_apps HPC data).
        self.assertTrue(hotspots.is_gpu_entry("int llvm::array_pod_sort_by_key(...)", "wall_clock-1.txt"))

    def test_plain_cpu_application_label_unaffected(self):
        self.assertFalse(hotspots.is_gpu_entry("run_simulation", "wall_clock-1.txt"))


class IsRocprofsysWrapperNoiseTests(unittest.TestCase):
    # Shared with extract_calltree.py's own wrapper-splice tier (see
    # ROCPROFSYS_WRAPPER_SUBSTRINGS's comment) -- this is the same
    # classification, now usable here too so it isn't left unfiltered in
    # aggregate()'s output the way it used to be.
    def test_matches_gotcha_and_dynamic_linker_frames(self):
        self.assertTrue(hotspots.is_rocprofsys_wrapper_noise("gotcha_wrap"))
        self.assertTrue(hotspots.is_rocprofsys_wrapper_noise("lookup_hashtable"))
        self.assertTrue(hotspots.is_rocprofsys_wrapper_noise("lookup.constprop.0"))
        self.assertTrue(hotspots.is_rocprofsys_wrapper_noise("tim::component::gotcha<101ul>::wrap"))

    def test_rejects_real_application_label(self):
        self.assertFalse(hotspots.is_rocprofsys_wrapper_noise("compute_stencil"))


class AggregateTests(unittest.TestCase):
    def test_single_rank_bucketing_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "single_rank"))
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
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(
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
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(
            os.path.join(FIXTURES, "mpi_spawned_thread_noise")
        )
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertNotIn("start_thread", cpu_labels)
        self.assertNotIn("start_thread", gpu_labels)
        self.assertEqual(cpu_labels, {"main", "MPI_Init", "pthread_create"})
        self.assertAlmostEqual(total_runtime, 10.0)

    def test_mpi_ranks_aggregate_by_function_name_and_total_runtime(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
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
        cpu, _gpu, _scanned, _total = hotspots.aggregate(os.path.join(FIXTURES, "mpi_2rank"))
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["main"]["sum"], 21.824161)  # huge inclusive time
        self.assertLess(by_label["main"]["self_sum"], 0.03)  # but almost no self time
        self.assertLess(by_label["main"]["self_sum"], by_label["compute_stencil"]["self_sum"])

    def test_directory_with_no_timing_files(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(scanned, [])
        self.assertEqual(cpu, [])
        self.assertEqual(gpu, [])
        self.assertEqual(total_runtime, 0)


class ScanRanksMultiMetricFileTests(unittest.TestCase):
    """rocprof-sys's default (sampling-enabled) config writes THREE per-rank
    text tables (wall_clock, sampling_wall_clock, sampling_cpu_clock) -- these
    exercise the fix that stops each metric-type file from being counted as
    its own rank, confirmed against a real HPC-generated directory that hit
    this exact bug (see DEVELOPMENT_HISTORY.md)."""

    DIR = os.path.join(FIXTURES, "multi_metric_rank")

    def test_aggregate_counts_two_ranks_not_six_files(self):
        cpu, _gpu, scanned, total_runtime = hotspots.aggregate(self.DIR)
        # 2 ranks x 2 included files each (wall_clock, sampling_wall_clock) --
        # sampling_cpu_clock is excluded outright, so never "scanned".
        self.assertEqual(len(scanned), 4)
        # total_runtime = rank0's own max (main=10.0) + rank1's own max (main=10.5),
        # NOT the sum of all 4 files' own maxes (which would double-count main).
        self.assertAlmostEqual(total_runtime, 20.5)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["main"]["sum"], 20.5)

    def test_wall_clock_wins_over_sampling_wall_clock_for_shared_label(self):
        cpu, _gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        by_label = {e["label"]: e for e in cpu}
        # shared_func is in both wall_clock (2.0/2.1) and sampling_wall_clock
        # (2.5/2.6) per rank -- wall_clock's values must win outright, not sum.
        self.assertAlmostEqual(by_label["shared_func"]["sum"], 2.0 + 2.1)

    def test_sampling_wall_clock_used_when_label_only_there(self):
        cpu, _gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["compute_A"]["sum"], 5.0 + 5.2)

    def test_total_runtime_not_corrupted_by_multi_thread_same_label_rows(self):
        # Each rank's wall_clock file has TWO raw rows labeled "worker_loop"
        # (mirrors real rocprof-sys output, where e.g. "start_thread" gets one
        # row per worker thread). Merged by label, worker_loop's summed value
        # (12.5/12.8 per rank) exceeds "main"'s own value (10.0/10.5) -- but
        # total_runtime must still be driven by main (the true root scope, by
        # raw per-row SUM before merging), not the merged worker_loop total.
        cpu, _gpu, _scanned, total_runtime = hotspots.aggregate(self.DIR)
        self.assertAlmostEqual(total_runtime, 20.5)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["worker_loop"]["sum"], 6.0 + 6.5 + 6.2 + 6.6)

    def test_sampling_cpu_clock_data_never_appears(self):
        cpu, gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        all_labels = {e["label"] for e in cpu} | {e["label"] for e in gpu}
        self.assertNotIn("cpu_only_ghost", all_labels)

    def test_aggregate_per_rank_returns_two_dicts_not_six(self):
        per_rank, rank_keys = hotspots.aggregate_per_rank(self.DIR)
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
        cpu, gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        cpu_labels = {e["label"] for e in cpu}
        gpu_labels = {e["label"] for e in gpu}
        self.assertEqual(cpu_labels, {"main", "compute_stencil"})
        self.assertIn("rocr::core::BusyWaitSignal::WaitAcquire(hsa_signal_condition_t, long)", gpu_labels)


class AttachAncestryTests(unittest.TestCase):
    def test_chain_within_one_thread(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 1, "thread_id": "0"},
            {"label": "c", "depth": 2, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertIsNone(rows[0]["parent"])
        self.assertFalse(rows[0]["is_thread_root"])
        self.assertIs(rows[1]["parent"], rows[0])
        self.assertFalse(rows[1]["is_thread_root"])
        self.assertIs(rows[2]["parent"], rows[1])
        self.assertFalse(rows[2]["is_thread_root"])

    def test_sibling_depth_zero_rows_have_no_parent(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 0, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertIsNone(rows[1]["parent"])
        self.assertFalse(rows[1]["is_thread_root"])

    def test_thread_change_relative_to_parent_flags_thread_root(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 1, "thread_id": "0"},
            {"label": "c", "depth": 2, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertIs(rows[2]["parent"], rows[1])
        self.assertTrue(rows[2]["is_thread_root"])

    def test_same_thread_as_parent_is_not_a_thread_root(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 1, "thread_id": "0"},
            {"label": "c", "depth": 2, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertFalse(rows[2]["is_thread_root"])


class ClassifyGpuTests(unittest.TestCase):
    def test_thread_root_with_gpu_ancestor_classifies_as_gpu(self):
        rows = [
            {"label": "hipRuntimeGetVersion", "depth": 0, "thread_id": "0"},
            {"label": "pthread_create", "depth": 1, "thread_id": "0"},
            {"label": "start_thread", "depth": 2, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertTrue(hotspots.classify_gpu(rows[2], "wall_clock-0.txt"))

    def test_thread_root_with_no_gpu_ancestor_stays_cpu(self):
        rows = [
            {"label": "compute_stencil", "depth": 0, "thread_id": "0"},
            {"label": "pthread_create", "depth": 1, "thread_id": "0"},
            {"label": "start_thread", "depth": 2, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertFalse(hotspots.classify_gpu(rows[2], "wall_clock-0.txt"))

    def test_non_thread_root_row_ignores_gpu_ancestor(self):
        # Only a thread-root node's OWN classification can come from ancestry --
        # a normal (non-thread-root) row is classified purely by its own label,
        # regardless of what its ancestors look like.
        rows = [
            {"label": "hipRuntimeGetVersion", "depth": 0, "thread_id": "0"},
            {"label": "some_cpu_helper", "depth": 1, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertFalse(rows[1]["is_thread_root"])
        self.assertFalse(hotspots.classify_gpu(rows[1], "wall_clock-0.txt"))

    def test_own_label_match_wins_outright(self):
        rows = [{"label": "hipLaunchKernel", "depth": 0, "thread_id": "0"}]
        hotspots.attach_ancestry(rows)
        self.assertTrue(hotspots.classify_gpu(rows[0], "wall_clock-0.txt"))


class IsRuntimeThreadNoiseTests(unittest.TestCase):
    # Same ancestry idea as ClassifyGpuTests, generalized to a second real
    # shape found in test_apps HPC data: Cray MPICH spawning pthread_create
    # directly under MPI_Init (not under a GPU-API call), each resulting
    # start_thread living the whole run at 100% self -- confirmed via real
    # wall_clock-0.txt: MPI_Init -> pthread_create -> start_thread (x3).
    def test_thread_root_with_mpi_ancestor_is_dropped(self):
        rows = [
            {"label": "MPI_Init", "depth": 0, "thread_id": "0"},
            {"label": "pthread_create", "depth": 1, "thread_id": "0"},
            {"label": "start_thread", "depth": 2, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertTrue(hotspots.is_runtime_thread_noise(rows[2]))

    def test_thread_root_with_no_mpi_ancestor_stays_untouched(self):
        rows = [
            {"label": "compute_stencil", "depth": 0, "thread_id": "0"},
            {"label": "pthread_create", "depth": 1, "thread_id": "0"},
            {"label": "start_thread", "depth": 2, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertFalse(hotspots.is_runtime_thread_noise(rows[2]))

    def test_non_thread_root_row_ignores_mpi_ancestor(self):
        rows = [
            {"label": "MPI_Init", "depth": 0, "thread_id": "0"},
            {"label": "some_cpu_helper", "depth": 1, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        self.assertFalse(rows[1]["is_thread_root"])
        self.assertFalse(hotspots.is_runtime_thread_noise(rows[1]))


class IsMpiTerritoryTests(unittest.TestCase):
    def test_matches_mpich_and_open_mpi_prefixes(self):
        self.assertTrue(hotspots.is_mpi_territory("MPI_Init"))
        self.assertTrue(hotspots.is_mpi_territory("PMPI_Allreduce"))
        self.assertTrue(hotspots.is_mpi_territory("MPIDI_CRAY_Setup_Shared_Mem_Coll"))
        self.assertTrue(hotspots.is_mpi_territory("ompi_request_complete"))

    def test_rejects_real_application_label(self):
        self.assertFalse(hotspots.is_mpi_territory("compute_stencil"))


class MarkWrapperContaminatedBranchesTests(unittest.TestCase):
    # Real shape confirmed via test_apps HPC data: main branches into a
    # dedicated, self-contained rocprof-sys/GOTCHA startup-bookkeeping
    # branch (mostly generic std::set<string>/std::map<ulong,set<ulong>>
    # container internals that don't match anything on their own -- only
    # get_library/create_hashtable/etc. do, deep inside) alongside real
    # branches like run_simulation.
    def test_contaminated_sibling_marked_clean_sibling_untouched(self):
        rows = [
            {"label": "main", "depth": 0, "thread_id": "0"},
            {"label": "std::pair<std::_Rb_tree_iterator<int>, bool> ...", "depth": 1, "thread_id": "0"},
            {"label": "get_library", "depth": 2, "thread_id": "0"},
            {"label": "run_simulation", "depth": 1, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        hotspots.mark_wrapper_contaminated_branches(rows)
        self.assertTrue(rows[1].get("wrapper_branch_noise"))
        self.assertFalse(rows[3].get("wrapper_branch_noise"))

    def test_linear_ancestor_wrapper_chain_is_not_nuked(self):
        # __libc_start_main -> rocprofsys_main -> main: a linear chain with
        # NO siblings at any step. Even though it "contains" a wrapper match
        # (rocprofsys_main itself), main must survive -- this shape is
        # splice_out_wrapper_nodes()'s job in extract_calltree.py (real code
        # sits inside the wrapper), not this function's, and this function
        # must never wholesale-delete it.
        rows = [
            {"label": "__libc_start_main", "depth": 0, "thread_id": "0"},
            {"label": "rocprofsys_main", "depth": 1, "thread_id": "0"},
            {"label": "main", "depth": 2, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        hotspots.mark_wrapper_contaminated_branches(rows)
        for row in rows:
            self.assertFalse(row.get("wrapper_branch_noise"), row["label"])

    def test_multiple_roots_noise_root_marked_real_root_untouched(self):
        rows = [
            {"label": "std::pair<std::_Rb_tree_iterator<int>, bool> noise_root", "depth": 0, "thread_id": "0"},
            {"label": "get_library", "depth": 1, "thread_id": "0"},
            {"label": "main", "depth": 0, "thread_id": "1"},
        ]
        hotspots.attach_ancestry(rows)
        hotspots.mark_wrapper_contaminated_branches(rows)
        self.assertTrue(rows[0].get("wrapper_branch_noise"))
        self.assertTrue(rows[1].get("wrapper_branch_noise"))
        self.assertFalse(rows[2].get("wrapper_branch_noise"))

    def test_directly_matching_sibling_with_real_content_is_not_touched(self):
        # gotcha_wrapper_call itself matches directly -- that's a plain
        # is_rocprofsys_wrapper_noise() exclusion (own row only), NOT this
        # function's job. Its real child underneath must be untouched: this
        # function only targets a sibling whose OWN label does NOT match but
        # has a match buried inside it (the opposite shape).
        rows = [
            {"label": "main", "depth": 0, "thread_id": "0"},
            {"label": "gotcha_wrapper_call", "depth": 1, "thread_id": "0"},
            {"label": "real_child_under_wrapper", "depth": 2, "thread_id": "0"},
            {"label": "run_simulation", "depth": 1, "thread_id": "0"},
        ]
        hotspots.attach_ancestry(rows)
        hotspots.mark_wrapper_contaminated_branches(rows)
        for row in rows:
            self.assertFalse(row.get("wrapper_branch_noise"), row["label"])


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
        cpu, gpu, _scanned, _total = hotspots.aggregate(self.DIR)
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
        cpu, gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        gpu_labels = {e["label"] for e in gpu}
        self.assertIn("start_thread", gpu_labels)
        by_label = {e["label"]: e for e in gpu}
        self.assertAlmostEqual(by_label["start_thread"]["self_sum"], 18.0)

    def test_app_spawned_thread_stays_in_cpu_bucket(self):
        # Both start_thread rows share the exact same label -- they must NOT be
        # merged together across buckets; the app-spawned one (12.0s, no GPU
        # ancestor) has to be told apart from the HIP-spawned one (18.0s).
        cpu, gpu, _scanned, _total = hotspots.aggregate(self.DIR)
        cpu_labels = {e["label"] for e in cpu}
        self.assertIn("start_thread", cpu_labels)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["start_thread"]["self_sum"], 12.0)


class SelectEntriesTests(unittest.TestCase):
    # self_sum deliberately diverges from sum for "a" so rank_by="self" vs
    # "inclusive" pick different top entries (see the two ordering tests below).
    ENTRIES = [
        {"label": "a", "count": 1, "sum": 9.0, "self_sum": 1.0},
        {"label": "b", "count": 1, "sum": 5.0, "self_sum": 5.0},
        {"label": "c", "count": 1, "sum": 3.0, "self_sum": 3.0},
    ]

    def test_default_ranks_by_self_time(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c", "a"])
        self.assertIn("top 20", desc)

    def test_rank_by_inclusive_reverses_a_and_b(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, rank_by="inclusive")
        self.assertEqual([e["label"] for e in selected], ["a", "b", "c"])

    def test_top_n_truncates(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, top=2)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn("top 2 of 3", desc)

    def test_threshold_filters_by_pct_total(self):
        # by self_sum: b=50%, c=30%, a=10% of total_runtime=10
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, threshold=30.0)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn(">= 30% of total runtime (2 of 3 entries)", desc)

    def test_threshold_with_unknown_total_runtime_falls_back_to_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=0, threshold=30.0)
        self.assertEqual(len(selected), 3)
        self.assertIn("total runtime unknown, threshold ignored", desc)

    def test_show_all(self):
        selected, desc = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, show_all=True)
        self.assertEqual(len(selected), 3)
        self.assertIn("all 3 entries", desc)

    def test_pct_total_reflects_rank_by_metric(self):
        selected, _ = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, show_all=True)
        by_label = {e["label"]: e for e in selected}
        self.assertAlmostEqual(by_label["a"]["pct_total"], 10.0)  # self_sum-based: 1.0/10*100
        selected, _ = hotspots.select_entries(self.ENTRIES, total_runtime=10.0, show_all=True, rank_by="inclusive")
        by_label = {e["label"]: e for e in selected}
        self.assertAlmostEqual(by_label["a"]["pct_total"], 90.0)  # sum-based: 9.0/10*100


class FormatTableTests(unittest.TestCase):
    def test_includes_self_and_pct_total_columns(self):
        entries = [{"label": "a", "count": 1, "sum": 4.0, "self_sum": 1.0, "pct_self": 25.0, "pct_total": 25.0}]
        table = hotspots.format_table(entries)
        self.assertIn("self(s)", table)
        self.assertIn("%total", table)
        self.assertIn("25.0", table)
        self.assertIn("1.000000", table)  # self_sum
        self.assertIn("4.000000", table)  # sum (total(s))

    def test_pct_total_none_renders_as_na(self):
        entries = [{"label": "a", "count": 1, "sum": 1.0, "self_sum": 0.1, "pct_self": 10.0, "pct_total": None}]
        table = hotspots.format_table(entries)
        self.assertIn("n/a", table)

    def test_empty_entries(self):
        self.assertIn("none found", hotspots.format_table([]))


class MetadataGuessingTests(unittest.TestCase):
    def test_guesses_from_mpi_fixture_metadata_json(self):
        metadata = hotspots.load_metadata(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(hotspots.guess_executable(metadata), "jacobi_mpi")
        self.assertEqual(hotspots.guess_run_datetime(metadata, "irrelevant"), "2026-07-21T07:40:00")
        self.assertEqual(hotspots.guess_total_runtime(metadata), "21.824161 sec")
        # world_size is nested under "settings" -- exercises the one-level-deep search
        self.assertEqual(hotspots.guess_num_ranks(metadata, []), 2)

    def test_missing_metadata_json_leaves_fields_blank(self):
        metadata = hotspots.load_metadata(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(metadata, {})
        self.assertIsNone(hotspots.guess_executable(metadata))
        self.assertIsNone(hotspots.guess_total_runtime(metadata))

    def test_num_ranks_falls_back_to_distinct_pids_in_filenames(self):
        metadata = {}
        scanned = ["/x/wall_clock-1001.txt", "/x/wall_clock-1002.txt", "/x/roctracer-1001.txt"]
        self.assertEqual(hotspots.guess_num_ranks(metadata, scanned), 2)

    def test_run_datetime_falls_back_to_output_dir_timestamp_pattern(self):
        metadata = {}
        output_dir = "/some/rocprof-sys-app-output/2025-01-21_07.40"
        self.assertEqual(hotspots.guess_run_datetime(metadata, output_dir), "2025-01-21_07.40")

    def test_gather_run_info_end_to_end_with_metadata(self):
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "mpi_2rank"), [])
        self.assertEqual(info["executable"], "jacobi_mpi")
        self.assertEqual(info["num_ranks"], 2)

    def test_gather_run_info_end_to_end_without_metadata(self):
        scanned = [os.path.join(FIXTURES, "single_rank", "wall_clock-1234.txt")]
        info = hotspots.gather_run_info(os.path.join(FIXTURES, "single_rank"), scanned)
        self.assertIsNone(info["executable"])
        self.assertIsNone(info["run_datetime"])
        self.assertIsNone(info["total_runtime"])
        self.assertEqual(info["num_ranks"], 1)  # one distinct pid (1234) among scanned files


class AggregatePerRankTests(unittest.TestCase):
    def test_single_rank_returns_one_dict(self):
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "single_rank"))
        self.assertEqual(len(scanned), 1)
        self.assertEqual(len(per_file), 1)
        self.assertEqual(set(per_file[0]), {"main", "compute_stencil", "apply_boundary"})
        self.assertNotIn("hipLaunchKernel", per_file[0])
        self.assertNotIn("hipMemcpy", per_file[0])

    def test_mpi_2rank_keeps_ranks_separate(self):
        # default (self-time) values: compute_stencil is 9.5*0.94 / 9.4*0.935, not the raw sum
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"))
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(per_file), 2)
        values = sorted(ft["compute_stencil"] for ft in per_file)
        self.assertAlmostEqual(values[0], 9.4 * 0.935)
        self.assertAlmostEqual(values[1], 9.5 * 0.94)
        # GPU-API bucket excluded per-rank the same way aggregate() excludes it globally
        for ft in per_file:
            self.assertNotIn("hipMemcpy", ft)

    def test_mpi_2rank_unfiltered_uses_inclusive_time(self):
        per_file, _scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"), unfiltered=True)
        values = sorted(ft["compute_stencil"] for ft in per_file)
        self.assertAlmostEqual(values[0], 9.4)
        self.assertAlmostEqual(values[1], 9.5)

    def test_no_timing_data_returns_empty(self):
        per_file, scanned = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "no_timing_data"))
        self.assertEqual(per_file, [])
        self.assertEqual(scanned, [])


class ComputeLoadImbalanceTests(unittest.TestCase):
    def test_matches_independently_computed_statistics_on_real_fixture(self):
        per_file, _ = hotspots.aggregate_per_rank(os.path.join(FIXTURES, "mpi_2rank"))
        selected, _ = hotspots.compute_load_imbalance(per_file, show_all=True)
        by_label = {e["label"]: e for e in selected}

        cs_values = [ft["compute_stencil"] for ft in per_file]
        self.assertAlmostEqual(by_label["compute_stencil"]["avg"], statistics.mean(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["std_dev"], statistics.pstdev(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["min"], min(cs_values))
        self.assertAlmostEqual(by_label["compute_stencil"]["max"], max(cs_values))

        main_values = [ft["main"] for ft in per_file]
        self.assertAlmostEqual(by_label["main"]["std_dev"], statistics.pstdev(main_values))

        # compute_stencil varies more (9.4 vs 9.5) than main (10.9 vs 10.924161)
        # in absolute terms -- confirm it sorts first when ranked by std_dev.
        self.assertGreater(by_label["compute_stencil"]["std_dev"], by_label["main"]["std_dev"])
        sorted_labels = [e["label"] for e in selected]
        self.assertEqual(sorted_labels.index("compute_stencil"), 0)

    def test_missing_rank_scores_zero_not_omitted(self):
        per_file_totals = [{"only_on_rank0": 10.0}, {}]
        selected, _ = hotspots.compute_load_imbalance(per_file_totals, show_all=True)
        entry = next(e for e in selected if e["label"] == "only_on_rank0")
        self.assertAlmostEqual(entry["avg"], 5.0)
        self.assertAlmostEqual(entry["std_dev"], 5.0)
        self.assertAlmostEqual(entry["min"], 0.0)
        self.assertAlmostEqual(entry["max"], 10.0)

    def test_top_n_selects_highest_std_dev(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        selected, desc = hotspots.compute_load_imbalance(per_file_totals, top=1)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("top 1 of 3", desc)

    def test_threshold_is_coefficient_of_variation(self):
        per_file_totals = [
            {"a": 10.0, "b": 5.0, "c": 100.0},
            {"a": 10.0, "b": 15.0, "c": 100.0},
        ]
        # b: avg=10, std_dev=5 -> cv=50%; a and c: std_dev=0 -> cv=0%
        selected, desc = hotspots.compute_load_imbalance(per_file_totals, threshold=10.0)
        self.assertEqual([e["label"] for e in selected], ["b"])
        self.assertIn("coefficient of variation", desc)

    def test_show_all(self):
        per_file_totals = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
        selected, desc = hotspots.compute_load_imbalance(per_file_totals, show_all=True)
        self.assertEqual(len(selected), 1)
        self.assertIn("all 1 entries", desc)


class FormatTableLoadImbalanceTests(unittest.TestCase):
    def test_includes_expected_columns(self):
        entries = [{"label": "foo", "avg": 1.0, "std_dev": 0.5, "min": 0.5, "max": 1.5, "cv_pct": 50.0}]
        table = hotspots.format_table_load_imbalance(entries)
        self.assertIn("avg(s)", table)
        self.assertIn("std_dev", table)
        self.assertIn("min(s)", table)
        self.assertIn("max(s)", table)
        self.assertIn("foo", table)

    def test_empty_entries(self):
        self.assertIn("none found", hotspots.format_table_load_imbalance([]))


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
            self.assertIn("total runtime: 21.824161 sec", report)
            self.assertIn("MPI ranks: 2", report)

    def test_end_to_end_on_single_rank_fixture_blank_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "single_rank"), dest)
            self.assertIn("executable: \n", report)
            self.assertIn("run date/time: \n", report)
            self.assertIn("total runtime: \n", report)
            self.assertIn("MPI ranks: 1\n", report)

    def test_threshold_selection_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, threshold=50.0)
            self.assertIn(">= 50% of total runtime", report)
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
            hotspots_section = report[report.index("CPU compute hotspots"):report.index("GPU API")]
            self.assertLess(hotspots_section.index("compute_stencil"), hotspots_section.index("main"))

    def test_unfiltered_ranks_by_inclusive_time_main_beats_compute_stencil(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(os.path.join(FIXTURES, "mpi_2rank"), dest, show_all=True, unfiltered=True)
            self.assertIn("Ranked by inclusive (total) time", report)
            hotspots_section = report[report.index("CPU compute hotspots"):report.index("GPU API")]
            self.assertLess(hotspots_section.index("main"), hotspots_section.index("compute_stencil"))


class NestedDatedSubdirectoryTests(unittest.TestCase):
    """rocprof-sys's default ROCPROFSYS_TIME_OUTPUT behavior nests every per-process file one
    level deeper, inside an auto-generated timestamped subdirectory (e.g. "2026-08-03_09.24/") --
    reported as a real bug against this fixture's real-world equivalent. Every scan in this
    module must find files there, not just directly under output_dir."""

    DIR = os.path.join(FIXTURES, "mpi_2rank_dated_subdir")

    def test_aggregate_finds_files_one_level_down(self):
        cpu, gpu, scanned, total_runtime = hotspots.aggregate(self.DIR)
        self.assertEqual(len(scanned), 2)
        by_label = {e["label"]: e for e in cpu}
        self.assertAlmostEqual(by_label["compute_stencil"]["sum"], 18.9)
        self.assertAlmostEqual(total_runtime, 21.824161)

    def test_aggregate_per_rank_finds_files_one_level_down(self):
        per_file, scanned = hotspots.aggregate_per_rank(self.DIR)
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(per_file), 2)

    def test_load_metadata_finds_nested_metadata_json(self):
        metadata = hotspots.load_metadata(self.DIR)
        self.assertEqual(hotspots.guess_executable(metadata), "jacobi_mpi")
        self.assertEqual(hotspots.guess_num_ranks(metadata, []), 2)

    def test_guess_run_datetime_falls_back_to_nested_scanned_file_dirname(self):
        _cpu, _gpu, scanned, _total = hotspots.aggregate(self.DIR)
        metadata = hotspots.load_metadata(self.DIR)  # no start_time field in this fixture
        self.assertEqual(hotspots.guess_run_datetime(metadata, self.DIR, scanned), "2026-08-03_09.24")

    def test_write_report_succeeds_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "hotspots.txt")
            report = hotspots.write_report(self.DIR, dest)
            self.assertIn("compute_stencil", report)
            self.assertIn("run date/time: 2026-08-03_09.24", report)
            self.assertIn("MPI ranks: 2", report)


if __name__ == "__main__":
    unittest.main()
