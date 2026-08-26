import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_trace_aggregate.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_trace_aggregate", MODULE_PATH)
agg = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_trace_aggregate"] = agg
spec.loader.exec_module(agg)

import stage6_time_range_config as trc  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
SINGLE_RANK_CSV = os.path.join(FIXTURES, "trace_single_rank", "rank0.csv")
MULTI_INSTANCE_CSV = os.path.join(FIXTURES, "trace_multi_instance", "rank0.csv")
NO_MATCH_CSV = os.path.join(FIXTURES, "trace_corr_id_no_match", "rank0.csv")
AMBIGUOUS_CSV = os.path.join(FIXTURES, "trace_corr_id_ambiguous", "rank0.csv")
NOISE_CSV = os.path.join(FIXTURES, "trace_calltree_noise", "rank0.csv")
OWNER_REANCHOR_CSV = os.path.join(FIXTURES, "trace_kernel_owner_reanchor", "rank0.csv")
TIME_RANGE_CSV = os.path.join(FIXTURES, "trace_time_range", "rank0.csv")

SINGLE_RANK_LABELS = {"main", "jacobi_sweep", "hipLaunchKernel", "MPI_Barrier", "jacobi_kernel.kd"}


def by_label(rows, label):
    return next(r for r in rows if r["label"] == label)


class BuildRankAggregateTests(unittest.TestCase):
    def test_self_sum_subtracts_only_structural_children(self):
        rows, _extent = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        main = by_label(rows, "main")
        sweep = by_label(rows, "jacobi_sweep")
        launch = by_label(rows, "hipLaunchKernel")
        self.assertAlmostEqual(main["self_sum"], 0.4)
        self.assertAlmostEqual(sweep["self_sum"], 0.45)
        # not reduced by the kernel dispatch it gets reparented onto -- corr_id joining
        # happens after self_sum is computed, precisely to avoid this.
        self.assertAlmostEqual(launch["self_sum"], 0.05)

    def test_corr_id_exact_match_reparents_kernel_onto_launch_row(self):
        rows, _extent = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        launch = by_label(rows, "hipLaunchKernel")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIs(kernel["parent"], launch)
        self.assertIn("gpu_kernel", kernel["tags"])

    def test_corr_id_no_match_leaves_kernel_as_its_own_root(self):
        rows, _extent = agg.build_rank_aggregate(NO_MATCH_CSV, "r0")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(kernel["parent"])

    def test_corr_id_ambiguous_warns_once_and_leaves_unattached(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows, _extent = agg.build_rank_aggregate(AMBIGUOUS_CSV, "r0")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(kernel["parent"])
        warnings = [line for line in buf.getvalue().splitlines() if line.startswith("warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 kernel-dispatch row(s)", warnings[0])

    def test_repeated_same_position_calls_collapse_into_one_merged_row(self):
        rows, _extent = agg.build_rank_aggregate(MULTI_INSTANCE_CSV, "r0")
        launch = by_label(rows, "hipLaunchKernel")
        self.assertEqual(launch["count"], 3)
        self.assertAlmostEqual(launch["self_sum"], 0.06)
        self.assertAlmostEqual(launch["sum"], 0.06)

    def test_no_row_dropped_and_every_tag_kept(self):
        # tool-independent: nothing filtered even though a tool's own flags might later hide
        # some of these -- that's a stage5/6 decision, never baked in here.
        rows, _extent = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        self.assertEqual({r["label"] for r in rows}, SINGLE_RANK_LABELS)

    def test_untethered_kernel_dispatch_row_does_not_contaminate_the_real_root_via_tag_rows(self):
        # Regression: tag_rows() must run AFTER the corr_id join, not before. Before the join, a
        # kernel-dispatch row is still an untethered "root" -- if tag_rows() ran at that point, its
        # own sibling-group derivation would compare it against the real CPU thread root as if they
        # were siblings sharing one parent, and (since the kernel-dispatch row's own tiny subtree
        # never matches wrapper_noise) wrongly flag the CPU root's entire subtree as
        # wrapper_branch_noise-contaminated, even though nothing about that CPU subtree is
        # genuinely contaminated relative to a real sibling.
        rows, _extent = agg.build_rank_aggregate(NOISE_CSV, "r0")
        main = by_label(rows, "main")
        self.assertEqual(main["structural_drop_tags"], set())
        # And the kernel is exactly where the corr_id join should have put it.
        launch = by_label(rows, "hipLaunchKernel")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIs(kernel["parent"], launch)


class GetRankAggregateCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "rank0.csv")
        shutil.copy(SINGLE_RANK_CSV, self.csv_path)
        self.cache_path = os.path.join(self.tmp, "r0.agg.json")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cache_miss_builds_and_writes_cache_file(self):
        rows = agg.get_rank_aggregate(self.csv_path, "r0")
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertEqual({r["label"] for r in rows}, SINGLE_RANK_LABELS)

    def test_cache_hit_does_not_rebuild(self):
        agg.get_rank_aggregate(self.csv_path, "r0")  # populate the cache
        with mock.patch.object(agg, "build_rank_aggregate") as mocked:
            agg.get_rank_aggregate(self.csv_path, "r0")
            mocked.assert_not_called()

    def test_stale_cache_triggers_rebuild(self):
        agg.get_rank_aggregate(self.csv_path, "r0")
        future = time.time() + 10
        os.utime(self.csv_path, (future, future))  # source now newer than the cache
        with mock.patch.object(agg, "build_rank_aggregate", wraps=agg.build_rank_aggregate) as mocked:
            agg.get_rank_aggregate(self.csv_path, "r0")
            mocked.assert_called_once()

    def test_corrupt_cache_falls_back_to_a_fresh_build(self):
        agg.get_rank_aggregate(self.csv_path, "r0")
        with open(self.cache_path, "w") as f:
            f.write("not valid json{{{")
        rows = agg.get_rank_aggregate(self.csv_path, "r0")
        self.assertEqual({r["label"] for r in rows}, SINGLE_RANK_LABELS)

    def test_decoded_cache_matches_a_fresh_build_field_by_field(self):
        fresh, _extent = agg.build_rank_aggregate(self.csv_path, "r0")
        agg.get_rank_aggregate(self.csv_path, "r0")  # cache miss -- writes the cache
        cached = agg.get_rank_aggregate(self.csv_path, "r0")  # cache hit -- decodes it back

        fresh_by_label = {r["label"]: r for r in fresh}
        cached_by_label = {r["label"]: r for r in cached}
        self.assertEqual(set(fresh_by_label), set(cached_by_label))
        for label, fresh_row in fresh_by_label.items():
            cached_row = cached_by_label[label]
            self.assertEqual(fresh_row["tags"], cached_row["tags"])
            self.assertEqual(fresh_row["structural_drop_tags"], cached_row["structural_drop_tags"])
            self.assertEqual(fresh_row["count"], cached_row["count"])
            self.assertAlmostEqual(fresh_row["self_sum"], cached_row["self_sum"])
            self.assertAlmostEqual(fresh_row["sum"], cached_row["sum"])
            fresh_parent = fresh_row["parent"]["label"] if fresh_row["parent"] else None
            cached_parent = cached_row["parent"]["label"] if cached_row["parent"] else None
            self.assertEqual(fresh_parent, cached_parent)


def make_raw_row(name, tid=None, ts=0.0, category="host", parent=None, corr_id=None):
    """A minimal raw (pre-merge) row -- the shape _reanchor_kernels_by_owner_and_time() operates
    on: parse_trace_csv()/attach_ancestry()'s own per-instance rows, keyed by "name" (not
    "label" -- see stage1_rocprofsys_trace.LABEL_KEY), each carrying its own "ts"/"tid"/"category"/
    "parent"/"corr_id"."""
    return {"name": name, "tid": tid, "ts": ts, "category": category, "parent": parent, "corr_id": corr_id}


class ReanchorKernelsByOwnerAndTimeTests(unittest.TestCase):
    def test_exact_owner_match_reanchors_onto_the_instance_preceding_it_in_time(self):
        owner = make_raw_row("foo$mod_", tid=1000, ts=10.0)
        launch = make_raw_row("hipModuleLaunchKernel", tid=1000, ts=15.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "foo$mod_$ck_L1_1", tid=None, ts=16.0, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([owner, launch, kernel])
        self.assertIs(kernel["parent"], owner)

    def test_no_ck_or_omp_offloading_marker_leaves_in_place(self):
        launch = make_raw_row("hipLaunchKernel", tid=1000, ts=15.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "SomeKernel", tid=None, ts=16.0, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([launch, kernel])
        self.assertIs(kernel["parent"], launch)

    def test_no_owner_match_leaves_in_place(self):
        launch = make_raw_row("hipModuleLaunchKernel", tid=1000, ts=15.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "foo$mod_$ck_L1_1", tid=None, ts=16.0, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([launch, kernel])  # no "foo$mod_" row anywhere
        self.assertIs(kernel["parent"], launch)

    def test_kernel_dispatched_before_any_owner_instance_leaves_in_place(self):
        owner = make_raw_row("foo$mod_", tid=1000, ts=20.0)  # starts AFTER the kernel dispatches
        launch = make_raw_row("hipModuleLaunchKernel", tid=1000, ts=5.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "foo$mod_$ck_L1_1", tid=None, ts=6.0, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([owner, launch, kernel])
        self.assertIs(kernel["parent"], launch)

    def test_no_corr_id_resolved_parent_leaves_in_place(self):
        # A kernel _attach_kernels_by_corr_id() never resolved (parent is still None) has no known
        # launching thread to scope the owner search to.
        owner = make_raw_row("foo$mod_", tid=1000, ts=10.0)
        kernel = make_raw_row(
            "foo$mod_$ck_L1_1", tid=None, ts=16.0, category="rocm_kernel_dispatch", parent=None,
        )
        agg._reanchor_kernels_by_owner_and_time([owner, kernel])
        self.assertIsNone(kernel["parent"])

    def test_two_same_thread_candidates_resolve_to_the_chronologically_correct_one(self):
        # Ambiguous by label alone (two real "bar$mod_" instances on the same thread, at two
        # different tree positions) -- resolved exactly by time, not split or guessed.
        owner_early = make_raw_row("bar$mod_", tid=2000, ts=10.0)
        owner_late = make_raw_row("bar$mod_", tid=2000, ts=30.0)
        launch = make_raw_row("hipModuleLaunchKernel", tid=2000, ts=31.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "bar$mod_$ck_L2_2", tid=None, ts=31.5, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([owner_early, owner_late, launch, kernel])
        self.assertIs(kernel["parent"], owner_late)

    def test_same_label_on_a_different_thread_is_not_picked_over_the_launching_threads_own(self):
        # A same-owner-labeled row on a DIFFERENT thread, with a start time that would win a
        # naive rank-wide "nearest preceding" search, must lose to the correct (older, but
        # same-thread) candidate -- proving tid-scoping, not just time, is load-bearing.
        correct_owner = make_raw_row("baz$mod_", tid=3000, ts=10.0)
        wrong_thread_owner = make_raw_row("baz$mod_", tid=4000, ts=20.0)
        launch = make_raw_row("hipModuleLaunchKernel", tid=3000, ts=21.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "baz$mod_$ck_L3_3", tid=None, ts=21.5, category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time(
            [correct_owner, wrong_thread_owner, launch, kernel],
        )
        self.assertIs(kernel["parent"], correct_owner)

    def test_generic_omp_offloading_name_also_reanchors(self):
        # The generalized (non-Cray-Fortran) owner-name convention -- see
        # stage4_rocprofsys_common.kernel_owner_label() -- resolves the same way.
        owner = make_raw_row("launch_omp_kernel", tid=1000, ts=10.0)
        launch = make_raw_row("hipModuleLaunchKernel", tid=1000, ts=15.0, category="rocm_hip_api")
        kernel = make_raw_row(
            "__omp_offloading_4f_8fb8827_launch_omp_kernel_l6", tid=None, ts=16.0,
            category="rocm_kernel_dispatch", parent=launch,
        )
        agg._reanchor_kernels_by_owner_and_time([owner, launch, kernel])
        self.assertIs(kernel["parent"], owner)

    def test_end_to_end_against_a_real_shaped_fixture(self):
        # Mirrors the real bug: the kernel's corr_id-exact position (nested under a generic
        # ompt_implicit_task -> ompt_target -> hipModuleLaunchKernel chain) is structurally
        # uninformative, but its own name still identifies the real owning subroutine, which
        # exists as its own CPU row elsewhere in the tree, and started before the kernel dispatched.
        rows, _extent = agg.build_rank_aggregate(OWNER_REANCHOR_CSV, "r0")
        kernel = by_label(rows, "convection_stable_dt$convection_time_integrator_mod_$ck_L558_99.kd")
        owner = by_label(rows, "convection_stable_dt$convection_time_integrator_mod_")
        self.assertIs(kernel["parent"], owner)


class OverlapWithRangesTests(unittest.TestCase):
    def test_fully_inside_a_range_keeps_the_whole_width(self):
        width, touches = agg._overlap_with_ranges(5.0, 2.0, [(0.0, 10.0)])
        self.assertAlmostEqual(width, 2.0)
        self.assertTrue(touches)

    def test_fully_outside_every_range_is_zero_and_does_not_touch(self):
        width, touches = agg._overlap_with_ranges(20.0, 2.0, [(0.0, 10.0)])
        self.assertEqual(width, 0.0)
        self.assertFalse(touches)

    def test_straddling_a_boundary_clips_to_the_overlapping_portion(self):
        width, touches = agg._overlap_with_ranges(8.0, 5.0, [(0.0, 10.0)])  # [8,13) vs [0,10)
        self.assertAlmostEqual(width, 2.0)
        self.assertTrue(touches)

    def test_zero_duration_event_exactly_on_a_boundary_touches_but_has_no_width(self):
        width, touches = agg._overlap_with_ranges(10.0, 0.0, [(0.0, 10.0)])
        self.assertEqual(width, 0.0)
        self.assertTrue(touches)

    def test_open_start_and_open_end_bounds_are_honored(self):
        width, touches = agg._overlap_with_ranges(-100.0, 5.0, [(None, 0.0)])
        self.assertAlmostEqual(width, 5.0)
        self.assertTrue(touches)
        width2, touches2 = agg._overlap_with_ranges(1000.0, 5.0, [(500.0, None)])
        self.assertAlmostEqual(width2, 5.0)
        self.assertTrue(touches2)

    def test_multiple_disjoint_ranges_sum_their_own_overlaps(self):
        width, touches = agg._overlap_with_ranges(0.0, 20.0, [(2.0, 5.0), (10.0, 12.0)])
        self.assertAlmostEqual(width, 5.0)
        self.assertTrue(touches)


class TimeRangeClippingAndExtentTests(unittest.TestCase):
    def tearDown(self):
        trc.configure(None)

    def test_extent_reflects_the_real_unfiltered_span_regardless_of_active_range(self):
        trc.configure("30:70")
        _rows, extent = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        self.assertEqual(extent, (0.0, 100.0))

    def test_extent_with_no_range_active_matches_ranged_extent(self):
        trc.configure(None)
        _rows, extent_unranged = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        trc.configure("30:70")
        _rows2, extent_ranged = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        self.assertEqual(extent_unranged, extent_ranged)

    def test_straddling_function_keeps_only_its_in_range_portion(self):
        # compute_phase spans [20,80]; range [30,70] clips it to width 40.
        trc.configure("30:70")
        rows, _extent = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        compute_phase = by_label(rows, "compute_phase")
        self.assertAlmostEqual(compute_phase["sum"], 40.0)
        # self_sum = clipped width (40) minus its clipped structural children (hipLaunchKernel: 0,
        # mpi_call: 5) = 35.
        self.assertAlmostEqual(compute_phase["self_sum"], 35.0)
        self.assertEqual(compute_phase["count"], 1)

    def test_call_entirely_outside_every_range_gets_zero_count_and_zero_time(self):
        trc.configure("30:70")
        rows, _extent = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        init_phase = by_label(rows, "init_phase")
        teardown_phase = by_label(rows, "teardown_phase")
        for row in (init_phase, teardown_phase):
            self.assertEqual(row["count"], 0)
            self.assertEqual(row["self_sum"], 0.0)
            self.assertEqual(row["sum"], 0.0)

    def test_concurrently_executing_reanchored_kernel_keeps_its_own_in_range_time(self):
        # jacobi_kernel.kd is reparented onto hipLaunchKernel via corr_id, but its raw span
        # [22,62] extends well beyond hipLaunchKernel's own [22,25] -- concurrent GPU execution.
        # hipLaunchKernel's own span never touches [30,70] at all; the kernel's does.
        trc.configure("30:70")
        rows, _extent = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        launch = by_label(rows, "hipLaunchKernel")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertEqual(launch["count"], 0)
        self.assertEqual(launch["self_sum"], 0.0)
        self.assertIs(kernel["parent"], launch)
        self.assertAlmostEqual(kernel["self_sum"], 32.0)  # overlap([22,62], [30,70]) = 32

    def test_ts_is_never_mutated_by_clipping(self):
        # _reanchor_kernels_by_owner_and_time() depends on real, unclamped ts values -- confirmed
        # indirectly here by checking the reanchored parent is still correct under an active range
        # covering only PART of the owner/kernel timeline (a clamped ts would corrupt the
        # bisect-based ordering this depends on).
        trc.configure("30:70")
        rows, _extent = agg.build_rank_aggregate(OWNER_REANCHOR_CSV, "r0")
        kernel = by_label(rows, "convection_stable_dt$convection_time_integrator_mod_$ck_L558_99.kd")
        owner = by_label(rows, "convection_stable_dt$convection_time_integrator_mod_")
        self.assertIs(kernel["parent"], owner)

    def test_no_active_range_leaves_count_self_sum_sum_unchanged(self):
        trc.configure(None)
        unranged, _extent = agg.build_rank_aggregate(TIME_RANGE_CSV, "r0")
        by_label_unranged = {r["label"]: r for r in unranged}
        main = by_label_unranged["main"]
        self.assertEqual(main["count"], 1)
        self.assertAlmostEqual(main["sum"], 100.0)


class RangeAwareCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "rank0.csv")
        shutil.copy(TIME_RANGE_CSV, self.csv_path)
        self.cache_path = os.path.join(self.tmp, "r0.agg.json")

    def tearDown(self):
        shutil.rmtree(self.tmp)
        trc.configure(None)

    def test_switching_ranges_rebuilds_and_overwrites_the_same_file_not_a_new_one(self):
        trc.configure(None)
        agg.get_rank_aggregate(self.csv_path, "r0")
        self.assertEqual(sorted(os.listdir(self.tmp)), ["r0.agg.json", "rank0.csv"])

        trc.configure("30:70")
        rows = agg.get_rank_aggregate(self.csv_path, "r0")
        self.assertEqual(sorted(os.listdir(self.tmp)), ["r0.agg.json", "rank0.csv"])  # still just one
        by_label_ranged = {r["label"]: r for r in rows}
        self.assertAlmostEqual(by_label_ranged["compute_phase"]["sum"], 40.0)

    def test_same_range_requested_twice_is_a_cache_hit(self):
        trc.configure("30:70")
        agg.get_rank_aggregate(self.csv_path, "r0")
        with mock.patch.object(agg, "build_rank_aggregate") as mocked:
            agg.get_rank_aggregate(self.csv_path, "r0")
            mocked.assert_not_called()

    def test_different_range_requested_is_a_cache_miss(self):
        trc.configure("30:70")
        agg.get_rank_aggregate(self.csv_path, "r0")
        trc.configure("0:10")
        with mock.patch.object(agg, "build_rank_aggregate", wraps=agg.build_rank_aggregate) as mocked:
            agg.get_rank_aggregate(self.csv_path, "r0")
            mocked.assert_called_once()

    def test_ranged_cache_then_no_range_request_is_also_a_cache_miss(self):
        trc.configure("30:70")
        agg.get_rank_aggregate(self.csv_path, "r0")
        trc.configure(None)
        with mock.patch.object(agg, "build_rank_aggregate", wraps=agg.build_rank_aggregate) as mocked:
            agg.get_rank_aggregate(self.csv_path, "r0")
            mocked.assert_called_once()

    def test_old_bare_array_cache_format_degrades_gracefully_to_a_rebuild(self):
        trc.configure(None)
        with open(self.cache_path, "w") as f:
            json.dump([{
                "label": "stale", "tags": [], "structural_drop_tags": [],
                "count": 1, "self_sum": 1.0, "sum": 1.0, "parent_index": None,
            }], f)
        rows = agg.get_rank_aggregate(self.csv_path, "r0")
        labels = {r["label"] for r in rows}
        self.assertNotEqual(labels, {"stale"})
        self.assertIn("main", labels)


class GetRankTimeExtentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "rank0.csv")
        shutil.copy(TIME_RANGE_CSV, self.csv_path)

    def tearDown(self):
        shutil.rmtree(self.tmp)
        trc.configure(None)

    def test_returns_the_real_extent_with_no_range_active(self):
        trc.configure(None)
        extent = agg.get_rank_time_extent(self.csv_path, "r0")
        self.assertEqual(extent, (0.0, 100.0))

    def test_returns_the_real_extent_even_with_a_range_active(self):
        trc.configure("30:70")
        extent = agg.get_rank_time_extent(self.csv_path, "r0")
        self.assertEqual(extent, (0.0, 100.0))

    def test_falls_back_to_a_fresh_build_if_the_cache_file_is_missing(self):
        trc.configure(None)
        agg.get_rank_aggregate(self.csv_path, "r0")
        os.remove(os.path.join(self.tmp, "r0.agg.json"))
        extent = agg.get_rank_time_extent(self.csv_path, "r0")
        self.assertEqual(extent, (0.0, 100.0))


if __name__ == "__main__":
    unittest.main()
