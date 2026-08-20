import contextlib
import importlib.util
import io
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

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
SINGLE_RANK_CSV = os.path.join(FIXTURES, "trace_single_rank", "rank0.csv")
MULTI_INSTANCE_CSV = os.path.join(FIXTURES, "trace_multi_instance", "rank0.csv")
NO_MATCH_CSV = os.path.join(FIXTURES, "trace_corr_id_no_match", "rank0.csv")
AMBIGUOUS_CSV = os.path.join(FIXTURES, "trace_corr_id_ambiguous", "rank0.csv")
NOISE_CSV = os.path.join(FIXTURES, "trace_calltree_noise", "rank0.csv")

SINGLE_RANK_LABELS = {"main", "jacobi_sweep", "hipLaunchKernel", "MPI_Barrier", "jacobi_kernel.kd"}


def by_label(rows, label):
    return next(r for r in rows if r["label"] == label)


class BuildRankAggregateTests(unittest.TestCase):
    def test_self_sum_subtracts_only_structural_children(self):
        rows = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        main = by_label(rows, "main")
        sweep = by_label(rows, "jacobi_sweep")
        launch = by_label(rows, "hipLaunchKernel")
        self.assertAlmostEqual(main["self_sum"], 0.4)
        self.assertAlmostEqual(sweep["self_sum"], 0.45)
        # not reduced by the kernel dispatch it gets reparented onto -- corr_id joining
        # happens after self_sum is computed, precisely to avoid this.
        self.assertAlmostEqual(launch["self_sum"], 0.05)

    def test_corr_id_exact_match_reparents_kernel_onto_launch_row(self):
        rows = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        launch = by_label(rows, "hipLaunchKernel")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIs(kernel["parent"], launch)
        self.assertIn("gpu_kernel", kernel["tags"])

    def test_corr_id_no_match_leaves_kernel_as_its_own_root(self):
        rows = agg.build_rank_aggregate(NO_MATCH_CSV, "r0")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(kernel["parent"])

    def test_corr_id_ambiguous_warns_once_and_leaves_unattached(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = agg.build_rank_aggregate(AMBIGUOUS_CSV, "r0")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(kernel["parent"])
        warnings = [line for line in buf.getvalue().splitlines() if line.startswith("warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 kernel-dispatch row(s)", warnings[0])

    def test_repeated_same_position_calls_collapse_into_one_merged_row(self):
        rows = agg.build_rank_aggregate(MULTI_INSTANCE_CSV, "r0")
        launch = by_label(rows, "hipLaunchKernel")
        self.assertEqual(launch["count"], 3)
        self.assertAlmostEqual(launch["self_sum"], 0.06)
        self.assertAlmostEqual(launch["sum"], 0.06)

    def test_no_row_dropped_and_every_tag_kept(self):
        # tool-independent: nothing filtered even though a tool's own flags might later hide
        # some of these -- that's a stage5/6 decision, never baked in here.
        rows = agg.build_rank_aggregate(SINGLE_RANK_CSV, "r0")
        self.assertEqual({r["label"] for r in rows}, SINGLE_RANK_LABELS)

    def test_untethered_kernel_dispatch_row_does_not_contaminate_the_real_root_via_tag_rows(self):
        # Regression: tag_rows() must run AFTER the corr_id join, not before. Before the join, a
        # kernel-dispatch row is still an untethered "root" -- if tag_rows() ran at that point, its
        # own sibling-group derivation would compare it against the real CPU thread root as if they
        # were siblings sharing one parent, and (since the kernel-dispatch row's own tiny subtree
        # never matches wrapper_noise) wrongly flag the CPU root's entire subtree as
        # wrapper_branch_noise-contaminated, even though nothing about that CPU subtree is
        # genuinely contaminated relative to a real sibling.
        rows = agg.build_rank_aggregate(NOISE_CSV, "r0")
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
        fresh = agg.build_rank_aggregate(self.csv_path, "r0")
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


if __name__ == "__main__":
    unittest.main()
