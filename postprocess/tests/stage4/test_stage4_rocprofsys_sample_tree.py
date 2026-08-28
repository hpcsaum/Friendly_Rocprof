"""Tests for stage4_rocprofsys_sample_tree.py's rank loading and rocprofv3-kernel-onto-CPU-subroutine
attachment for the sample (timemory text-table) calltree pipeline.

IsKernelLaunchTests                   -- label patterns recognized as a GPU kernel-launch call site
KernelAnchorAttributionTests          -- real GPU kernel data split across one or more launch-call anchors, by owner-name match or launch-count proportion
MakeKernelNodeParentTests             -- make_kernel_node()'s parent defaulting/override
AttachKernelSummariesCollectIntoTests -- collect_into gathering the synthetic group/kernel nodes attach_kernel_summaries() creates
LoadRankTreesTests                    -- per-rank parse+ancestry loading, thread-root detection, primary/fallback file pattern selection, postprocess hook
KernelTotalsWithCountsTests           -- per-rank (count, seconds) totals from rocprofv3's kernel_stats.csv
PairGpuPerRankTests                   -- pairing CPU rank keys to GPU kernel data, only when rank counts match
AttachGpuKernelsTests                 -- end-to-end wiring of load_rank_trees()+merge+attach_kernel_summaries() into one tree
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

s4t = load_module_by_path("stage4_rocprofsys_sample_tree", "stage4", "stage4_rocprofsys_sample_tree.py")

from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs sys.path insert above first)
from stage4_rocprofsys_common import caller_chains_for_label, flatten_tree, merge_rank_trees  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
SAMPLING_FALLBACK_DIR = os.path.join(FIXTURES, "calltree_sampling_fallback")
GPU_SPAWNED_THREAD_DIR = os.path.join(FIXTURES, "gpu_spawned_thread")
MULTI_METRIC_RANK_DIR = os.path.join(FIXTURES, "multi_metric_rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")

RANK = "r0"  # every hand-built test tree in this file simulates one rank


def make_merged_tree_node(label, parent=None, count=1, self_sum=0.0, total_sum=None, gpu=False):
    """A minimal hand-built merged-tree node -- same shape merge_rank_trees()
    produces (a "per_rank" dict, not flat count/self_sum/sum fields), without
    needing a fixture file or a real multi-rank merge. Every test in this
    file builds its own tiny tree directly, so the shared module's math
    (kernel attribution, rendering, aggregation) can be exercised in
    isolation from any one tool's own filtering rules or from
    merge_rank_trees() itself (covered separately in
    test_stage4_rocprofsys_common.MergeRankTreesTests)."""
    total_sum = total_sum if total_sum is not None else self_sum
    return {
        "label": label, "parent": parent, "children": {}, "static_children": [],
        "gpu": gpu, "compiler_runtime": False, "mpi_territory": False,
        "per_rank": {RANK: {"count": count, "self_sum": self_sum, "sum": total_sum}},
    }


def rank_values(node, rank=RANK):
    """This node's own single-rank (count, self_sum, sum) -- a shortcut for
    tests that only ever simulate one rank, instead of going through
    aggregate_node_stats()'s full avg/std_dev/min/max (covered separately in
    AggregateNodeStatsTests)."""
    entry = node["per_rank"][rank]
    return entry["count"], entry["self_sum"], entry["sum"]


NEVER_PRUNED = lambda node: False  # noqa: E731


class IsKernelLaunchTests(unittest.TestCase):
    CASES = [
        ("hip_prefix", "hipLaunchKernel", True),
        ("hip_module_prefix", "hipModuleLaunchKernel", True),
        # substring match, not startswith -- catches a demangled C++ symbol where the launch
        # call isn't the first thing in the label.
        ("namespace_qualified_symbol", "hip::hipModuleLaunchKernel(ihipModuleSymbol_t*, ...)", True),
        ("cray_acc_entry_point", "__cray_start_acc_kernel", True),
        # LLVM libomptarget's launch entry point -- confirmed in real test_apps HPC data under
        # both amdclang++ and Cray CCE.
        ("omp_target_offload_entry_point", "__tgt_target_kernel", True),
        ("unrelated_label", "compute_stencil", False),
    ]

    def test_is_kernel_launch(self):
        for name, label, expected in self.CASES:
            with self.subTest(case=name):
                self.assertEqual(s4t.is_kernel_launch(label), expected)


class KernelAnchorAttributionTests(unittest.TestCase):
    def test_single_anchor_gets_full_attribution(self):
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=100)
        rows = [main, compute, launch]

        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"MyKernel": (100, 5.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertEqual(len(compute["static_children"]), 1)
        kernel_node = compute["static_children"][0]
        self.assertEqual(kernel_node["label"], "[GPU kernels -- rocprofv3]")
        count, _self_sum, total_sum = rank_values(kernel_node)
        self.assertAlmostEqual(count, 100)
        self.assertAlmostEqual(total_sum, 5.0)
        self.assertEqual(kernel_node["static_children"][0]["label"], "MyKernel")

    def test_multiple_anchors_split_proportionally_by_launch_count(self):
        main = make_merged_tree_node("main")
        compute_a = make_merged_tree_node("compute_a", parent=main)
        compute_b = make_merged_tree_node("compute_b", parent=main)
        launch_a = make_merged_tree_node("hipLaunchKernel", parent=compute_a, count=300)
        launch_b = make_merged_tree_node("hipLaunchKernel", parent=compute_b, count=200)
        rows = [main, compute_a, compute_b, launch_a, launch_b]

        s4t.attach_kernel_summaries(rows, {RANK: {"K": (500, 10.0)}}, NEVER_PRUNED)
        node_a = compute_a["static_children"][0]
        node_b = compute_b["static_children"][0]
        self.assertAlmostEqual(rank_values(node_a)[2], 6.0)   # 300/500 of 10.0s
        self.assertAlmostEqual(rank_values(node_b)[2], 4.0)   # 200/500 of 10.0s
        self.assertIn("~60% estimate: this site issued 300/500", node_a["label"])
        self.assertIn("~40% estimate: this site issued 200/500", node_b["label"])

    def test_multiple_anchors_all_zero_weight_split_evenly_not_divide_by_zero(self):
        # Both launch-call anchors have count=0 -- total_weight is 0, so the proportional-split
        # fraction (weight/total_weight) would divide by zero; must fall back to an even split
        # across the anchors instead.
        main = make_merged_tree_node("main")
        compute_a = make_merged_tree_node("compute_a", parent=main)
        compute_b = make_merged_tree_node("compute_b", parent=main)
        launch_a = make_merged_tree_node("hipLaunchKernel", parent=compute_a, count=0)
        launch_b = make_merged_tree_node("hipLaunchKernel", parent=compute_b, count=0)
        rows = [main, compute_a, compute_b, launch_a, launch_b]

        s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED)
        node_a = compute_a["static_children"][0]
        node_b = compute_b["static_children"][0]
        self.assertAlmostEqual(rank_values(node_a)[2], 0.5)  # even 50/50 split of 1.0s
        self.assertAlmostEqual(rank_values(node_b)[2], 0.5)

    def test_no_launch_call_anywhere_returns_remainder_unattached(self):
        main = make_merged_tree_node("main")
        rows = [main]
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, {"K"})
        self.assertEqual(main["static_children"], [])

    def test_kernel_attaches_to_exact_owner_subroutine_not_launch_anchor(self):
        # A CPU tree node named exactly after the kernel's compiler-embedded
        # owner subroutine exists, but it sits far from the nearest
        # hipLaunchKernel call (a different, unrelated subroutine) -- the
        # kernel must attach to its real owner, not the launch-call anchor.
        main = make_merged_tree_node("main")
        unrelated_caller = make_merged_tree_node("unrelated_caller", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=unrelated_caller, count=1)
        jacobi_sweep = make_merged_tree_node("jacobi_sweep$pressure_solver_mod_", parent=main, count=716)
        rows = [main, unrelated_caller, launch, jacobi_sweep]

        kernels = {RANK: {"jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form": (716, 7.25)}}
        unattached = s4t.attach_kernel_summaries(rows, kernels, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(jacobi_sweep["static_children"], [])
        self.assertEqual(unrelated_caller["static_children"], [])

    def test_unmatched_kernel_falls_back_to_launch_anchor(self):
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=5)
        rows = [main, compute, launch]

        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"UnnamedKernel": (5, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(compute["static_children"], [])

    def test_mixed_named_and_unmatched_kernels_split_correctly(self):
        main = make_merged_tree_node("main")
        unrelated_caller = make_merged_tree_node("unrelated_caller", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=unrelated_caller, count=1)
        jacobi_sweep = make_merged_tree_node("jacobi_sweep$pressure_solver_mod_", parent=main, count=1)
        rows = [main, unrelated_caller, launch, jacobi_sweep]

        kernels = {RANK: {
            "jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form": (1, 5.0),
            "SomeOtherKernel": (1, 2.0),
        }}
        unattached = s4t.attach_kernel_summaries(rows, kernels, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        jacobi_node = jacobi_sweep["static_children"][0]
        launch_node = unrelated_caller["static_children"][0]
        self.assertAlmostEqual(rank_values(jacobi_node)[2], 5.0)
        self.assertAlmostEqual(rank_values(launch_node)[2], 2.0)

    def test_per_rank_breakdown_preserved_not_pre_averaged(self):
        # Two ranks contribute very different amounts of the same kernel --
        # the synthetic node's own per_rank dict must keep them distinct
        # (for later load-balance columns), not collapse to one combined sum.
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]

        gpu_kernel_by_rank = {"r0": {"K": (10, 1.0)}, "r1": {"K": (30, 3.0)}}
        s4t.attach_kernel_summaries(rows, gpu_kernel_by_rank, NEVER_PRUNED)
        kernel_node = compute["static_children"][0]["static_children"][0]
        self.assertAlmostEqual(kernel_node["per_rank"]["r0"]["sum"], 1.0)
        self.assertAlmostEqual(kernel_node["per_rank"]["r1"]["sum"], 3.0)

    def test_anchor_resolution_skips_pruned_ancestors(self):
        # hipLaunchKernel's immediate parent is itself pruned (e.g. GPU-API
        # noise) -- the anchor must be the nearest VISIBLE ancestor instead.
        main = make_merged_tree_node("main")
        noisy_wrapper = make_merged_tree_node("hipStreamCreate", parent=main, gpu=True)
        launch = make_merged_tree_node("hipLaunchKernel", parent=noisy_wrapper, count=10)
        rows = [main, noisy_wrapper, launch]

        is_pruned = lambda node: node.get("gpu", False)  # noqa: E731
        s4t.attach_kernel_summaries(rows, {RANK: {"K": (10, 1.0)}}, is_pruned)
        self.assertNotEqual(main["static_children"], [])
        self.assertEqual(noisy_wrapper["static_children"], [])


class MakeKernelNodeParentTests(unittest.TestCase):
    def test_parent_defaults_to_none(self):
        node = s4t.make_kernel_node("K", {})
        self.assertIsNone(node["parent"])

    def test_parent_is_set_when_given(self):
        anchor = make_merged_tree_node("compute")
        node = s4t.make_kernel_node("K", {}, parent=anchor)
        self.assertIs(node["parent"], anchor)


class AttachKernelSummariesCollectIntoTests(unittest.TestCase):
    def test_omitting_collect_into_changes_nothing(self):
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(compute["static_children"], [])

    def test_collect_into_gathers_group_and_kernel_leaf_single_anchor(self):
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]
        collected = []
        s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED, collect_into=collected)
        self.assertEqual(len(collected), 2)
        group_node, kernel_node = collected
        self.assertEqual(group_node["label"], "[GPU kernels -- rocprofv3]")
        self.assertIs(group_node["parent"], compute)
        self.assertEqual(kernel_node["label"], "K")
        self.assertIs(kernel_node["parent"], group_node)

    def test_collect_into_gathers_every_group_across_multiple_anchors(self):
        main = make_merged_tree_node("main")
        compute_a = make_merged_tree_node("compute_a", parent=main)
        compute_b = make_merged_tree_node("compute_b", parent=main)
        launch_a = make_merged_tree_node("hipLaunchKernel", parent=compute_a, count=1)
        launch_b = make_merged_tree_node("hipLaunchKernel", parent=compute_b, count=1)
        rows = [main, compute_a, compute_b, launch_a, launch_b]
        collected = []
        s4t.attach_kernel_summaries(rows, {RANK: {"K": (2, 2.0)}}, NEVER_PRUNED, collect_into=collected)
        # two anchors -> two (group, kernel) pairs -- equal weight gives both groups identical
        # label text, so identity (not the label string) is what distinguishes them here.
        self.assertEqual(len(collected), 4)
        group_nodes = [n for n in collected if n["label"].startswith("[GPU kernels")]
        self.assertEqual(len(group_nodes), 2)
        self.assertIsNot(group_nodes[0], group_nodes[1])
        self.assertNotEqual(id(group_nodes[0]["parent"]), id(group_nodes[1]["parent"]))

    def test_caller_chains_for_label_walks_through_an_attached_kernel(self):
        # The actual point of Phase B: once a kernel is attached with collect_into, its own
        # synthetic leaf node is walkable by caller_chains_for_label() exactly like any real
        # CPU function -- no new tree-walking code needed for this to work.
        main = make_merged_tree_node("main")
        compute = make_merged_tree_node("compute", parent=main)
        launch = make_merged_tree_node("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]
        collected = []
        s4t.attach_kernel_summaries(rows, {RANK: {"MyKernel": (1, 5.0)}}, NEVER_PRUNED, collect_into=collected)
        rows.extend(collected)

        chains = caller_chains_for_label(rows, "MyKernel")
        self.assertEqual(len(chains), 1)
        labels = [n["label"] for n in chains[0]]
        self.assertEqual(labels, ["main", "compute", "[GPU kernels -- rocprofv3]", "MyKernel"])

    def test_unattached_kernel_never_collected(self):
        main = make_merged_tree_node("main")
        rows = [main]
        collected = []
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED, collect_into=collected)
        self.assertEqual(unattached, {"K"})
        self.assertEqual(collected, [])


class LoadRankTreesTests(unittest.TestCase):
    def test_basic_two_rank_tree(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)
        for rank_key, rows, roots in ranks:
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["label"], "main")
            # compute_stencil and hipMemcpy are main's only two children
            children = [r for r in rows if r["parent"] is roots[0]]
            self.assertEqual({c["label"] for c in children}, {"compute_stencil", "hipMemcpy"})

    def test_raises_on_empty_input_is_just_empty_list(self):
        # load_rank_trees() itself doesn't raise -- a report-building tool composed over it
        # does, once it sees an empty list. Confirmed here so that distinction stays intentional.
        cpu_dir, _gpu_dir = resolve_run_dirs(EMPTY_DIR)
        self.assertEqual(s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt"), [])

    def test_is_thread_root_flagged_case(self):
        # gpu_spawned_thread: start_thread's DEPTH nests one level under its
        # spawning pthread_create call -- is_thread_root correctly fires, but
        # root-enumeration here relies only on parent is None, not that flag.
        cpu_dir, _gpu_dir = resolve_run_dirs(GPU_SPAWNED_THREAD_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 1)
        _rank_key, _rows, roots = ranks[0]
        # main, hipRuntimeGetVersion, compute_stencil are three independent
        # DEPTH-0 roots in this fixture -- all three must be detected.
        self.assertEqual({r["label"] for r in roots}, {"main", "hipRuntimeGetVersion", "compute_stencil"})

    def test_depth_resets_to_zero_case(self):
        # multi_metric_rank: a second OS thread's own root row sits at DEPTH 0,
        # the same depth as thread 0's other roots -- parent is None still
        # correctly separates it without relying on is_thread_root.
        cpu_dir, _gpu_dir = resolve_run_dirs(MULTI_METRIC_RANK_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        _rank_key, _rows, roots = ranks[0]
        labels = [r["label"] for r in roots]
        self.assertEqual(labels.count("worker_loop"), 2)  # two distinct thread roots, not merged

    def test_primary_pattern_file_used_when_present(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)

    def test_fallback_pattern_used_only_when_primary_missing(self):
        # calltree_sampling_fallback: one rank has both files (primary wins), the other
        # rank has only the fallback pattern's file.
        ranks = s4t.load_rank_trees(SAMPLING_FALLBACK_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)
        labels_by_rank = {rk: {r["label"] for r in rows} for rk, rows, _roots in ranks}
        self.assertIn({"main_fallback", "instrumented_leaf"}, labels_by_rank.values())

    def test_swapping_which_pattern_is_primary_changes_the_winner(self):
        # Same fixture, patterns reversed -- the rank with both files now takes its data
        # from wall_clock instead of sampling_wall_clock.
        sampling_first = s4t.load_rank_trees(SAMPLING_FALLBACK_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt")
        wall_clock_first = s4t.load_rank_trees(SAMPLING_FALLBACK_DIR, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        sampling_labels = {rk: {r["label"] for r in rows} for rk, rows, _roots in sampling_first}
        wall_clock_labels = {rk: {r["label"] for r in rows} for rk, rows, _roots in wall_clock_first}
        self.assertNotEqual(sampling_labels, wall_clock_labels)

    def test_postprocess_hook_defaults_to_noop(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        _rank_key, rows, _roots = ranks[0]
        self.assertTrue(any(r["label"] == "hipMemcpy" for r in rows))  # nothing stripped by default

    def test_postprocess_hook_applied_when_given(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)

        def _drop_hipmemcpy(rows):
            return [r for r in rows if r["label"] != "hipMemcpy"]

        ranks = s4t.load_rank_trees(
            cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt", postprocess=_drop_hipmemcpy
        )
        _rank_key, rows, _roots = ranks[0]
        self.assertFalse(any(r["label"] == "hipMemcpy" for r in rows))


class KernelTotalsWithCountsTests(unittest.TestCase):
    def test_returns_count_and_seconds_per_kernel(self):
        _cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        totals = s4t.kernel_totals_with_counts(gpu_dir, 0)
        self.assertIn("JacobiIterationKernel", totals)
        count, seconds = totals["JacobiIterationKernel"]
        self.assertGreater(count, 0)
        self.assertGreater(seconds, 0)


class PairGpuPerRankTests(unittest.TestCase):
    def test_returns_none_when_gpu_dir_is_none(self):
        self.assertIsNone(s4t.pair_gpu_per_rank(None, "run", ["r0"]))

    def test_pairs_when_rank_counts_match(self):
        cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        rank_keys = [rk for rk, _rows, _roots in ranks]
        result = s4t.pair_gpu_per_rank(gpu_dir, KERNEL_ANCHOR_DIR, rank_keys)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(rank_keys))

    def test_returns_none_on_mismatched_rank_count(self):
        _cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        result = s4t.pair_gpu_per_rank(gpu_dir, KERNEL_ANCHOR_DIR, ["r0", "r1"])  # gpu side has only 1 file
        self.assertIsNone(result)


class AttachGpuKernelsTests(unittest.TestCase):
    """attach_gpu_kernels() is the stage4 half of what used to be one mixed attach-and-render
    function in stage5_tree_render.py -- rendering the fallback text for whatever comes back
    unattached is covered separately, in test_stage5_tree_render.RenderGpuKernelFallbackTests."""

    def _build_tree(self, run_dir):
        cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        rank_keys = [rk for rk, _rows, _roots in ranks]
        merged_roots = merge_rank_trees(ranks)
        flat = flatten_tree(merged_roots)
        gpu_per_rank = s4t.pair_gpu_per_rank(gpu_dir, run_dir, rank_keys)
        return flat, rank_keys, gpu_dir, gpu_per_rank

    def test_returns_empty_when_no_gpu_data(self):
        unattached, gpu_kernel_by_rank = s4t.attach_gpu_kernels([], None, None, [], NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertEqual(gpu_kernel_by_rank, {})

    def test_attaches_matched_kernel_and_returns_nothing_unattached(self):
        flat, rank_keys, gpu_dir, gpu_per_rank = self._build_tree(KERNEL_ANCHOR_DIR)
        unattached, gpu_kernel_by_rank = s4t.attach_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, NEVER_PRUNED)
        self.assertEqual(unattached, set())  # matched by name -- nothing left unattached
        self.assertIn("JacobiIterationKernel", gpu_kernel_by_rank[rank_keys[0]])
        has_kernel_group = any(
            any("GPU kernels" in c["label"] for c in node.get("static_children", []))
            for node in flat
        )
        self.assertTrue(has_kernel_group)

    def test_unmatched_kernel_comes_back_in_unattached(self):
        flat, rank_keys, gpu_dir, gpu_per_rank = self._build_tree(KERNEL_NO_ANCHOR_DIR)
        unattached, gpu_kernel_by_rank = s4t.attach_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, NEVER_PRUNED)
        self.assertIn("JacobiIterationKernel", unattached)
        self.assertIn("JacobiIterationKernel", gpu_kernel_by_rank[rank_keys[0]])


if __name__ == "__main__":
    unittest.main()
