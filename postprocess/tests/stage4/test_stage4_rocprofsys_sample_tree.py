import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_sample_tree.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_sample_tree", MODULE_PATH)
s4t = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_sample_tree"] = s4t
spec.loader.exec_module(s4t)

from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs sys.path insert above first)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
SAMPLING_FALLBACK_DIR = os.path.join(FIXTURES, "calltree_sampling_fallback")

RANK = "r0"  # every hand-built test tree in this file simulates one rank


def make_row(label, parent=None, count=1, self_sum=0.0, total_sum=None, gpu=False):
    """A minimal hand-built merged-tree node -- same shape merge_rank_trees()
    produces (a "per_rank" dict, not flat count/self_sum/sum fields), without
    needing a fixture file or a real multi-rank merge. Every test in this
    file builds its own tiny tree directly, so the shared module's math
    (kernel attribution, rendering, aggregation) can be exercised in
    isolation from any one tool's own filtering rules or from
    merge_rank_trees() itself (covered separately in MergeRankTreesTests)."""
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
    def test_matches_known_prefixes(self):
        self.assertTrue(s4t.is_kernel_launch("hipLaunchKernel"))
        self.assertTrue(s4t.is_kernel_launch("hipModuleLaunchKernel"))

    def test_matches_namespace_qualified_symbol(self):
        # substring match, not startswith -- catches a demangled C++ symbol
        # where the launch call isn't the first thing in the label.
        self.assertTrue(s4t.is_kernel_launch("hip::hipModuleLaunchKernel(ihipModuleSymbol_t*, ...)"))

    def test_matches_cray_acc_entry_point(self):
        self.assertTrue(s4t.is_kernel_launch("__cray_start_acc_kernel"))

    def test_matches_omp_target_offload_entry_point(self):
        # LLVM libomptarget's launch entry point -- confirmed in real
        # test_apps HPC data under both amdclang++ and Cray CCE.
        self.assertTrue(s4t.is_kernel_launch("__tgt_target_kernel"))

    def test_rejects_unrelated_label(self):
        self.assertFalse(s4t.is_kernel_launch("compute_stencil"))


class KernelAnchorAttributionTests(unittest.TestCase):
    def test_single_anchor_gets_full_attribution(self):
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=100)
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
        main = make_row("main")
        compute_a = make_row("compute_a", parent=main)
        compute_b = make_row("compute_b", parent=main)
        launch_a = make_row("hipLaunchKernel", parent=compute_a, count=300)
        launch_b = make_row("hipLaunchKernel", parent=compute_b, count=200)
        rows = [main, compute_a, compute_b, launch_a, launch_b]

        s4t.attach_kernel_summaries(rows, {RANK: {"K": (500, 10.0)}}, NEVER_PRUNED)
        node_a = compute_a["static_children"][0]
        node_b = compute_b["static_children"][0]
        self.assertAlmostEqual(rank_values(node_a)[2], 6.0)   # 300/500 of 10.0s
        self.assertAlmostEqual(rank_values(node_b)[2], 4.0)   # 200/500 of 10.0s
        self.assertIn("~60% estimate: this site issued 300/500", node_a["label"])
        self.assertIn("~40% estimate: this site issued 200/500", node_b["label"])

    def test_no_launch_call_anywhere_returns_remainder_unattached(self):
        main = make_row("main")
        rows = [main]
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, {"K"})
        self.assertEqual(main["static_children"], [])

    def test_kernel_owner_label_strips_ck_suffix(self):
        self.assertEqual(
            s4t.kernel_owner_label("jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form"),
            "jacobi_sweep$pressure_solver_mod_",
        )

    def test_kernel_owner_label_unchanged_without_ck_marker(self):
        self.assertEqual(s4t.kernel_owner_label("JacobiIterationKernel"), "JacobiIterationKernel")

    def test_kernel_attaches_to_exact_owner_subroutine_not_launch_anchor(self):
        # A CPU tree node named exactly after the kernel's compiler-embedded
        # owner subroutine exists, but it sits far from the nearest
        # hipLaunchKernel call (a different, unrelated subroutine) -- the
        # kernel must attach to its real owner, not the launch-call anchor.
        main = make_row("main")
        unrelated_caller = make_row("unrelated_caller", parent=main)
        launch = make_row("hipLaunchKernel", parent=unrelated_caller, count=1)
        jacobi_sweep = make_row("jacobi_sweep$pressure_solver_mod_", parent=main, count=716)
        rows = [main, unrelated_caller, launch, jacobi_sweep]

        kernels = {RANK: {"jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form": (716, 7.25)}}
        unattached = s4t.attach_kernel_summaries(rows, kernels, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(jacobi_sweep["static_children"], [])
        self.assertEqual(unrelated_caller["static_children"], [])

    def test_unmatched_kernel_falls_back_to_launch_anchor(self):
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=5)
        rows = [main, compute, launch]

        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"UnnamedKernel": (5, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(compute["static_children"], [])

    def test_mixed_named_and_unmatched_kernels_split_correctly(self):
        main = make_row("main")
        unrelated_caller = make_row("unrelated_caller", parent=main)
        launch = make_row("hipLaunchKernel", parent=unrelated_caller, count=1)
        jacobi_sweep = make_row("jacobi_sweep$pressure_solver_mod_", parent=main, count=1)
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
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]

        gpu_kernel_by_rank = {"r0": {"K": (10, 1.0)}, "r1": {"K": (30, 3.0)}}
        s4t.attach_kernel_summaries(rows, gpu_kernel_by_rank, NEVER_PRUNED)
        kernel_node = compute["static_children"][0]["static_children"][0]
        self.assertAlmostEqual(kernel_node["per_rank"]["r0"]["sum"], 1.0)
        self.assertAlmostEqual(kernel_node["per_rank"]["r1"]["sum"], 3.0)

    def test_anchor_resolution_skips_pruned_ancestors(self):
        # hipLaunchKernel's immediate parent is itself pruned (e.g. GPU-API
        # noise) -- the anchor must be the nearest VISIBLE ancestor instead.
        main = make_row("main")
        noisy_wrapper = make_row("hipStreamCreate", parent=main, gpu=True)
        launch = make_row("hipLaunchKernel", parent=noisy_wrapper, count=10)
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
        anchor = make_row("compute")
        node = s4t.make_kernel_node("K", {}, parent=anchor)
        self.assertIs(node["parent"], anchor)


class AttachKernelSummariesCollectIntoTests(unittest.TestCase):
    def test_omitting_collect_into_changes_nothing(self):
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED)
        self.assertEqual(unattached, set())
        self.assertNotEqual(compute["static_children"], [])

    def test_collect_into_gathers_group_and_kernel_leaf_single_anchor(self):
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=1)
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
        main = make_row("main")
        compute_a = make_row("compute_a", parent=main)
        compute_b = make_row("compute_b", parent=main)
        launch_a = make_row("hipLaunchKernel", parent=compute_a, count=1)
        launch_b = make_row("hipLaunchKernel", parent=compute_b, count=1)
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
        main = make_row("main")
        compute = make_row("compute", parent=main)
        launch = make_row("hipLaunchKernel", parent=compute, count=1)
        rows = [main, compute, launch]
        collected = []
        s4t.attach_kernel_summaries(rows, {RANK: {"MyKernel": (1, 5.0)}}, NEVER_PRUNED, collect_into=collected)
        rows.extend(collected)

        chains = s4t.caller_chains_for_label(rows, "MyKernel")
        self.assertEqual(len(chains), 1)
        labels = [n["label"] for n in chains[0]]
        self.assertEqual(labels, ["main", "compute", "[GPU kernels -- rocprofv3]", "MyKernel"])

    def test_unattached_kernel_never_collected(self):
        main = make_row("main")
        rows = [main]
        collected = []
        unattached = s4t.attach_kernel_summaries(rows, {RANK: {"K": (1, 1.0)}}, NEVER_PRUNED, collect_into=collected)
        self.assertEqual(unattached, {"K"})
        self.assertEqual(collected, [])


class MergeRankTreesTests(unittest.TestCase):
    def test_merges_same_label_across_ranks_by_structural_position(self):
        # Two independent single-rank trees (distinct row objects, as if
        # parsed from two separate files) sharing the same "main" ->
        # "compute" structure must merge into ONE node per position, with
        # both ranks' contributions kept separately in per_rank.
        main_a = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 10.0, "gpu": False}
        compute_a = {"label": "compute", "parent": main_a, "count": 5, "self_sum": 1.0, "sum": 1.0, "gpu": False}
        main_b = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 20.0, "gpu": False}
        compute_b = {"label": "compute", "parent": main_b, "count": 7, "self_sum": 3.0, "sum": 3.0, "gpu": False}

        ranks = [
            ("rankA", [main_a, compute_a], [main_a]),
            ("rankB", [main_b, compute_b], [main_b]),
        ]
        merged_roots = s4t.merge_rank_trees(ranks)
        self.assertEqual(len(merged_roots), 1)
        merged_main = merged_roots[0]
        self.assertEqual(merged_main["label"], "main")
        self.assertEqual(set(merged_main["per_rank"].keys()), {"rankA", "rankB"})

        merged_compute = merged_main["children"]["compute"]
        self.assertEqual(merged_compute["per_rank"]["rankA"]["self_sum"], 1.0)
        self.assertEqual(merged_compute["per_rank"]["rankB"]["self_sum"], 3.0)
        self.assertIs(merged_compute["parent"], merged_main)

    def test_tags_unioned_across_ranks(self):
        row_a = {"label": "start_thread", "parent": None, "count": 1, "self_sum": 1.0, "sum": 1.0, "tags": set()}
        row_b = {"label": "start_thread", "parent": None, "count": 1, "self_sum": 1.0, "sum": 1.0, "tags": {"gpu_api"}}
        ranks = [("rankA", [row_a], [row_a]), ("rankB", [row_b], [row_b])]
        merged_roots = s4t.merge_rank_trees(ranks)
        self.assertEqual(merged_roots[0]["tags"], {"gpu_api"})


class AggregateNodeStatsTests(unittest.TestCase):
    def test_missing_rank_counts_as_zero_not_omitted(self):
        per_rank = {"r0": {"count": 10, "self_sum": 2.0, "sum": 2.0}}
        stats = s4t.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
        self.assertAlmostEqual(stats["self_avg"], 2.0 / 3)
        self.assertAlmostEqual(stats["self_min"], 0.0)
        self.assertAlmostEqual(stats["self_max"], 2.0)

    def test_std_dev_and_min_max(self):
        per_rank = {
            "r0": {"count": 1, "self_sum": 1.0, "sum": 1.0},
            "r1": {"count": 1, "self_sum": 3.0, "sum": 3.0},
            "r2": {"count": 1, "self_sum": 5.0, "sum": 5.0},
        }
        stats = s4t.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
        self.assertAlmostEqual(stats["self_avg"], 3.0)
        self.assertAlmostEqual(stats["self_std"], 1.632993161855452)
        self.assertAlmostEqual(stats["self_min"], 1.0)
        self.assertAlmostEqual(stats["self_max"], 5.0)


class CallerChainsForLabelTests(unittest.TestCase):
    def test_single_call_site_returns_one_root_to_target_chain(self):
        main = make_row("main")
        compute = make_row("compute", parent=main)
        target = make_row("hot_function", parent=compute)
        rows = [main, compute, target]

        chains = s4t.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main", "compute", "hot_function"])

    def test_two_distinct_call_sites_return_two_chains(self):
        main = make_row("main")
        compute_a = make_row("compute_a", parent=main)
        compute_b = make_row("compute_b", parent=main)
        target_a = make_row("hot_function", parent=compute_a)
        target_b = make_row("hot_function", parent=compute_b)
        rows = [main, compute_a, compute_b, target_a, target_b]

        chains = s4t.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 2)
        labels = {tuple(n["label"] for n in chain) for chain in chains}
        self.assertEqual(labels, {
            ("main", "compute_a", "hot_function"),
            ("main", "compute_b", "hot_function"),
        })

    def test_no_match_returns_empty_list(self):
        main = make_row("main")
        rows = [main]
        self.assertEqual(s4t.caller_chains_for_label(rows, "never_called"), [])

    def test_target_is_a_root_returns_single_node_chain(self):
        main = make_row("main")
        rows = [main]
        chains = s4t.caller_chains_for_label(rows, "main")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main"])
        self.assertIsNone(chains[0][0]["parent"])


class LoadRankTreesTests(unittest.TestCase):
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
        merged_roots = s4t.merge_rank_trees(ranks)
        flat = s4t.flatten_tree(merged_roots)
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
