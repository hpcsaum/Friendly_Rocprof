import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_common.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_common", MODULE_PATH)
s4c = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_common"] = s4c
spec.loader.exec_module(s4c)


def make_row(label, parent=None):
    """A minimal row for caller_chains_for_label() -- it only ever reads "label"/"parent"."""
    return {"label": label, "parent": parent}


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
        merged_roots = s4c.merge_rank_trees(ranks)
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
        merged_roots = s4c.merge_rank_trees(ranks)
        self.assertEqual(merged_roots[0]["tags"], {"gpu_api"})

    def test_single_rank_collapses_repeated_same_position_calls(self):
        # A single rank's own repeated calls at the same tree position (e.g. many identical
        # hipLaunchKernel siblings under one shared parent object) collapse into one merged
        # node -- the same by-id(parent)+label mechanism this function already uses across
        # ranks, exercised here with len(ranks) == 1.
        main = {"label": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 10.0}
        launch_1 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 1.0, "sum": 1.0}
        launch_2 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 2.0, "sum": 2.0}
        launch_3 = {"label": "hipLaunchKernel", "parent": main, "count": 1, "self_sum": 3.0, "sum": 3.0}
        rows = [main, launch_1, launch_2, launch_3]

        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])])
        self.assertEqual(len(merged_roots), 1)
        merged_main = merged_roots[0]
        self.assertEqual(len(merged_main["children"]), 1)
        merged_launch = merged_main["children"]["hipLaunchKernel"]
        entry = merged_launch["per_rank"]["r0"]
        self.assertEqual(entry["count"], 3)
        self.assertAlmostEqual(entry["self_sum"], 6.0)
        self.assertAlmostEqual(entry["sum"], 6.0)

    def test_label_key_reads_display_name_from_a_different_column(self):
        # Trace-pipeline rows use "name", not "label" (see stage1_rocprofsys_trace.LABEL_KEY) --
        # merge_rank_trees() must read whichever key label_key names, not assume "label".
        main = {"name": "main", "parent": None, "count": 1, "self_sum": 0.0, "sum": 1.0}
        child = {"name": "jacobi_sweep", "parent": main, "count": 1, "self_sum": 1.0, "sum": 1.0}
        rows = [main, child]

        merged_roots = s4c.merge_rank_trees([("r0", rows, [main])], label_key="name")
        self.assertEqual(merged_roots[0]["label"], "main")
        self.assertEqual(merged_roots[0]["children"]["jacobi_sweep"]["label"], "jacobi_sweep")


class AggregateNodeStatsTests(unittest.TestCase):
    def test_missing_rank_counts_as_zero_not_omitted(self):
        per_rank = {"r0": {"count": 10, "self_sum": 2.0, "sum": 2.0}}
        stats = s4c.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
        self.assertAlmostEqual(stats["self_avg"], 2.0 / 3)
        self.assertAlmostEqual(stats["self_min"], 0.0)
        self.assertAlmostEqual(stats["self_max"], 2.0)

    def test_std_dev_and_min_max(self):
        per_rank = {
            "r0": {"count": 1, "self_sum": 1.0, "sum": 1.0},
            "r1": {"count": 1, "self_sum": 3.0, "sum": 3.0},
            "r2": {"count": 1, "self_sum": 5.0, "sum": 5.0},
        }
        stats = s4c.aggregate_node_stats(per_rank, ["r0", "r1", "r2"])
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

        chains = s4c.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main", "compute", "hot_function"])

    def test_two_distinct_call_sites_return_two_chains(self):
        main = make_row("main")
        compute_a = make_row("compute_a", parent=main)
        compute_b = make_row("compute_b", parent=main)
        target_a = make_row("hot_function", parent=compute_a)
        target_b = make_row("hot_function", parent=compute_b)
        rows = [main, compute_a, compute_b, target_a, target_b]

        chains = s4c.caller_chains_for_label(rows, "hot_function")
        self.assertEqual(len(chains), 2)
        labels = {tuple(n["label"] for n in chain) for chain in chains}
        self.assertEqual(labels, {
            ("main", "compute_a", "hot_function"),
            ("main", "compute_b", "hot_function"),
        })

    def test_no_match_returns_empty_list(self):
        main = make_row("main")
        rows = [main]
        self.assertEqual(s4c.caller_chains_for_label(rows, "never_called"), [])

    def test_target_is_a_root_returns_single_node_chain(self):
        main = make_row("main")
        rows = [main]
        chains = s4c.caller_chains_for_label(rows, "main")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main"])
        self.assertIsNone(chains[0][0]["parent"])


class KernelOwnerLabelTests(unittest.TestCase):
    def test_strips_ck_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label("jacobi_sweep$pressure_solver_mod_$ck_L36_1_cce$noloop$form"),
            "jacobi_sweep$pressure_solver_mod_",
        )

    def test_unchanged_without_ck_marker(self):
        self.assertEqual(s4c.kernel_owner_label("JacobiIterationKernel"), "JacobiIterationKernel")

    # The following mirror the real rocprofv3 kernel_stats.csv "Name" values observed in
    # test_apps/results/ for each language/compiler combo (Cray Fortran, covered above, is the
    # only one NOT using this generic shape).
    def test_amd_c_omp_offloading_plain_name(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_4f_8fb8827_launch_omp_kernel_l6"),
            "launch_omp_kernel",
        )

    def test_cray_c_omp_offloading_plain_name_with_compiler_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_34_8fb8827_launch_omp_kernel_l6_cce$noloop$form"),
            "launch_omp_kernel",
        )

    def test_amd_cpp_omp_offloading_itanium_mangled_name(self):
        self.assertEqual(
            s4c.kernel_owner_label("__omp_offloading_4f_8fb8861__Z17launch_omp_kernelPdi_l6"),
            "launch_omp_kernel",
        )

    def test_cray_cpp_omp_offloading_itanium_mangled_name_with_compiler_suffix(self):
        self.assertEqual(
            s4c.kernel_owner_label(
                "__omp_offloading_34_8fb8861__Z17launch_omp_kernelPdi_l6_cce$noloop$form",
            ),
            "launch_omp_kernel",
        )

    def test_amd_fortran_omp_offloading_flang_mangled_name(self):
        self.assertEqual(
            s4c.kernel_owner_label(
                "__omp_offloading_4f_8fb7ea6__QMkernel_omp_modPlaunch_omp_kernel_l12",
            ),
            "launch_omp_kernel",
        )

    def test_hip_kernel_name_matches_neither_convention(self):
        # test_apps' native HIP kernel ("stencil_kernel") isn't affected by the OMPT-shared-entry-
        # point bug this decoding exists for -- its corr_id ancestry is already call-site-exact --
        # so it must return unchanged, same as any other unrecognized name.
        self.assertEqual(s4c.kernel_owner_label("stencil_kernel"), "stencil_kernel")


if __name__ == "__main__":
    unittest.main()
