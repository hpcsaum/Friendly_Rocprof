import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage4", "stage4_rocprofsys_trace_tree.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage4_rocprofsys_trace_tree", MODULE_PATH)
tree = importlib.util.module_from_spec(spec)
sys.modules["stage4_rocprofsys_trace_tree"] = tree
spec.loader.exec_module(tree)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
TWO_RANK_DIR = os.path.join(FIXTURES, "trace_two_rank")
RANK_INPUTS = [
    ("r0", os.path.join(TWO_RANK_DIR, "rank0.csv")),
    ("r1", os.path.join(TWO_RANK_DIR, "rank1.csv")),
]

TREE_NODE_CONTRACT_KEYS = {"label", "parent", "children", "per_rank", "tags", "structural_drop_tags", "static_children"}


def flatten(roots):
    flat = []

    def visit(node):
        flat.append(node)
        for child in node["children"].values():
            visit(child)

    for root in roots:
        visit(root)
    return flat


class MergeRanksTests(unittest.TestCase):
    def test_every_merged_node_matches_the_documented_tree_node_contract(self):
        roots = tree.merge_ranks(RANK_INPUTS)
        for node in flatten(roots):
            self.assertEqual(set(node.keys()), TREE_NODE_CONTRACT_KEYS)

    def test_both_ranks_contribute_to_the_same_merged_position(self):
        roots = tree.merge_ranks(RANK_INPUTS)
        main = next(n for n in roots if n["label"] == "main")
        self.assertEqual(set(main["per_rank"].keys()), {"r0", "r1"})
        self.assertAlmostEqual(main["per_rank"]["r0"]["sum"], 10.0)
        self.assertAlmostEqual(main["per_rank"]["r1"]["sum"], 12.0)

    def test_corr_id_joined_kernel_stays_nested_under_its_launch_site_after_cross_rank_merge(self):
        roots = tree.merge_ranks(RANK_INPUTS)
        main = next(n for n in roots if n["label"] == "main")
        sweep = main["children"]["jacobi_sweep"]
        launch = sweep["children"]["hipLaunchKernel"]
        self.assertIn("jacobi_kernel.kd", launch["children"])
        kernel = launch["children"]["jacobi_kernel.kd"]
        self.assertAlmostEqual(kernel["per_rank"]["r0"]["sum"], 3.0)
        self.assertAlmostEqual(kernel["per_rank"]["r1"]["sum"], 4.0)

    def test_tags_survive_the_cross_rank_merge(self):
        roots = tree.merge_ranks(RANK_INPUTS)
        main = next(n for n in roots if n["label"] == "main")
        sweep = main["children"]["jacobi_sweep"]
        launch = sweep["children"]["hipLaunchKernel"]
        mpi = sweep["children"]["MPI_Barrier"]
        self.assertIn("gpu_api", launch["tags"])
        self.assertIn("mpi_territory", mpi["tags"])
        self.assertIn("gpu_kernel", launch["children"]["jacobi_kernel.kd"]["tags"])

    def test_postprocess_hook_edits_each_ranks_rows_before_the_cross_rank_merge(self):
        # Mirrors stage4_rocprofsys_sample_tree.load_rank_trees()'s own postprocess contract:
        # a per-rank edit applied before roots are recomputed and before the merge.
        def drop_mpi_barrier(rows):
            return [row for row in rows if row["label"] != "MPI_Barrier"]

        roots = tree.merge_ranks(RANK_INPUTS, postprocess=drop_mpi_barrier)
        main = next(n for n in roots if n["label"] == "main")
        sweep = main["children"]["jacobi_sweep"]
        self.assertNotIn("MPI_Barrier", sweep["children"])
        self.assertIn("hipLaunchKernel", sweep["children"])

    def test_postprocess_hook_defaults_to_a_noop(self):
        with_none = tree.merge_ranks(RANK_INPUTS)
        with_explicit_none = tree.merge_ranks(RANK_INPUTS, postprocess=None)
        labels = lambda roots: sorted(n["label"] for n in flatten(roots))  # noqa: E731
        self.assertEqual(labels(with_none), labels(with_explicit_none))


if __name__ == "__main__":
    unittest.main()
