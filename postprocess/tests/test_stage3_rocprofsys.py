import importlib.util
import os
import sys
import unittest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "stage3_rocprofsys.py")

spec = importlib.util.spec_from_file_location("stage3_rocprofsys", MODULE_PATH)
s3 = importlib.util.module_from_spec(spec)
sys.modules["stage3_rocprofsys"] = s3
spec.loader.exec_module(s3)

# A minimal synthetic tag vocabulary, independent of the real shipped
# default_noise_patterns.json (covered separately by LoadDefaultPatternsTests) -- keeps these
# mechanism tests stable even if the real pattern lists change.
TAG_DEFS = {
    "gpu_api": {
        "prefixes": ["hip"],
        "suffixes": [".kd"],
        "ancestor_for_thread_roots": True,
        "first_real_descendant_skip_tag": "wrapper_noise",
    },
    "wrapper_noise": {
        "substrings": ["gotcha"],
    },
    "mpi_territory": {
        "prefixes": ["mpi_"],
        "suffixes": ["_f08ts_"],
        "ancestor_for_thread_roots": True,
        "first_real_descendant_skip_tag": "wrapper_noise",
    },
    "wrapper_branch_contamination": {
        "sibling_group_source_tag": "wrapper_noise",
    },
}


def make_row(label, parent=None, is_thread_root=False, self_sum=0.0, sum_=0.0):
    return {"label": label, "parent": parent, "is_thread_root": is_thread_root, "self_sum": self_sum, "sum": sum_}


class SelfScopeMatchingTests(unittest.TestCase):
    def test_prefix_match(self):
        row = make_row("hipLaunchKernel")
        s3.tag_rows([row], TAG_DEFS)
        self.assertIn("gpu_api", row["tags"])

    def test_substring_match(self):
        row = make_row("gotcha_wrap")
        s3.tag_rows([row], TAG_DEFS)
        self.assertIn("wrapper_noise", row["tags"])

    def test_suffix_match_mpi_fortran_shim(self):
        # closes a real coverage gap: no existing test fed a _f08ts_-suffixed label to
        # is_mpi_territory() directly before this.
        row = make_row("mpi_allreduce_f08ts_")
        s3.tag_rows([row], TAG_DEFS)
        self.assertIn("mpi_territory", row["tags"])

    def test_suffix_match_kd_is_case_insensitive(self):
        lower = make_row("some_kernel.kd")
        upper = make_row("some_kernel.KD")
        s3.tag_rows([lower, upper], TAG_DEFS)
        self.assertIn("gpu_api", lower["tags"])
        self.assertIn("gpu_api", upper["tags"])

    def test_rejects_unrelated_label(self):
        row = make_row("compute_stencil")
        s3.tag_rows([row], TAG_DEFS)
        self.assertEqual(row["tags"], set())


class AncestorForThreadRootsTests(unittest.TestCase):
    def test_thread_root_with_gpu_ancestor_tags_positive(self):
        ancestor = make_row("hipLaunchKernel")
        thread_root = make_row("start_thread", parent=ancestor, is_thread_root=True)
        s3.tag_rows([ancestor, thread_root], TAG_DEFS)
        self.assertIn("gpu_api", thread_root["tags"])

    def test_thread_root_with_mpi_ancestor_tags_positive(self):
        ancestor = make_row("mpi_init")
        thread_root = make_row("start_thread", parent=ancestor, is_thread_root=True)
        s3.tag_rows([ancestor, thread_root], TAG_DEFS)
        self.assertIn("mpi_territory", thread_root["tags"])

    def test_non_thread_root_ignores_matching_ancestor(self):
        ancestor = make_row("hipLaunchKernel")
        child = make_row("start_thread", parent=ancestor, is_thread_root=False)
        s3.tag_rows([ancestor, child], TAG_DEFS)
        self.assertEqual(child["tags"], set())

    def test_own_direct_match_wins_regardless_of_ancestry(self):
        ancestor = make_row("compute_stencil")
        thread_root = make_row("hipLaunchKernel", parent=ancestor, is_thread_root=True)
        s3.tag_rows([ancestor, thread_root], TAG_DEFS)
        self.assertIn("gpu_api", thread_root["tags"])


class SiblingGroupTests(unittest.TestCase):
    def test_contaminated_sibling_marked_clean_sibling_untouched(self):
        root = make_row("main")
        contaminated_top = make_row("std::pair<...>", parent=root)
        buried = make_row("gotcha_call", parent=contaminated_top)
        clean_sibling = make_row("run_simulation", parent=root)
        rows = [root, contaminated_top, buried, clean_sibling]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertIn("wrapper_branch_contamination", contaminated_top["structural_drop_tags"])
        self.assertEqual(clean_sibling["structural_drop_tags"], set())

    def test_linear_no_sibling_chain_is_not_marked(self):
        root = make_row("root_frame")
        mid = make_row("mid_frame", parent=root)
        buried = make_row("gotcha_call", parent=mid)
        rows = [root, mid, buried]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertEqual(mid["structural_drop_tags"], set())
        self.assertEqual(root["structural_drop_tags"], set())

    def test_directly_matching_sibling_excluded_from_derivation(self):
        root = make_row("main")
        direct_match = make_row("gotcha_wrapper_call", parent=root)
        real_child_under = make_row("real_work", parent=direct_match)
        clean_sibling = make_row("compute_stencil", parent=root)
        rows = [root, direct_match, real_child_under, clean_sibling]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertEqual(direct_match["structural_drop_tags"], set())

    def test_all_contaminated_siblings_untouched(self):
        root = make_row("main")
        a = make_row("branch_a", parent=root)
        gotcha_a = make_row("gotcha_a", parent=a)
        b = make_row("branch_b", parent=root)
        gotcha_b = make_row("gotcha_b", parent=b)
        rows = [root, a, gotcha_a, b, gotcha_b]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertEqual(a["structural_drop_tags"], set())
        self.assertEqual(b["structural_drop_tags"], set())


class FirstRealDescendantTests(unittest.TestCase):
    def test_gpu_api_inherited_through_wrapper_hop(self):
        root = make_row("start_thread")
        wrapper_hop = make_row("gotcha_call", parent=root)
        real_child = make_row("hipLaunchKernel", parent=wrapper_hop)
        rows = [root, wrapper_hop, real_child]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertIn("gpu_api", root["tags"])

    def test_mpi_territory_inherited_through_wrapper_hop(self):
        # proves the scope-5 generalization works mechanically for a tag other than gpu_api --
        # still unconfirmed whether this shape occurs in real captures, tracked separately.
        root = make_row("start_thread")
        wrapper_hop = make_row("gotcha_call", parent=root)
        real_child = make_row("mpi_init", parent=wrapper_hop)
        rows = [root, wrapper_hop, real_child]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertIn("mpi_territory", root["tags"])

    def test_already_self_tagged_root_is_a_no_op(self):
        root = make_row("hipLaunchKernel")
        s3.tag_rows([root], TAG_DEFS)
        self.assertIn("gpu_api", root["tags"])

    def test_non_untethered_root_never_inherits(self):
        parent = make_row("main")
        child = make_row("start_thread", parent=parent)
        gpu_grandchild = make_row("hipLaunchKernel", parent=child)
        rows = [parent, child, gpu_grandchild]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertEqual(child["tags"], set())


class RemoveTaggedSubtreesTests(unittest.TestCase):
    def test_tagged_node_and_whole_subtree_removed(self):
        root = make_row("main")
        noisy = make_row("gotcha_call", parent=root)
        child_of_noisy = make_row("hidden_child", parent=noisy)
        sibling = make_row("compute_stencil", parent=root)
        rows = [root, noisy, child_of_noisy, sibling]
        s3.tag_rows(rows, TAG_DEFS)
        result = s3.remove_tagged_subtrees(rows, {"wrapper_noise"})
        self.assertEqual({r["label"] for r in result}, {"main", "compute_stencil"})

    def test_structural_drop_tags_also_removed(self):
        root = make_row("main")
        contaminated_top = make_row("std::pair<...>", parent=root)
        buried = make_row("gotcha_call", parent=contaminated_top)
        clean_sibling = make_row("run_simulation", parent=root)
        rows = [root, contaminated_top, buried, clean_sibling]
        s3.tag_rows(rows, TAG_DEFS)
        result = s3.remove_tagged_subtrees(rows, {"wrapper_branch_contamination"})
        self.assertEqual({r["label"] for r in result}, {"main", "run_simulation"})


class SpliceByTagTests(unittest.TestCase):
    def test_children_reparented_and_matched_rows_removed(self):
        root = make_row("main")
        wrapper = make_row("gotcha_call", parent=root)
        real_child = make_row("compute_stencil", parent=wrapper)
        rows = [root, wrapper, real_child]
        s3.tag_rows(rows, TAG_DEFS)
        result = s3.splice_by_tag(rows, "wrapper_noise", fold=False)
        self.assertEqual({r["label"] for r in result}, {"main", "compute_stencil"})
        self.assertIs(real_child["parent"], root)

    def test_whole_ancestor_chain_matched_promotes_new_root(self):
        wrapper1 = make_row("gotcha_a")
        wrapper2 = make_row("gotcha_b", parent=wrapper1)
        real_root_candidate = make_row("main", parent=wrapper2)
        rows = [wrapper1, wrapper2, real_root_candidate]
        s3.tag_rows(rows, TAG_DEFS)
        result = s3.splice_by_tag(rows, "wrapper_noise", fold=False)
        self.assertEqual([r["label"] for r in result], ["main"])
        self.assertIsNone(real_root_candidate["parent"])

    def test_fold_true_adds_removed_self_time_to_new_parent(self):
        root = make_row("main", self_sum=1.0, sum_=10.0)
        wrapper = make_row("gotcha_call", parent=root, self_sum=2.0, sum_=2.0)
        child = make_row("compute_stencil", parent=wrapper, self_sum=3.0, sum_=3.0)
        rows = [root, wrapper, child]
        s3.tag_rows(rows, TAG_DEFS)
        s3.splice_by_tag(rows, "wrapper_noise", fold=True)
        self.assertAlmostEqual(root["self_sum"], 3.0)
        self.assertAlmostEqual(root["sum"], 12.0)
        self.assertAlmostEqual(child["self_sum"], 3.0)

    def test_fold_false_discards_removed_self_time(self):
        root = make_row("main", self_sum=1.0, sum_=10.0)
        wrapper = make_row("gotcha_call", parent=root, self_sum=2.0, sum_=2.0)
        rows = [root, wrapper]
        s3.tag_rows(rows, TAG_DEFS)
        s3.splice_by_tag(rows, "wrapper_noise", fold=False)
        self.assertAlmostEqual(root["self_sum"], 1.0)
        self.assertAlmostEqual(root["sum"], 10.0)


class ClosureTests(unittest.TestCase):
    def test_make_collapses_children_true_for_tagged_row(self):
        collapses = s3.make_collapses_children({"mpi_territory"})
        self.assertTrue(collapses({"tags": {"mpi_territory"}}))
        self.assertFalse(collapses({"tags": set()}))

    def test_make_is_pruned_checks_both_tag_fields(self):
        is_pruned = s3.make_is_pruned({"gpu_api", "wrapper_branch_contamination"})
        self.assertTrue(is_pruned({"tags": {"gpu_api"}, "structural_drop_tags": set()}))
        self.assertTrue(is_pruned({"tags": set(), "structural_drop_tags": {"wrapper_branch_contamination"}}))
        self.assertFalse(is_pruned({"tags": set(), "structural_drop_tags": set()}))


class LoadDefaultPatternsTests(unittest.TestCase):
    def test_loads_expected_tag_names(self):
        patterns = s3.load_default_patterns()
        self.assertEqual(
            set(patterns.keys()),
            {"gpu_api", "wrapper_noise", "mpi_territory", "compiler_runtime_noise", "wrapper_branch_contamination"},
        )


if __name__ == "__main__":
    unittest.main()
