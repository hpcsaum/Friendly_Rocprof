import importlib.util
import json
import os
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage3", "stage3_rocprofsys_sample.py")

# stage3_rocprofsys_sample.py does a plain top-level "from stage6_noise_config import ...", relying on
# its own directory being on sys.path -- true automatically when it's run directly, but not when
# loaded here by explicit file path, so replicate that manually (same technique as other test
# files in this suite).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage3_rocprofsys_sample", MODULE_PATH)
s3 = importlib.util.module_from_spec(spec)
sys.modules["stage3_rocprofsys_sample"] = s3
spec.loader.exec_module(s3)

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)
from stage6_noise_config import load_default_patterns  # noqa: E402

# A minimal synthetic tag vocabulary, independent of the real shipped
# default_noise_patterns.json (covered separately by LoadDefaultPatternsTests) -- keeps these
# mechanism tests stable even if the real pattern lists change.
TAG_DEFS = {
    "gpu_api": {
        "prefixes": ["hip"],
        "suffixes": [".kd"],
        "filename_substrings": ["roctracer", "hsa"],
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
    "wrapper_branch_noise": {
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


class FilenameHintTests(unittest.TestCase):
    def test_row_in_matching_file_tags_positive_regardless_of_label(self):
        row = make_row("some_unrelated_label")
        s3.tag_rows([row], TAG_DEFS, filename="/path/to/roctracer-1234.txt")
        self.assertIn("gpu_api", row["tags"])

    def test_row_in_non_matching_file_unaffected(self):
        row = make_row("some_unrelated_label")
        s3.tag_rows([row], TAG_DEFS, filename="/path/to/wall_clock-1234.txt")
        self.assertEqual(row["tags"], set())

    def test_hint_applies_to_every_row_in_the_file(self):
        a = make_row("first_label")
        b = make_row("second_label")
        s3.tag_rows([a, b], TAG_DEFS, filename="/path/to/hsa-1234.txt")
        self.assertIn("gpu_api", a["tags"])
        self.assertIn("gpu_api", b["tags"])

    def test_no_filename_given_is_a_no_op(self):
        row = make_row("some_unrelated_label")
        s3.tag_rows([row], TAG_DEFS)
        self.assertEqual(row["tags"], set())


class SelfTagsTests(unittest.TestCase):
    def test_self_match_appears_in_both_tags_and_self_tags(self):
        row = make_row("hipLaunchKernel")
        s3.tag_rows([row], TAG_DEFS)
        self.assertIn("gpu_api", row["tags"])
        self.assertIn("gpu_api", row["self_tags"])

    def test_ancestor_only_match_appears_in_tags_not_self_tags(self):
        ancestor = make_row("mpi_init")
        thread_root = make_row("start_thread", parent=ancestor, is_thread_root=True)
        s3.tag_rows([ancestor, thread_root], TAG_DEFS)
        self.assertIn("mpi_territory", thread_root["tags"])
        self.assertNotIn("mpi_territory", thread_root["self_tags"])

    def test_filename_hint_counts_as_self_tag(self):
        row = make_row("some_unrelated_label")
        s3.tag_rows([row], TAG_DEFS, filename="/path/to/roctracer-1234.txt")
        self.assertIn("gpu_api", row["self_tags"])


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
        self.assertIn("wrapper_branch_noise", contaminated_top["structural_drop_tags"])
        self.assertEqual(clean_sibling["structural_drop_tags"], set())

    def test_structural_drop_tags_not_cascaded_to_descendants(self):
        # tag_rows() itself only marks the TOP of a contaminated subtree -- a caller
        # iterating rows as a flat list (not remove_tagged_subtrees()'s recursive removal)
        # must not assume every descendant carries the tag too. See tag_rows()'s docstring.
        root = make_row("main")
        contaminated_top = make_row("std::pair<...>", parent=root)
        buried = make_row("gotcha_call", parent=contaminated_top)
        clean_sibling = make_row("run_simulation", parent=root)
        rows = [root, contaminated_top, buried, clean_sibling]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertEqual(buried["structural_drop_tags"], set())

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
        # Proves the mechanism itself works generically for any tag configured with
        # first_real_descendant_skip_tag, using mpi_territory as the example tag here --
        # NOT a claim that the real shipped default_noise_patterns.json enables this for
        # mpi_territory. It deliberately doesn't: see
        # test_real_root_with_matching_first_child_is_a_known_limitation below for why.
        root = make_row("start_thread")
        wrapper_hop = make_row("gotcha_call", parent=root)
        real_child = make_row("mpi_init", parent=wrapper_hop)
        rows = [root, wrapper_hop, real_child]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertIn("mpi_territory", root["tags"])

    def test_real_root_with_matching_first_child_is_a_known_limitation(self):
        # The mechanism can't tell a genuine, real program root apart from an untethered
        # thread-spawn artifact -- both just have parent=None. A real root whose very first
        # traced child happens to match the tag gets it inherited too, same as an actual
        # untethered noise root would. This is exactly why default_noise_patterns.json does
        # NOT enable first_real_descendant_skip_tag for mpi_territory: a real "main" whose
        # first call is MPI_Init (extremely common) would otherwise be misclassified.
        # Confirmed via test_extract_CPU_hotspots.py's mpi_spawned_thread_noise fixture.
        main = make_row("main")
        mpi_init = make_row("mpi_init", parent=main)
        rows = [main, mpi_init]
        s3.tag_rows(rows, TAG_DEFS)
        self.assertIn("mpi_territory", main["tags"])

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
        result = s3.remove_tagged_subtrees(rows, {"wrapper_branch_noise"})
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
        is_pruned = s3.make_is_pruned({"gpu_api", "wrapper_branch_noise"})
        self.assertTrue(is_pruned({"tags": {"gpu_api"}, "structural_drop_tags": set()}))
        self.assertTrue(is_pruned({"tags": set(), "structural_drop_tags": {"wrapper_branch_noise"}}))
        self.assertFalse(is_pruned({"tags": set(), "structural_drop_tags": set()}))


class LoadDefaultPatternsTests(unittest.TestCase):
    def test_loads_expected_tag_names(self):
        patterns = load_default_patterns()
        self.assertEqual(
            set(patterns.keys()),
            {"gpu_api", "wrapper_noise", "mpi_territory", "compiler_runtime_noise", "wrapper_branch_noise"},
        )


class OpenMpiPrefixTests(unittest.TestCase):
    # ompi_/opal_/orte_ -- no real Open MPI test_apps capture exists yet, added as a
    # "most probable" list per real Open MPI naming conventions. Against the REAL
    # shipped patterns (not the synthetic TAG_DEFS above), since this is specifically
    # about default_noise_patterns.json's own mpi_territory prefix list.
    def test_matches_open_mpi_prefixes(self):
        real_defs = load_default_patterns()
        mpi_def = real_defs["mpi_territory"]
        self.assertTrue(s3._label_matches("ompi_request_complete", mpi_def))
        self.assertTrue(s3._label_matches("opal_progress", mpi_def))
        self.assertTrue(s3._label_matches("orte_grpcomm_base_pack", mpi_def))

    def test_rejects_mid_string_match_inside_gotcha_template(self):
        # startswith-based, so this must NOT match an Open-MPI opaque-handle typename
        # appearing mid-string inside rocprof-sys's own generic GOTCHA-wrapper template
        # signature.
        real_defs = load_default_patterns()
        mpi_def = real_defs["mpi_territory"]
        self.assertFalse(s3._label_matches(
            "tim::component::gotcha<101ul, int, ompi_group_t**>::construct", mpi_def
        ))


class TagRowsFallbackTests(unittest.TestCase):
    # tag_rows()'s own fallback to stage6_noise_config.tag_defs() -- every other test in this
    # file passes an explicit hand-built tag_defs and is unaffected by any of this.
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_omitting_tag_defs_falls_back_to_the_process_wide_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_config.json")
            with open(path, "w") as f:
                json.dump({"add": {"other": ["my_custom_noise_"]}}, f)
            stage6_noise_config.configure(path)

        row = make_row("my_custom_noise_helper")
        s3.tag_rows([row], filename=None)  # no tag_defs given
        self.assertIn("other", row["tags"])

    def test_explicit_tag_defs_still_overrides_the_process_wide_config(self):
        stage6_noise_config.configure(None)
        row = make_row("my_custom_noise_helper")
        s3.tag_rows([row], TAG_DEFS)  # this file's own synthetic dict, has no "other" tag at all
        self.assertEqual(row["tags"], set())


if __name__ == "__main__":
    unittest.main()
