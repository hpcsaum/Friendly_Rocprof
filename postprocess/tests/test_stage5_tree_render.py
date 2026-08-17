import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location(
    "stage5_tree_render", os.path.join(POSTPROCESS_DIR, "stage5_tree_render.py")
)
tr = importlib.util.module_from_spec(spec)
sys.modules["stage5_tree_render"] = tr
spec.loader.exec_module(tr)

# render_forest() takes a node_values(node) callable -- in real usage this always
# comes from stage4_rocprofsys_tree.make_node_values(), so the fixture here uses the
# same real function rather than a stand-in, mirroring production code's own
# dependency between the two modules.
spec4 = importlib.util.spec_from_file_location("stage4_rocprofsys_tree", os.path.join(POSTPROCESS_DIR, "stage4_rocprofsys_tree.py"))
s4t = importlib.util.module_from_spec(spec4)
sys.modules["stage4_rocprofsys_tree"] = s4t
spec4.loader.exec_module(s4t)

from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs sys.path insert above first)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")
SAMPLING_FALLBACK_DIR = os.path.join(FIXTURES, "calltree_sampling_fallback")

RANK = "r0"  # every hand-built test tree in this file simulates one rank


def make_row(label, parent=None, count=1, self_sum=0.0, total_sum=None, gpu=False):
    """A minimal hand-built merged-tree node -- same shape merge_rank_trees()
    produces (a "per_rank" dict, not flat count/self_sum/sum fields), without
    needing a fixture file or a real multi-rank merge."""
    total_sum = total_sum if total_sum is not None else self_sum
    return {
        "label": label, "parent": parent, "children": {}, "static_children": [],
        "gpu": gpu, "compiler_runtime": False, "mpi_territory": False,
        "per_rank": {RANK: {"count": count, "self_sum": self_sum, "sum": total_sum}},
    }


NEVER_PRUNED = lambda node: False  # noqa: E731
DEFAULT_NODE_VALUES = s4t.make_node_values([RANK])
DEFAULT_HEADERS = [("CALLS", 8, "d"), ("SELF(s)", 12, ".6f"), ("TOTAL(s)", 12, ".6f")]


class RenderingTests(unittest.TestCase):
    def test_render_forest_uses_tree_connectors(self):
        main = make_row("main", count=1, self_sum=0.0, total_sum=10.0)
        child_a = make_row("child_a", parent=main, count=1, self_sum=4.0, total_sum=4.0)
        child_b = make_row("child_b", parent=main, count=1, self_sum=6.0, total_sum=6.0)
        rows = [main, child_a, child_b]
        children_map = tr.build_children_map(rows)

        out = tr.render_forest([main], children_map, None, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        labels = [text for text, _values in out]
        self.assertEqual(labels[0], "main")  # root prints flush, no connector
        self.assertTrue(labels[1].startswith("├── child_a"))
        self.assertTrue(labels[2].startswith("└── child_b"))

    def test_max_depth_truncates_with_hidden_count(self):
        main = make_row("main")
        child = make_row("child", parent=main)
        grandchild = make_row("grandchild", parent=child)
        rows = [main, child, grandchild]
        children_map = tr.build_children_map(rows)

        out = tr.render_forest([main], children_map, 0, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        marker_lines = [text for text, values in out if values is None]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("2 more node(s) hidden", marker_lines[0])

    def test_build_children_map_collapse_hides_grandchildren(self):
        main = make_row("main")
        mpi_call = make_row("MPI_Allreduce", parent=main)
        internal = make_row("MPIR_Allreduce_cdesc", parent=mpi_call)
        rows = [main, mpi_call, internal]

        children_map = tr.build_children_map(rows, collapses_children=lambda row: row["label"] == "MPI_Allreduce")
        out = tr.render_forest([main], children_map, None, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        labels = [text for text, _values in out]
        self.assertTrue(any("MPI_Allreduce" in l for l in labels))
        self.assertFalse(any("MPIR_Allreduce_cdesc" in l for l in labels))


class FormatAlignedRowsTests(unittest.TestCase):
    def test_real_columns_not_bracketed_string(self):
        rows = [("main", (10, 1.5, 3.0))]
        text = tr.format_aligned_rows(rows, DEFAULT_HEADERS)
        self.assertIn("CALLS", text)
        self.assertIn("SELF(s)", text)
        self.assertNotIn("[calls=", text)

    def test_none_value_renders_as_dash(self):
        headers = [("CALLS", 8, "d"), ("EXTRA(s)", 12, ".6f")]
        rows = [("main", (10, None))]
        text = tr.format_aligned_rows(rows, headers)
        self.assertIn("-", text.splitlines()[-1])

    def test_empty_block_renders_as_empty_string(self):
        self.assertEqual(tr.format_aligned_rows([], DEFAULT_HEADERS), "")

    def test_long_label_wraps_without_dragging_sibling_alignment(self):
        # End-to-end: format_aligned_rows() actually calls wrap_leading_labels() -- a short
        # sibling row's numeric columns stay aligned at the capped width, not dragged out to the
        # long label's full 200 characters.
        long_label = "z" * 200
        rows = [("short", (1, 2.0, 3.0)), (long_label, (4, 5.0, 6.0))]
        suffix_width = sum(2 + width for _name, width, _fmt in DEFAULT_HEADERS)
        expected_label_width, _ = tr.wrap_leading_labels(rows, suffix_width)
        text = tr.format_aligned_rows(rows, DEFAULT_HEADERS)
        lines = text.splitlines()

        self.assertEqual(lines[1][expected_label_width:expected_label_width + 2], "  ")
        self.assertEqual(lines[2][expected_label_width:expected_label_width + 2], "  ")
        # header + short row + long row's first line + 2 continuation lines (ceil(200/available) - 1)
        self.assertEqual(len(lines), 5)
        for cont in lines[3:]:
            self.assertEqual(cont, cont.lstrip(" "))  # flush left, not indented
            self.assertFalse(any(ch.isdigit() for ch in cont))  # no numeric cells


class WrapLeadingLabelsTests(unittest.TestCase):
    def test_labels_under_cap_are_unaffected(self):
        rows = [("main", (10, 1.5, 3.0)), ("child", (5, 0.5, 1.0))]
        label_width, wrapped = tr.wrap_leading_labels(rows, suffix_width=38)
        self.assertEqual(label_width, max(len("main"), len("child")))  # today's exact formula
        # every data row's text comes back as a 1-element chunk list, ready for the caller to
        # print its only element -- byte-identical content either way, just always list-shaped
        self.assertEqual(wrapped, [(["main"], (10, 1.5, 3.0)), (["child"], (5, 0.5, 1.0))])

    def test_short_labels_wrapped_as_single_element_chunk_lists(self):
        rows = [("main", (10, 1.5, 3.0))]
        _label_width, wrapped = tr.wrap_leading_labels(rows, suffix_width=1000)
        # a huge suffix_width forces available down to the 20-column floor, but "main" still
        # fits comfortably -- still returned as a 1-element chunk list, not wrapped
        self.assertEqual(wrapped, [(["main"], (10, 1.5, 3.0))])

    def test_long_label_is_capped_and_reconstructs_when_joined(self):
        long_label = "x" * 200
        rows = [("short", (1, 2, 3)), (long_label, (4, 5, 6))]
        label_width, wrapped = tr.wrap_leading_labels(rows, suffix_width=38)  # available == 82
        self.assertEqual(label_width, 82)  # capped, not dragged to 200 by the one long row
        self.assertEqual(wrapped[0], (["short"], (1, 2, 3)))
        chunks, values = wrapped[1]
        self.assertEqual(values, (4, 5, 6))
        self.assertEqual(len(chunks), 3)  # ceil(200 / 82)
        self.assertEqual("".join(chunks), long_label)

    def test_marker_row_passes_through_untouched(self):
        rows = [("main", (1, 2, 3)), ("... (2 more node(s) hidden)", None)]
        _label_width, wrapped = tr.wrap_leading_labels(rows, suffix_width=38)
        self.assertEqual(wrapped[1], ("... (2 more node(s) hidden)", None))


class LoadRankTreesTests(unittest.TestCase):
    def test_primary_pattern_file_used_when_present(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = tr.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)

    def test_fallback_pattern_used_only_when_primary_missing(self):
        # calltree_sampling_fallback: one rank has both files (primary wins), the other
        # rank has only the fallback pattern's file.
        ranks = tr.load_rank_trees(SAMPLING_FALLBACK_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt")
        self.assertEqual(len(ranks), 2)
        labels_by_rank = {rk: {r["label"] for r in rows} for rk, rows, _roots in ranks}
        self.assertIn({"main_fallback", "instrumented_leaf"}, labels_by_rank.values())

    def test_swapping_which_pattern_is_primary_changes_the_winner(self):
        # Same fixture, patterns reversed -- the rank with both files now takes its data
        # from wall_clock instead of sampling_wall_clock.
        sampling_first = tr.load_rank_trees(SAMPLING_FALLBACK_DIR, "sampling_wall_clock-*.txt", "wall_clock-*.txt")
        wall_clock_first = tr.load_rank_trees(SAMPLING_FALLBACK_DIR, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        sampling_labels = {rk: {r["label"] for r in rows} for rk, rows, _roots in sampling_first}
        wall_clock_labels = {rk: {r["label"] for r in rows} for rk, rows, _roots in wall_clock_first}
        self.assertNotEqual(sampling_labels, wall_clock_labels)

    def test_postprocess_hook_defaults_to_noop(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)
        ranks = tr.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        _rank_key, rows, _roots = ranks[0]
        self.assertTrue(any(r["label"] == "hipMemcpy" for r in rows))  # nothing stripped by default

    def test_postprocess_hook_applied_when_given(self):
        cpu_dir, _gpu_dir = resolve_run_dirs(MPI_2RANK_DIR)

        def _drop_hipmemcpy(rows):
            return [r for r in rows if r["label"] != "hipMemcpy"]

        ranks = tr.load_rank_trees(
            cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt", postprocess=_drop_hipmemcpy
        )
        _rank_key, rows, _roots = ranks[0]
        self.assertFalse(any(r["label"] == "hipMemcpy" for r in rows))


class KernelTotalsWithCountsTests(unittest.TestCase):
    def test_returns_count_and_seconds_per_kernel(self):
        _cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        totals = tr.kernel_totals_with_counts(gpu_dir, 0)
        self.assertIn("JacobiIterationKernel", totals)
        count, seconds = totals["JacobiIterationKernel"]
        self.assertGreater(count, 0)
        self.assertGreater(seconds, 0)


class PairGpuPerRankTests(unittest.TestCase):
    def test_returns_none_when_gpu_dir_is_none(self):
        self.assertIsNone(tr.pair_gpu_per_rank(None, "run", ["r0"]))

    def test_pairs_when_rank_counts_match(self):
        cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        ranks = tr.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        rank_keys = [rk for rk, _rows, _roots in ranks]
        result = tr.pair_gpu_per_rank(gpu_dir, KERNEL_ANCHOR_DIR, rank_keys)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(rank_keys))

    def test_returns_none_on_mismatched_rank_count(self):
        _cpu_dir, gpu_dir = resolve_run_dirs(KERNEL_ANCHOR_DIR)
        result = tr.pair_gpu_per_rank(gpu_dir, KERNEL_ANCHOR_DIR, ["r0", "r1"])  # gpu side has only 1 file
        self.assertIsNone(result)


class AttachAndRenderGpuKernelsTests(unittest.TestCase):
    def _build_tree(self, run_dir):
        cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
        ranks = tr.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        rank_keys = [rk for rk, _rows, _roots in ranks]
        merged_roots = s4t.merge_rank_trees(ranks)
        flat = s4t.flatten_tree(merged_roots)
        node_values = s4t.make_node_values(rank_keys)
        gpu_per_rank = tr.pair_gpu_per_rank(gpu_dir, run_dir, rank_keys)
        return flat, rank_keys, node_values, gpu_dir, gpu_per_rank

    def test_returns_empty_string_when_no_gpu_data(self):
        result = tr.attach_and_render_gpu_kernels([], None, None, [], NEVER_PRUNED, DEFAULT_NODE_VALUES)
        self.assertEqual(result, "")

    def test_attaches_matched_kernel_and_returns_no_fallback(self):
        flat, rank_keys, node_values, gpu_dir, gpu_per_rank = self._build_tree(KERNEL_ANCHOR_DIR)
        fallback = tr.attach_and_render_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, NEVER_PRUNED, node_values)
        self.assertEqual(fallback, "")  # matched by name -- nothing left unattached
        has_kernel_group = any(
            any("GPU kernels" in c["label"] for c in node.get("static_children", []))
            for node in flat
        )
        self.assertTrue(has_kernel_group)

    def test_unmatched_kernel_produces_fallback_text(self):
        flat, rank_keys, node_values, gpu_dir, gpu_per_rank = self._build_tree(KERNEL_NO_ANCHOR_DIR)
        fallback = tr.attach_and_render_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, NEVER_PRUNED, node_values)
        self.assertIn("=== GPU kernels (rocprofv3)", fallback)
        self.assertIn("JacobiIterationKernel", fallback)
        # ends in exactly the table's own trailing newline -- no self-appended blank line;
        # any blank line between sections is the caller's (render_report()'s) job now.
        self.assertFalse(fallback.endswith("\n\n"))
        self.assertTrue(fallback.endswith("\n"))


class RenderCalltreeTextTests(unittest.TestCase):
    def test_renders_tree_ending_in_single_newline(self):
        main = make_row("main", count=1, self_sum=0.0, total_sum=10.0)
        child = make_row("child", parent=main, count=1, self_sum=4.0, total_sum=4.0)
        rows = [main, child]
        text = tr.render_calltree_text([main], rows, None, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        self.assertIn("main", text)
        self.assertIn("child", text)
        # ends in exactly the table's own trailing newline -- no self-appended blank line;
        # any blank line between sections is the caller's (render_report()'s) job now.
        self.assertFalse(text.endswith("\n\n"))
        self.assertTrue(text.endswith("\n"))

    def test_collapses_children_hides_grandchildren(self):
        main = make_row("main")
        mpi_call = make_row("MPI_Allreduce", parent=main)
        internal = make_row("MPIR_Allreduce_cdesc", parent=mpi_call)
        rows = [main, mpi_call, internal]
        text = tr.render_calltree_text(
            [main], rows, None, NEVER_PRUNED, DEFAULT_NODE_VALUES,
            collapses_children=lambda row: row["label"] == "MPI_Allreduce",
        )
        self.assertIn("MPI_Allreduce", text)
        self.assertNotIn("MPIR_Allreduce_cdesc", text)


class AggregationNoteTests(unittest.TestCase):
    def test_fixed_bulleted_text_lowercase_columns(self):
        note = tr.aggregation_note()
        self.assertTrue(note.startswith("  - "))
        self.assertIn("calls and", note)
        self.assertIn("total-avg(s)", note)
        self.assertNotIn("CALLS", note)
        self.assertNotIn("SELF-AVG", note)


class TreeViewNoteTests(unittest.TestCase):
    def test_all_four_tiers(self):
        note = tr.tree_view_note(
            ["1001", "1002"], 5, show_gpu_api=True, show_rocprofsys_internals=False,
            show_mpi_internals=False, show_compiler_runtime=True,
        )
        self.assertIn("Ranks aggregated: 1001, 1002\n", note)
        self.assertIn("GPU-API/runtime noise (shown)", note)
        self.assertIn("rocprof-sys internals (hidden)", note)
        self.assertIn("MPI internals (hidden)", note)
        self.assertIn("compiler-runtime helpers (shown)", note)
        self.assertIn("Max depth: 5\n", note)

    def test_only_gpu_api_tier_when_others_omitted(self):
        note = tr.tree_view_note(["1001"], None, show_gpu_api=False)
        self.assertIn("GPU-API/runtime noise (hidden)", note)
        self.assertNotIn("rocprof-sys internals", note)
        self.assertNotIn("MPI internals", note)
        self.assertNotIn("compiler-runtime helpers", note)
        self.assertIn("Max depth: unlimited\n", note)


if __name__ == "__main__":
    unittest.main()
