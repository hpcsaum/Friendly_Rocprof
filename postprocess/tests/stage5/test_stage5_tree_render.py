"""Tests for stage5_tree_render.py -- shared calltree rendering: drawing tree connectors and
truncating by depth (render_forest()), aligning rows into fixed-width columns
(format_aligned_rows()/wrap_leading_labels()), the GPU-kernel-fallback table
(render_gpu_kernel_fallback()), the render_calltree_text() end-to-end wrapper, and the two calltree
tools' shared bulleted-note text.

RenderingTests                -- render_forest() connectors, max-depth truncation, collapses_children
FormatAlignedRowsTests        -- format_aligned_rows() column formatting and long-label wrapping
WrapLeadingLabelsTests        -- wrap_leading_labels() chunking and capped label-column width
RenderGpuKernelFallbackTests  -- fallback table text for kernels attach_gpu_kernels() left unattached
RenderCalltreeTextTests       -- render_calltree_text() end to end, incl. collapses_children
AggregationNoteTests          -- aggregation_note()'s fixed prose
TreeViewNoteTests             -- tree_view_note()'s per-tier (shown/hidden) prose
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

tr = load_module_by_path("stage5_tree_render", "stage5", "stage5_tree_render.py")

# render_forest() takes a node_values(node) callable -- in real usage this always
# comes from stage4_rocprofsys_common.make_node_values(), so the fixture here uses the
# same real function rather than a stand-in, mirroring production code's own
# dependency between the two modules.
s4t = load_module_by_path("stage4_rocprofsys_sample_tree", "stage4", "stage4_rocprofsys_sample_tree.py")

s4c = load_module_by_path("stage4_rocprofsys_common", "stage4", "stage4_rocprofsys_common.py")

from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs sys.path insert above first)

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
KERNEL_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_anchor")
KERNEL_NO_ANCHOR_DIR = os.path.join(FIXTURES, "calltree_kernel_no_anchor")

RANK = "r0"  # every hand-built test tree in this file simulates one rank


def make_merged_tree_node(label, parent=None, count=1, self_sum=0.0, total_sum=None, gpu=False):
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
DEFAULT_NODE_VALUES = s4c.make_node_values([RANK])
DEFAULT_HEADERS = [("CALLS", 8, "d"), ("SELF(s)", 12, ".6f"), ("TOTAL(s)", 12, ".6f")]


class RenderingTests(unittest.TestCase):
    def test_render_forest_uses_tree_connectors(self):
        main = make_merged_tree_node("main", count=1, self_sum=0.0, total_sum=10.0)
        child_a = make_merged_tree_node("child_a", parent=main, count=1, self_sum=4.0, total_sum=4.0)
        child_b = make_merged_tree_node("child_b", parent=main, count=1, self_sum=6.0, total_sum=6.0)
        rows = [main, child_a, child_b]
        children_map = tr.build_children_map(rows)

        out = tr.render_forest([main], children_map, None, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        labels = [text for text, _values in out]
        self.assertEqual(labels[0], "main")  # root prints flush, no connector
        self.assertTrue(labels[1].startswith("├── child_a"))
        self.assertTrue(labels[2].startswith("└── child_b"))

    def test_max_depth_truncates_with_hidden_count(self):
        main = make_merged_tree_node("main")
        child = make_merged_tree_node("child", parent=main)
        grandchild = make_merged_tree_node("grandchild", parent=child)
        rows = [main, child, grandchild]
        children_map = tr.build_children_map(rows)

        out = tr.render_forest([main], children_map, 0, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        marker_lines = [text for text, values in out if values is None]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("2 more node(s) hidden", marker_lines[0])

    def test_build_children_map_collapse_hides_grandchildren(self):
        main = make_merged_tree_node("main")
        mpi_call = make_merged_tree_node("MPI_Allreduce", parent=main)
        internal = make_merged_tree_node("MPIR_Allreduce_cdesc", parent=mpi_call)
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


class RenderGpuKernelFallbackTests(unittest.TestCase):
    def _attach(self, run_dir):
        """Builds a real merged tree and runs the actual stage4 attachment step, so this
        rendering-only test exercises render_gpu_kernel_fallback() against real (unattached,
        gpu_kernel_by_rank) data, the same fixture-based approach the rest of this file uses --
        not a hand-mocked stand-in."""
        cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
        ranks = s4t.load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
        rank_keys = [rk for rk, _rows, _roots in ranks]
        merged_roots = s4c.merge_rank_trees(ranks)
        flat = s4c.flatten_tree(merged_roots)
        node_values = s4c.make_node_values(rank_keys)
        gpu_per_rank = s4t.pair_gpu_per_rank(gpu_dir, run_dir, rank_keys)
        unattached, gpu_kernel_by_rank = s4t.attach_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, NEVER_PRUNED)
        return unattached, gpu_kernel_by_rank, node_values

    def test_returns_empty_string_when_nothing_unattached(self):
        self.assertEqual(tr.render_gpu_kernel_fallback(set(), {}, DEFAULT_NODE_VALUES), "")

    def test_matched_kernel_leaves_nothing_to_render(self):
        unattached, gpu_kernel_by_rank, node_values = self._attach(KERNEL_ANCHOR_DIR)
        self.assertEqual(tr.render_gpu_kernel_fallback(unattached, gpu_kernel_by_rank, node_values), "")

    def test_unmatched_kernel_produces_fallback_text(self):
        unattached, gpu_kernel_by_rank, node_values = self._attach(KERNEL_NO_ANCHOR_DIR)
        fallback = tr.render_gpu_kernel_fallback(unattached, gpu_kernel_by_rank, node_values)
        self.assertIn("=== GPU kernels (rocprofv3)", fallback)
        self.assertIn("JacobiIterationKernel", fallback)
        # ends in exactly the table's own trailing newline -- no self-appended blank line;
        # any blank line between sections is the caller's (render_report()'s) job now.
        self.assertFalse(fallback.endswith("\n\n"))
        self.assertTrue(fallback.endswith("\n"))


class RenderCalltreeTextTests(unittest.TestCase):
    def test_renders_tree_ending_in_single_newline(self):
        main = make_merged_tree_node("main", count=1, self_sum=0.0, total_sum=10.0)
        child = make_merged_tree_node("child", parent=main, count=1, self_sum=4.0, total_sum=4.0)
        rows = [main, child]
        text = tr.render_calltree_text([main], rows, None, NEVER_PRUNED, DEFAULT_NODE_VALUES)
        self.assertIn("main", text)
        self.assertIn("child", text)
        # ends in exactly the table's own trailing newline -- no self-appended blank line;
        # any blank line between sections is the caller's (render_report()'s) job now.
        self.assertFalse(text.endswith("\n\n"))
        self.assertTrue(text.endswith("\n"))

    def test_collapses_children_hides_grandchildren(self):
        main = make_merged_tree_node("main")
        mpi_call = make_merged_tree_node("MPI_Allreduce", parent=main)
        internal = make_merged_tree_node("MPIR_Allreduce_cdesc", parent=mpi_call)
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
