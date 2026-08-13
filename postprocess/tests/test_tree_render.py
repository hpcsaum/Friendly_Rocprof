import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("tree_render", os.path.join(POSTPROCESS_DIR, "tree_render.py"))
tr = importlib.util.module_from_spec(spec)
sys.modules["tree_render"] = tr
spec.loader.exec_module(tr)

# render_forest() takes a node_values(node) callable -- in real usage this always
# comes from stage4_rocprofsys_tree.make_node_values(), so the fixture here uses the
# same real function rather than a stand-in, mirroring production code's own
# dependency between the two modules.
spec4 = importlib.util.spec_from_file_location("stage4_rocprofsys_tree", os.path.join(POSTPROCESS_DIR, "stage4_rocprofsys_tree.py"))
s4t = importlib.util.module_from_spec(spec4)
sys.modules["stage4_rocprofsys_tree"] = s4t
spec4.loader.exec_module(s4t)

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


if __name__ == "__main__":
    unittest.main()
