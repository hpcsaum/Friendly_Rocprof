import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "stage5", "stage5_calltree_text_parser.py")

sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

spec = importlib.util.spec_from_file_location("stage5_calltree_text_parser", MODULE_PATH)
parser = importlib.util.module_from_spec(spec)
sys.modules["stage5_calltree_text_parser"] = parser
spec.loader.exec_module(parser)

from stage4_rocprofsys_common import caller_chains_for_label
from stage5_tree_render import REPORT_HEADERS, build_children_map, format_aligned_rows, render_forest


def make_row(label, parent=None):
    return {"label": label, "parent": parent}


def _dummy_values(_node):
    return (1.0, 0.1, 0.0, 0.1, 0.1, 1.0)


def render_block(roots, flat, max_depth=None):
    """Renders a tree block through the REAL stage5_tree_render pipeline (same functions
    stage5_calltree_view.py actually calls), so fixtures are byte-exact to production output
    instead of hand-typed guesses at column alignment."""
    children_map = build_children_map(flat)
    return format_aligned_rows(
        render_forest(roots, children_map, max_depth, lambda n: False, _dummy_values), REPORT_HEADERS,
    )


def make_report_text(roots, flat, max_depth=None):
    """Wraps a rendered block the same way a real report does: free-form header text, a blank
    line, the tree block (already self-terminated with a newline), a blank line, then notes --
    exactly the boundary flat_rows_from_calltree_text() is documented to rely on."""
    block = render_block(roots, flat, max_depth=max_depth)
    return (
        "Some report header text\ngenerated: 2026-01-01\n\n"
        + block
        + "\n  - Every row is aggregated across all ranks.\n  - Ranks aggregated: 0, 1\n"
    )


class FlatRowsFromCalltreeTextTests(unittest.TestCase):
    def test_parses_a_simple_tree_with_correct_parent_links(self):
        main = make_row("main")
        jacobi_sweep = make_row("jacobi_sweep", parent=main)
        jacobi_kernel = make_row("jacobi_kernel.kd", parent=jacobi_sweep)
        mpi_barrier = make_row("MPI_Barrier", parent=main)
        text = make_report_text([main], [main, jacobi_sweep, jacobi_kernel, mpi_barrier])

        flat = parser.flat_rows_from_calltree_text(text)
        by_label = {r["label"]: r for r in flat}
        self.assertEqual(set(by_label), {"main", "jacobi_sweep", "jacobi_kernel.kd", "MPI_Barrier"})
        self.assertIsNone(by_label["main"]["parent"])
        self.assertIs(by_label["jacobi_sweep"]["parent"], by_label["main"])
        self.assertIs(by_label["jacobi_kernel.kd"]["parent"], by_label["jacobi_sweep"])
        self.assertIs(by_label["MPI_Barrier"]["parent"], by_label["main"])

    def test_stops_at_the_first_blank_line_after_the_tree(self):
        main = make_row("main")
        text = make_report_text([main], [main])

        flat = parser.flat_rows_from_calltree_text(text)
        labels = {r["label"] for r in flat}
        self.assertNotIn("Every row is aggregated across all ranks.", labels)

    def test_caller_chains_for_label_works_unchanged_on_reconstructed_rows(self):
        main = make_row("main")
        jacobi_sweep = make_row("jacobi_sweep", parent=main)
        jacobi_kernel = make_row("jacobi_kernel.kd", parent=jacobi_sweep)
        text = make_report_text([main], [main, jacobi_sweep, jacobi_kernel])

        flat = parser.flat_rows_from_calltree_text(text)
        chains = caller_chains_for_label(flat, "jacobi_kernel.kd")
        self.assertEqual(len(chains), 1)
        self.assertEqual([n["label"] for n in chains[0]], ["main", "jacobi_sweep", "jacobi_kernel.kd"])

    def test_hard_wrapped_label_is_reconstructed_exactly(self):
        long_label = "convection_stable_dt$convection_time_integrator_mod_$ck_L558_99.kd"
        main = make_row("main")
        kernel = make_row(long_label, parent=main)
        text = make_report_text([main], [main, kernel])

        # Sanity: this fixture actually exercises wrapping (the whole point of the test) --
        # if the label ever got short enough not to wrap, the test would be vacuous.
        self.assertIn("\n", render_block([main], [main, kernel]).rstrip("\n").split("\n", 1)[1])

        flat = parser.flat_rows_from_calltree_text(text)
        labels = {r["label"] for r in flat}
        self.assertIn(long_label, labels)

    def test_multiple_roots_each_get_parent_none(self):
        main = make_row("main")
        foo = make_row("foo", parent=main)
        other_root = make_row("other_root")
        bar = make_row("bar", parent=other_root)
        text = make_report_text([main, other_root], [main, foo, other_root, bar])

        flat = parser.flat_rows_from_calltree_text(text)
        by_label = {r["label"]: r for r in flat}
        self.assertIsNone(by_label["main"]["parent"])
        self.assertIsNone(by_label["other_root"]["parent"])
        self.assertIs(by_label["foo"]["parent"], by_label["main"])
        self.assertIs(by_label["bar"]["parent"], by_label["other_root"])

    def test_hidden_nodes_marker_line_is_skipped_not_merged_as_a_continuation(self):
        main = make_row("main")
        foo = make_row("foo", parent=main)
        baz = make_row("baz", parent=foo)
        text = make_report_text([main], [main, foo, baz], max_depth=1)

        flat = parser.flat_rows_from_calltree_text(text)
        labels = [r["label"] for r in flat]
        self.assertEqual(labels, ["main", "foo"])

    def test_no_tree_header_found_returns_empty_list(self):
        self.assertEqual(parser.flat_rows_from_calltree_text("just some unrelated text\n"), [])

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(parser.flat_rows_from_calltree_text(""), [])


if __name__ == "__main__":
    unittest.main()
