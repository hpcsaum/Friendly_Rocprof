import importlib.util
import os
import sys
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location(
    "stage5_table_render", os.path.join(POSTPROCESS_DIR, "stage5_table_render.py")
)
tr = importlib.util.module_from_spec(spec)
sys.modules["stage5_table_render"] = tr
spec.loader.exec_module(tr)


class SelectEntriesTests(unittest.TestCase):
    ENTRIES = [
        {"label": "a", "count": 1, "sum": 9.0, "self_sum": 1.0, "pct_total": 10.0},
        {"label": "b", "count": 1, "sum": 5.0, "self_sum": 5.0, "pct_total": 50.0},
        {"label": "c", "count": 1, "sum": 3.0, "self_sum": 3.0, "pct_total": 30.0},
    ]

    def test_ranks_descending_by_rank_field(self):
        selected, desc = tr.select_entries(self.ENTRIES, rank_field="self_sum")
        self.assertEqual([e["label"] for e in selected], ["b", "c", "a"])
        self.assertIn("top 20", desc)

    def test_ranking_by_a_different_field_changes_order(self):
        selected, _desc = tr.select_entries(self.ENTRIES, rank_field="sum")
        self.assertEqual([e["label"] for e in selected], ["a", "b", "c"])

    def test_top_n_truncates(self):
        selected, desc = tr.select_entries(self.ENTRIES, rank_field="self_sum", top=2)
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn("top 2 of 3", desc)

    def test_rank_label_appended_to_top_description(self):
        _selected, desc = tr.select_entries(self.ENTRIES, rank_field="self_sum", top=2, rank_label="std_dev")
        self.assertIn("top 2 of 3 entries by std_dev", desc)

    def test_no_rank_label_means_no_suffix(self):
        _selected, desc = tr.select_entries(self.ENTRIES, rank_field="self_sum", top=2)
        self.assertEqual(desc, "top 2 of 3 entries")

    def test_threshold_filters_by_threshold_field(self):
        selected, desc = tr.select_entries(
            self.ENTRIES, rank_field="self_sum", threshold_field="pct_total", threshold=30.0,
            threshold_unit="of total runtime",
        )
        self.assertEqual([e["label"] for e in selected], ["b", "c"])
        self.assertIn(">= 30% of total runtime (2 of 3 entries)", desc)

    def test_threshold_with_all_none_threshold_field_falls_back_to_all(self):
        entries = [dict(e, pct_total=None) for e in self.ENTRIES]
        selected, desc = tr.select_entries(
            entries, rank_field="self_sum", threshold_field="pct_total", threshold=30.0,
            threshold_unit="of total runtime",
        )
        self.assertEqual(len(selected), 3)
        self.assertIn("total runtime unknown, threshold ignored", desc)

    def test_threshold_unit_without_of_prefix_used_verbatim_in_unknown_message(self):
        entries = [dict(e, pct_total=None) for e in self.ENTRIES]
        _selected, desc = tr.select_entries(
            entries, rank_field="self_sum", threshold_field="pct_total", threshold=10.0,
            threshold_unit="coefficient of variation",
        )
        self.assertIn("coefficient of variation unknown, threshold ignored", desc)

    def test_show_all(self):
        selected, desc = tr.select_entries(self.ENTRIES, rank_field="self_sum", show_all=True)
        self.assertEqual(len(selected), 3)
        self.assertIn("all 3 entries", desc)

    def test_prepare_runs_before_ranking(self):
        entries = [{"label": "a", "raw": 1.0}, {"label": "b", "raw": 5.0}]

        def _double(es):
            for e in es:
                e["raw"] *= 2

        selected, _desc = tr.select_entries(entries, rank_field="raw", prepare=_double)
        self.assertEqual(entries[0]["raw"], 2.0)  # mutated in place
        self.assertEqual([e["label"] for e in selected], ["b", "a"])

    def test_tie_break_is_deterministic_ascending_by_label(self):
        entries = [
            {"label": "zeta", "std_dev": 5.0},
            {"label": "alpha", "std_dev": 5.0},
            {"label": "mid", "std_dev": 5.0},
        ]
        selected, _desc = tr.select_entries(entries, rank_field="std_dev")
        self.assertEqual([e["label"] for e in selected], ["alpha", "mid", "zeta"])

    def test_tie_break_field_can_be_overridden(self):
        entries = [
            {"label": "x", "std_dev": 5.0, "secondary": "z"},
            {"label": "y", "std_dev": 5.0, "secondary": "a"},
        ]
        selected, _desc = tr.select_entries(entries, rank_field="std_dev", tie_break_field="secondary")
        self.assertEqual([e["label"] for e in selected], ["y", "x"])


class RenderTableTests(unittest.TestCase):
    def test_empty_entries_returns_none_found(self):
        self.assertEqual(tr.render_table([{"header": "x", "width": 3, "value": lambda e, i: "x"}], []), "  (none found)\n")

    def test_right_aligned_column_padding(self):
        columns = [{"header": "#", "width": 3, "value": lambda e, i: str(i)}]
        table = tr.render_table(columns, [{"label": "a"}])
        self.assertIn("  #", table)
        self.assertIn("  1", table)

    def test_left_aligned_column(self):
        columns = [{"header": "run", "width": 6, "align": "left", "value": lambda e, i: "r0"}]
        table = tr.render_table(columns, [{"label": "a"}])
        lines = table.splitlines()
        self.assertTrue(lines[1].startswith("  r0"))

    def test_width_none_column_is_unpadded_trailing(self):
        columns = [
            {"header": "#", "width": 3, "value": lambda e, i: str(i)},
            {"header": "name", "width": None, "value": lambda e, i: e["label"]},
        ]
        table = tr.render_table(columns, [{"label": "a_very_long_function_name"}])
        self.assertIn("a_very_long_function_name", table)
        self.assertTrue(table.rstrip("\n").endswith("a_very_long_function_name"))

    def test_one_row_per_entry_one_indexed(self):
        columns = [{"header": "#", "width": 3, "value": lambda e, i: str(i)}]
        table = tr.render_table(columns, [{"label": "a"}, {"label": "b"}, {"label": "c"}])
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)  # header + 3 rows
        self.assertTrue(lines[1].strip().endswith("1"))
        self.assertTrue(lines[3].strip().endswith("3"))

    def test_long_trailing_label_comes_back_hard_wrapped(self):
        # End-to-end: render_table() actually calls wrap_trailing_label() (not just re-implements
        # the same idea inline) -- confirmed by parsing the rendered table back with
        # iter_table_rows(), the same reader the real select_hotspot_*.py consumers use.
        columns = [
            {"header": "#", "width": 3, "value": lambda e, i: str(i)},
            {"header": "function", "width": None, "value": lambda e, i: e["label"]},
        ]
        long_name = "MyNamespace::" + "A" * 150 + "::compute(int, double) const"
        table = tr.render_table(columns, [{"label": long_name}])
        self.assertGreater(len(table.splitlines()), 2)  # header + 2+ physical lines for one row
        rows = list(tr.iter_table_rows(table.splitlines()[1:], num_columns=2))
        self.assertEqual(rows, [["1", long_name]])


class WrapTrailingLabelTests(unittest.TestCase):
    def test_label_that_fits_is_unchanged(self):
        self.assertEqual(tr.wrap_trailing_label("  1  ", "short_name"), "  1  short_name")

    def test_long_label_wraps_into_expected_chunk_count(self):
        prefix = "  " * 5  # 10-char prefix -> available = 110
        label = "x" * 250
        result = tr.wrap_trailing_label(prefix, label, width=120)
        lines = result.split("\n")
        self.assertEqual(len(lines), 3)  # ceil(250 / 110) == 3

    def test_continuation_lines_indented_to_exact_prefix_width(self):
        prefix = "  1  12.340000  "
        label = "y" * 200
        result = tr.wrap_trailing_label(prefix, label, width=100)
        lines = result.split("\n")
        self.assertTrue(lines[0].startswith(prefix))
        for line in lines[1:]:
            self.assertTrue(line.startswith(" " * len(prefix)))
            self.assertFalse(line.startswith(" " * (len(prefix) + 1)))  # not over-indented

    def test_wrapped_chunks_reconstruct_the_original_label_byte_for_byte(self):
        prefix = "  42  9.999999  "
        label = "MyNamespace::" + "".join(f"Arg{i}, " for i in range(30)) + "Tail() const"
        result = tr.wrap_trailing_label(prefix, label, width=90)
        lines = result.split("\n")
        indent = " " * len(prefix)
        reconstructed = lines[0][len(prefix):] + "".join(line[len(indent):] for line in lines[1:])
        self.assertEqual(reconstructed, label)

    def test_pathological_prefix_falls_back_to_20_column_floor(self):
        prefix = " " * 115  # already exceeds width on its own
        label = "abcdefghijklmnopqrstuvwxyz"
        result = tr.wrap_trailing_label(prefix, label, width=120)
        lines = result.split("\n")
        self.assertGreater(len(lines), 1)  # still wraps instead of raising/looping forever


class IterTableRowsTests(unittest.TestCase):
    def test_unwrapped_table_round_trips_row_for_row(self):
        lines = [
            "    1     1.000000     10.0  a_short_name\n",
            "    2     2.000000     20.0  another_name\n",
        ]
        rows = list(tr.iter_table_rows(lines, num_columns=4))
        self.assertEqual(rows, [
            ["1", "1.000000", "10.0", "a_short_name"],
            ["2", "2.000000", "20.0", "another_name"],
        ])

    def test_wrapped_label_reconstructs_to_the_original_full_string(self):
        # No interior spaces in this label, by design: iter_table_rows()'s reconstruction
        # (like the addendum it's based on) strips each continuation line before appending it,
        # so a wrap point landing exactly on a space is a known, accepted lossy edge case -- not
        # what this test is checking. Long C++ namespace/template chains (the motivating real
        # case) are exactly this shape: one long contiguous identifier, no interior whitespace.
        prefix = "    1     1.000000     10.0  "
        label = "MyNamespace::" + "B" * 100 + "::Tail"
        wrapped = tr.wrap_trailing_label(prefix, label, width=90)
        rows = list(tr.iter_table_rows(wrapped.split("\n"), num_columns=4))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], label)

    def test_two_wrapped_rows_in_sequence_both_reconstruct(self):
        label_a = "Alpha::" + "X" * 100
        label_b = "Beta::" + "Y" * 100
        block = (
            tr.wrap_trailing_label("    1     1.000000     10.0  ", label_a, width=90) + "\n"
            + tr.wrap_trailing_label("    2     2.000000     20.0  ", label_b, width=90)
        )
        rows = list(tr.iter_table_rows(block.split("\n"), num_columns=4))
        self.assertEqual([r[3] for r in rows], [label_a, label_b])

    def test_stops_at_first_blank_line(self):
        lines = ["    1     1.000000     10.0  a\n", "\n", "    2     2.000000     20.0  b\n"]
        rows = list(tr.iter_table_rows(lines, num_columns=4))
        self.assertEqual(len(rows), 1)

    def test_malformed_short_line_left_for_caller_to_filter(self):
        lines = ["    1  too_few_tokens\n"]
        rows = list(tr.iter_table_rows(lines, num_columns=4))
        self.assertEqual(len(rows), 1)
        self.assertLess(len(rows[0]), 4)


class PctTotalNoteTests(unittest.TestCase):
    def test_reuses_the_given_threshold_unit_verbatim(self):
        note = tr.pct_total_note("function", "of total measured time (summed across all scanned files)")
        self.assertEqual(
            note,
            "  - '%total' is each function's share of total measured time (summed across all scanned files).\n",
        )

    def test_different_entry_noun(self):
        note = tr.pct_total_note("kernel", "of the combined pool above")
        self.assertIn("each kernel's share", note)


class RankingNoteTests(unittest.TestCase):
    def test_self_time_wording(self):
        note = tr.ranking_note(unfiltered=False)
        self.assertIn("Ranked by self time", note)
        self.assertTrue(note.startswith("  - "))

    def test_inclusive_time_wording(self):
        note = tr.ranking_note(unfiltered=True)
        self.assertIn("Ranked by inclusive (total) time", note)

    def test_extra_clause_appended(self):
        note = tr.ranking_note(unfiltered=False, extra_clause=" Extra sentence.")
        self.assertTrue(note.endswith("Extra sentence.\n"))


if __name__ == "__main__":
    unittest.main()
