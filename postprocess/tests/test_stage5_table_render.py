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


if __name__ == "__main__":
    unittest.main()
