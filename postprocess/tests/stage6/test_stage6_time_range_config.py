"""Tests for stage6_time_range_config.py's --time-range parsing, singleton, and report-note formatting.

ParseTimeRangeTests                -- parse_time_range()'s segment parsing, bound merging, and malformed-input errors
ConfigureAndActiveRangesTests      -- configure()/active_ranges()'s process-wide singleton lifecycle
FormatNoteTests                    -- _format_note()'s full-run vs active-range report-line formatting
DescribeTimeRangeTests             -- describe_time_range()'s active-range and multi-rank-extent-combining behavior
DescribeTimeRangeStage4WiringTests -- confirms describe_time_range() wires through to the real stage4 extent
                                       accessor, not a reimplementation
"""

import os
import shutil
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)

# Plain shared imports, not an isolated spec_from_file_location copy: unlike stage6_noise_config,
# nothing anywhere imports a bare function reference out of this module ("from
# stage6_time_range_config import X") -- every consumer (stage4_rocprofsys_trace_aggregate.py,
# stage5_trace_calltree_view.py, the 3 CLI tools) does a plain "import stage6_time_range_config"
# and resolves attributes at call time, so there's only ever one real module instance for the
# whole test process and no captured-reference hazard to guard against.
import stage4_rocprofsys_trace_aggregate as agg  # noqa: E402
import stage6_time_range_config as trc  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
TIME_RANGE_CSV = os.path.join(FIXTURES, "trace_time_range", "rank0.csv")


class ParseTimeRangeTests(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(trc.parse_time_range(None))

    def test_single_bounded_segment(self):
        self.assertEqual(trc.parse_time_range("5:12.5"), [(5.0, 12.5)])

    def test_open_end_segment(self):
        self.assertEqual(trc.parse_time_range("5:"), [(5.0, None)])

    def test_open_start_segment(self):
        self.assertEqual(trc.parse_time_range(":10"), [(None, 10.0)])

    def test_whitespace_around_segments_and_bounds_is_tolerated(self):
        self.assertEqual(trc.parse_time_range(" 5 : 12.5 , 20 : "), [(5.0, 12.5), (20.0, None)])

    def test_touching_segments_merge(self):
        self.assertEqual(trc.parse_time_range("0:5,5:10"), [(0.0, 10.0)])

    def test_overlapping_segments_merge(self):
        self.assertEqual(trc.parse_time_range("0:6,5:10"), [(0.0, 10.0)])

    def test_disjoint_segments_stay_separate_and_get_sorted(self):
        self.assertEqual(trc.parse_time_range("10:15,0:5"), [(0.0, 5.0), (10.0, 15.0)])

    def test_unbounded_segment_absorbs_everything_after_it(self):
        self.assertEqual(trc.parse_time_range("5:,0:3"), [(0.0, 3.0), (5.0, None)])

    def test_open_ended_segment_absorbs_a_later_overlapping_bounded_segment(self):
        # Unlike the disjoint case above, "8:12" starts inside "5:"'s open-ended span, so it's
        # merged away rather than kept as its own entry -- the open end must stay open (None), not
        # get clobbered back down to the bounded segment's own end (12.0).
        self.assertEqual(trc.parse_time_range("5:,8:12"), [(5.0, None)])

    # (case, malformed input, expected error message substring) -- every case is the same
    # assertRaisesRegex(SystemExit, ...) shape, differing only in the malformed segment and which
    # of parse_time_range()'s validation messages it should trip.
    MALFORMED_CASES = [
        ("segment_without_a_colon", "10", "no ':'"),
        ("empty_string", "", "no ':'"),
        ("neither_bound_given", ":", "needs at least a start or an end"),
        ("start_equal_to_end", "5:5", "start must be less than end"),
        ("start_greater_than_end", "10:5", "start must be less than end"),
        ("non_numeric_start", "a:5", "isn't a number"),
        ("non_numeric_end", "5:b", "isn't a number"),
    ]

    def test_malformed_segment_raises(self):
        for name, segment, expected_message in self.MALFORMED_CASES:
            with self.subTest(case=name):
                with self.assertRaisesRegex(SystemExit, expected_message):
                    trc.parse_time_range(segment)


class ConfigureAndActiveRangesTests(unittest.TestCase):
    def tearDown(self):
        trc.configure(None)

    def test_no_config_leaves_active_ranges_none(self):
        trc.configure(None)
        self.assertIsNone(trc.active_ranges())

    def test_configure_stores_the_parsed_result(self):
        trc.configure("5:10")
        self.assertEqual(trc.active_ranges(), [(5.0, 10.0)])

    def test_configure_again_fully_replaces_prior_value(self):
        trc.configure("5:10")
        trc.configure("20:30")
        self.assertEqual(trc.active_ranges(), [(20.0, 30.0)])
        trc.configure(None)
        self.assertIsNone(trc.active_ranges())


class FormatNoteTests(unittest.TestCase):
    def test_no_ranges_formats_the_full_run_extent(self):
        note = trc._format_note((0.0, 45.321), None)
        self.assertEqual(note, "  time range: 0.000s-45.321s (full run)\n")

    def test_ranges_given_formats_each_window(self):
        note = trc._format_note((0.0, 45.321), [(5.0, 12.5), (20.0, None)])
        self.assertEqual(note, "  time range: 5.000s-12.500s, 20.000s-45.321s\n")

    def test_open_start_resolves_against_the_run_extent(self):
        note = trc._format_note((3.0, 45.321), [(None, 12.5)])
        self.assertEqual(note, "  time range: 3.000s-12.500s\n")

    def test_empty_ranges_list_is_treated_like_none(self):
        note = trc._format_note((0.0, 10.0), [])
        self.assertEqual(note, "  time range: 0.000s-10.000s (full run)\n")


class DescribeTimeRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "rank0.csv")
        shutil.copy(TIME_RANGE_CSV, self.csv_path)
        self.rank_inputs = [("r0", self.csv_path)]

    def tearDown(self):
        shutil.rmtree(self.tmp)
        trc.configure(None)

    def test_active_range_is_described_directly(self):
        trc.configure("30:70")
        note = trc.describe_time_range(self.rank_inputs)
        self.assertEqual(note, "  time range: 30.000s-70.000s\n")

    def test_combines_extents_across_multiple_ranks(self):
        csv_path_1 = os.path.join(self.tmp, "rank1.csv")
        shutil.copy(TIME_RANGE_CSV, csv_path_1)
        rank_inputs = [("r0", self.csv_path), ("r1", csv_path_1)]
        trc.configure(None)
        note = trc.describe_time_range(rank_inputs)
        self.assertEqual(note, "  time range: 0.000s-100.000s (full run)\n")


class DescribeTimeRangeStage4WiringTests(unittest.TestCase):
    # The one deliberate stage6 -> stage4 import exception in the codebase (describe_time_range()
    # calling stage4_rocprofsys_trace_aggregate.get_rank_time_extent() directly) -- this confirms
    # the wiring reaches real data, not that get_rank_time_extent() itself is correct (that's
    # test_stage4_rocprofsys_trace_aggregate.py::GetRankTimeExtentTests' job).
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "rank0.csv")
        shutil.copy(TIME_RANGE_CSV, self.csv_path)
        self.rank_inputs = [("r0", self.csv_path)]

    def tearDown(self):
        shutil.rmtree(self.tmp)
        trc.configure(None)

    def test_uses_the_real_stage4_extent_accessor(self):
        # Confirms the deliberate stage6 -> stage4 dependency actually wires through to real data,
        # not a hand-rolled reimplementation of the extent computation.
        expected = agg.get_rank_time_extent(self.csv_path, "r0", cache_dir=self.tmp)
        trc.configure(None)
        note = trc.describe_time_range(self.rank_inputs, cache_dir=self.tmp)
        expected_note = trc._format_note(expected, None)
        self.assertEqual(note, expected_note)


if __name__ == "__main__":
    unittest.main()
