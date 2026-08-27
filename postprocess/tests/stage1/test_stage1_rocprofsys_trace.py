import contextlib
import io
import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

stage1 = load_module_by_path("stage1_rocprofsys_trace", "stage1", "stage1_rocprofsys_trace.py")

SINGLE_RANK_CSV = os.path.join(FIXTURES, "trace_single_rank", "rank0.csv")
PARTITIONED_OTHER = os.path.join(FIXTURES, "trace_partitioned_rank", "rank0-other.csv")
PARTITIONED_MPI = os.path.join(FIXTURES, "trace_partitioned_rank", "rank0-mpi.csv")
PARTITIONED_GPU = os.path.join(FIXTURES, "trace_partitioned_rank", "rank0-gpu.csv")
PARTIAL_GPU_ONLY = os.path.join(FIXTURES, "trace_partial_gpu_only", "rank0-gpu.csv")


def by_label(rows, name):
    return next(r for r in rows if r["name"] == name)


class ParseTraceCsvTests(unittest.TestCase):
    def test_every_csv_column_present_on_every_row(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        expected_keys = {
            "pid", "process_name", "tid", "thread_name", "slice_id", "parent_slice_id", "depth",
            "name", "category", "ts", "dur", "corr_id", "blockDimX", "gridDimX", "grid_size",
            "workgroup_size",
        }
        for row in rows:
            self.assertEqual(set(row.keys()), expected_keys)

    def test_gpu_row_keeps_its_wide_columns(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        launch = by_label(rows, "hipLaunchKernel")
        self.assertEqual(launch["corr_id"], "42")
        self.assertEqual(launch["blockDimX"], "64")
        self.assertEqual(launch["gridDimX"], "256")

    def test_non_gpu_row_has_wide_columns_as_none_not_dropped(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        main = by_label(rows, "main")
        self.assertIn("corr_id", main)
        self.assertIsNone(main["corr_id"])
        self.assertIsNone(main["grid_size"])

    def test_kernel_dispatch_row_keeps_comma_bearing_wide_values(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertEqual(kernel["grid_size"], "(256,1,1)")
        self.assertEqual(kernel["workgroup_size"], "(64,1,1)")

    def test_slice_id_and_parent_slice_id_are_int_or_none(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        main = by_label(rows, "main")
        sweep = by_label(rows, "jacobi_sweep")
        self.assertIsNone(main["parent_slice_id"])
        self.assertEqual(main["slice_id"], 1)
        self.assertIsInstance(sweep["parent_slice_id"], int)
        self.assertEqual(sweep["parent_slice_id"], 1)

    def test_ts_and_dur_converted_from_nanoseconds_to_seconds(self):
        rows = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        main = by_label(rows, "main")
        self.assertAlmostEqual(main["dur"], 1.0)
        self.assertAlmostEqual(main["ts"], 1.0)
        sweep = by_label(rows, "jacobi_sweep")
        self.assertAlmostEqual(sweep["dur"], 0.6)

    def test_single_path_and_list_of_one_path_are_equivalent(self):
        via_str = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        via_list = stage1.parse_trace_csv([SINGLE_RANK_CSV])
        self.assertEqual(via_str, via_list)

    def test_partitioned_files_concatenate_to_the_same_row_set(self):
        combined = stage1.parse_trace_csv([PARTITIONED_OTHER, PARTITIONED_MPI, PARTITIONED_GPU])
        single = stage1.parse_trace_csv(SINGLE_RANK_CSV)
        combined_by_id = {r["slice_id"]: r for r in combined}
        single_by_id = {r["slice_id"]: r for r in single}
        self.assertEqual(set(combined_by_id.keys()), set(single_by_id.keys()))
        for slice_id, row in single_by_id.items():
            self.assertEqual(combined_by_id[slice_id], row)


class AttachAncestryTests(unittest.TestCase):
    def test_multi_level_parent_chain_resolves_to_real_objects(self):
        rows = stage1.attach_ancestry(stage1.parse_trace_csv(SINGLE_RANK_CSV))
        main = by_label(rows, "main")
        sweep = by_label(rows, "jacobi_sweep")
        launch = by_label(rows, "hipLaunchKernel")
        barrier = by_label(rows, "MPI_Barrier")

        self.assertIsNone(main["parent"])
        self.assertIs(sweep["parent"], main)
        self.assertIs(launch["parent"], sweep)
        self.assertIs(barrier["parent"], sweep)
        # attach_ancestry() resolves parent_slice_id into a real object reference (above) without
        # removing the raw id it was resolved from.
        self.assertEqual(sweep["parent_slice_id"], 1)

    def test_untethered_gpu_root_gets_none_parent_with_no_warning(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = stage1.attach_ancestry(stage1.parse_trace_csv(SINGLE_RANK_CSV))
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(kernel["parent"])
        self.assertEqual(buf.getvalue(), "")

    def test_partitioned_set_resolves_identically_to_single_file(self):
        combined = stage1.attach_ancestry(
            stage1.parse_trace_csv([PARTITIONED_OTHER, PARTITIONED_MPI, PARTITIONED_GPU])
        )
        launch = by_label(combined, "hipLaunchKernel")
        sweep = by_label(combined, "jacobi_sweep")
        self.assertIs(launch["parent"], sweep)
        self.assertIsNone(by_label(combined, "jacobi_kernel.kd")["parent"])

    def test_partial_file_alone_warns_once_and_promotes_orphan_to_root(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = stage1.attach_ancestry(stage1.parse_trace_csv(PARTIAL_GPU_ONLY))

        launch = by_label(rows, "hipLaunchKernel")
        kernel = by_label(rows, "jacobi_kernel.kd")
        self.assertIsNone(launch["parent"])  # real parent (jacobi_sweep) wasn't in this file
        self.assertIsNone(kernel["parent"])  # legitimately parentless, not an orphan

        warnings = [line for line in buf.getvalue().splitlines() if line.startswith("warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 row(s)", warnings[0])


class LabelKeyTests(unittest.TestCase):
    def test_label_key_names_the_name_column(self):
        # Not a self-check: stage3_rocprofsys_trace.py and stage4_rocprofsys_trace_aggregate.py
        # both import this constant directly and pass it as merge_rank_trees()/tag_rows()'s own
        # label_key= parameter -- a silent change here would break which dict key those functions
        # treat as a row's display name, with no local error to catch it.
        self.assertEqual(stage1.LABEL_KEY, "name")


if __name__ == "__main__":
    unittest.main()
