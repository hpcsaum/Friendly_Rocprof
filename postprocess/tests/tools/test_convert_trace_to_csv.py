"""Tests for convert_trace_to_csv.py, the tool that turns a rocprof-sys trace-mode run's per-rank
Perfetto `.proto` files into the flat trace-CSV files this project's trace-based tools consume.

ParseTraceProcessorCsvTests -- trace_processor_shell's own quoted-CSV format, "[NULL]" -> ""
PivotArgsBySliceIdTests     -- long-format args rows pivoted into one dict per slice_id
WriteUnfilteredCsvTests     -- the plain 11-column CSV round-trips through stage1's own parser
WritePartitionedCsvsTests   -- gpu/mpi/other bucketing by category tag, args columns gpu-only
DiscoverProtoRanksTests     -- per-rank *.proto discovery, numeric sort, merged-file exclusion
ResolveTraceProcessorTests  -- trace_processor_shell resolution precedence: CLI arg > env var > PATH
RunTraceProcessorTests      -- non-zero subprocess exit surfaces as SystemExit with stderr
ConvertRankTests            -- one rank's conversion, with/without the --unfiltered extra file
MainCliTests                -- end-to-end smoke test plus missing-directory error handling
"""

import argparse
import csv
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

conv = load_module_by_path("convert_trace_to_csv", "tools", "convert_trace_to_csv.py")

from stage1_rocprofsys_trace import parse_trace_csv
from stage4_rocprofsys_trace_ranks import _RANK_FILE_RE


def _tp_csv(header, rows):
    """Builds a trace_processor_shell-shaped CSV block: quoted header/strings, unquoted numbers,
    literal "[NULL]" for a SQL NULL -- exactly what the real tool's own C++ CSV writer emits (see
    convert_trace_to_csv.py's own module docstring for the source confirmation), never what this
    project's other fixtures look like."""
    buf = io.StringIO()
    buf.write(",".join(f'"{h}"' for h in header) + "\n")
    for row in rows:
        buf.write(",".join(row) + "\n")
    return buf.getvalue()


# One thread-track row (main, tid=1000) and one process-track-only row (a GPU kernel dispatch,
# tid/thread_name genuinely NULL) -- exercises the COALESCE join and the "[NULL]" fixup together.
BASE_ROWS_CSV = _tp_csv(
    ["pid", "process_name", "tid", "thread_name", "slice_id", "parent_slice_id", "depth", "name",
     "category", "ts", "dur"],
    [
        ['1000', '"./app"', '1000', '"main"', '1', '"[NULL]"', '0', '"main"', '"host"', '0', '1000000000'],
        ['1000', '"./app"', '1000', '"main"', '2', '1', '1', '"MPI_Barrier"', '"mpi"', '10000000', '5000000'],
        ['1000', '"./app"', '"[NULL]"', '"[NULL]"', '3', '"[NULL]"', '0', '"jacobi_kernel.kd"',
         '"rocm_kernel_dispatch"', '20000000', '4000000'],
    ],
)

ARGS_ROWS_CSV = _tp_csv(
    ["slice_id", "flat_key", "int_value", "string_value", "real_value"],
    [
        ['3', '"corr_id"', '42', '"[NULL]"', '"[NULL]"'],
        ['3', '"blockDimX"', '64', '"[NULL]"', '"[NULL]"'],
    ],
)


class ParseTraceProcessorCsvTests(unittest.TestCase):
    def test_null_marker_becomes_empty_string(self):
        rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        self.assertEqual(rows[0]["parent_slice_id"], "")
        self.assertEqual(rows[2]["tid"], "")
        self.assertEqual(rows[2]["thread_name"], "")

    def test_ordinary_values_pass_through(self):
        rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        self.assertEqual(rows[0]["name"], "main")
        self.assertEqual(rows[0]["ts"], "0")
        self.assertEqual(rows[1]["category"], "mpi")


class PivotArgsBySliceIdTests(unittest.TestCase):
    def test_pivots_long_format_into_one_dict_per_slice(self):
        args_rows = conv._parse_trace_processor_csv(ARGS_ROWS_CSV)
        by_slice = conv._pivot_args_by_slice_id(args_rows)
        self.assertEqual(by_slice, {"3": {"corr_id": "42", "blockDimX": "64"}})

    def test_duplicate_key_keeps_first_value_and_warns_once(self):
        args_rows = conv._parse_trace_processor_csv(_tp_csv(
            ["slice_id", "flat_key", "int_value", "string_value", "real_value"],
            [
                ['1', '"corr_id"', '1', '"[NULL]"', '"[NULL]"'],
                ['1', '"corr_id"', '2', '"[NULL]"', '"[NULL]"'],
            ],
        ))
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            by_slice = conv._pivot_args_by_slice_id(args_rows)
        self.assertEqual(by_slice, {"1": {"corr_id": "1"}})
        self.assertIn("warning:", buf.getvalue())

    def test_real_value_used_when_int_and_string_are_null(self):
        args_rows = conv._parse_trace_processor_csv(_tp_csv(
            ["slice_id", "flat_key", "int_value", "string_value", "real_value"],
            [['1', '"ratio"', '"[NULL]"', '"[NULL]"', '0.500000']],
        ))
        by_slice = conv._pivot_args_by_slice_id(args_rows)
        self.assertEqual(by_slice["1"]["ratio"], "0.500000")


class WriteUnfilteredCsvTests(unittest.TestCase):
    def test_writes_base_11_columns_only_and_round_trips_through_stage1(self):
        base_rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "perfetto-trace-0.csv")
            conv.write_unfiltered_csv(base_rows, dest)

            with open(dest) as f:
                header = next(csv.reader(f))
            self.assertEqual(header, conv._BASE_COLUMNS)

            parsed = parse_trace_csv(dest)
            self.assertEqual(len(parsed), 3)
            self.assertEqual(parsed[0]["name"], "main")
            self.assertIsNone(parsed[0]["parent_slice_id"])  # "[NULL]" -> "" -> None
            self.assertAlmostEqual(parsed[0]["ts"], 0.0)
            self.assertAlmostEqual(parsed[0]["dur"], 1.0)
            self.assertIsNone(parsed[2]["tid"])


class WritePartitionedCsvsTests(unittest.TestCase):
    def test_buckets_match_tag_for_category_and_sum_to_every_row(self):
        base_rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        args_by_slice_id = conv._pivot_args_by_slice_id(conv._parse_trace_processor_csv(ARGS_ROWS_CSV))
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "perfetto-trace-0")
            conv.write_partitioned_csvs(base_rows, args_by_slice_id, stem)

            total_rows = 0
            for bin_name in ("gpu", "mpi", "other"):
                path = f"{stem}-{bin_name}.csv"
                self.assertTrue(os.path.isfile(path))
                with open(path) as f:
                    total_rows += sum(1 for _ in csv.DictReader(f))
            self.assertEqual(total_rows, len(base_rows))

    def test_gpu_file_gets_args_columns_others_do_not(self):
        base_rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        args_by_slice_id = conv._pivot_args_by_slice_id(conv._parse_trace_processor_csv(ARGS_ROWS_CSV))
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "perfetto-trace-0")
            conv.write_partitioned_csvs(base_rows, args_by_slice_id, stem)

            with open(f"{stem}-gpu.csv") as f:
                gpu_header = next(csv.reader(f))
            self.assertEqual(gpu_header, conv._BASE_COLUMNS + ["blockDimX", "corr_id"])  # alphabetical

            with open(f"{stem}-mpi.csv") as f:
                mpi_header = next(csv.reader(f))
            self.assertEqual(mpi_header, conv._BASE_COLUMNS)

            with open(f"{stem}-other.csv") as f:
                other_header = next(csv.reader(f))
            self.assertEqual(other_header, conv._BASE_COLUMNS)

    def test_output_filenames_match_discover_ranks_own_convention(self):
        base_rows = conv._parse_trace_processor_csv(BASE_ROWS_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "perfetto-trace-0")
            conv.write_partitioned_csvs(base_rows, {}, stem)
            for bin_name in ("gpu", "mpi", "other"):
                path = f"{stem}-{bin_name}.csv"
                m = _RANK_FILE_RE.search(os.path.basename(path))
                self.assertIsNotNone(m)
                self.assertEqual(m.group(1), "0")
                self.assertEqual(m.group(2), bin_name)


class DiscoverProtoRanksTests(unittest.TestCase):
    def test_finds_per_rank_files_sorted_numerically_and_skips_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["perfetto-trace-10.proto", "perfetto-trace-2.proto", "merged.proto"]:
                open(os.path.join(tmp, name), "w").close()
            ranks = conv.discover_proto_ranks(tmp)
            self.assertEqual([r for r, _ in ranks], ["2", "10"])

    def test_no_matches_raises_system_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                conv.discover_proto_ranks(tmp)


class ResolveTraceProcessorTests(unittest.TestCase):
    def _args(self, trace_processor=None):
        return argparse.Namespace(trace_processor=trace_processor)

    def test_cli_arg_wins(self):
        self.assertEqual(conv.resolve_trace_processor(self._args("/opt/tp")), "/opt/tp")

    def test_env_var_used_when_no_cli_arg(self):
        with mock.patch.dict(os.environ, {"FRIENDLY_ROCPROF_TRACE_PROCESSOR": "/env/tp"}, clear=False):
            self.assertEqual(conv.resolve_trace_processor(self._args()), "/env/tp")

    def test_path_lookup_when_neither_given(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FRIENDLY_ROCPROF_TRACE_PROCESSOR", None)
            with mock.patch.object(conv.shutil, "which", side_effect=lambda name: "/usr/bin/trace_processor_shell" if name == "trace_processor_shell" else None):
                self.assertEqual(conv.resolve_trace_processor(self._args()), "/usr/bin/trace_processor_shell")

    def test_raises_clear_error_when_nothing_resolves(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FRIENDLY_ROCPROF_TRACE_PROCESSOR", None)
            with mock.patch.object(conv.shutil, "which", return_value=None):
                with self.assertRaises(SystemExit):
                    conv.resolve_trace_processor(self._args())


class RunTraceProcessorTests(unittest.TestCase):
    def test_nonzero_exit_raises_system_exit_with_stderr(self):
        with mock.patch.object(conv.subprocess, "run") as mocked_run:
            mocked_run.return_value = mock.Mock(returncode=1, stdout="", stderr="boom")
            with self.assertRaises(SystemExit) as ctx:
                conv._run_trace_processor("/bin/trace_processor_shell", "trace.proto", "SELECT 1")
            self.assertIn("boom", str(ctx.exception))


class ConvertRankTests(unittest.TestCase):
    def test_default_writes_only_the_partitioned_trio(self):
        # side_effect order matches convert_rank()'s own two _run_trace_processor calls: the base
        # slices query first, the args query second.
        with mock.patch.object(conv, "_run_trace_processor", side_effect=[BASE_ROWS_CSV, ARGS_ROWS_CSV]):
            with tempfile.TemporaryDirectory() as tmp:
                written = conv.convert_rank("/anywhere/perfetto-trace-0.proto", tmp, "/bin/tp", False)
                self.assertEqual(len(written), 3)
                for suffix in ("-gpu.csv", "-mpi.csv", "-other.csv"):
                    self.assertTrue(any(p.endswith(suffix) for p in written))
                self.assertFalse(os.path.exists(os.path.join(tmp, "perfetto-trace-0.csv")))

    def test_unfiltered_flag_adds_the_plain_file(self):
        with mock.patch.object(conv, "_run_trace_processor", side_effect=[BASE_ROWS_CSV, ARGS_ROWS_CSV]):
            with tempfile.TemporaryDirectory() as tmp:
                written = conv.convert_rank("/anywhere/perfetto-trace-0.proto", tmp, "/bin/tp", True)
                self.assertEqual(len(written), 4)
                self.assertTrue(os.path.exists(os.path.join(tmp, "perfetto-trace-0.csv")))


class MainCliTests(unittest.TestCase):
    def test_end_to_end_smoke_test_with_placeholder_proto_files(self):
        with tempfile.TemporaryDirectory() as proto_dir:
            open(os.path.join(proto_dir, "perfetto-trace-0.proto"), "w").close()
            with mock.patch.object(conv, "_run_trace_processor", side_effect=[BASE_ROWS_CSV, ARGS_ROWS_CSV]):
                conv.main([proto_dir, "--trace-processor", "/bin/trace_processor_shell"])
            self.assertTrue(os.path.exists(os.path.join(proto_dir, "perfetto-trace-0-gpu.csv")))
            self.assertTrue(os.path.exists(os.path.join(proto_dir, "perfetto-trace-0-mpi.csv")))
            self.assertTrue(os.path.exists(os.path.join(proto_dir, "perfetto-trace-0-other.csv")))

    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            conv.main(["/no/such/directory", "--trace-processor", "/bin/trace_processor_shell"])


if __name__ == "__main__":
    unittest.main()
