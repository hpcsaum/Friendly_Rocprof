#!/usr/bin/env python3
"""Converts a rocprof-sys trace-mode run's per-rank Perfetto `.proto` trace files into the flat
trace-CSV files this project's trace-based tools already consume (see
stage4_rocprofsys_trace_ranks.py for the exact naming convention this tool's own output matches).

Scope: one `trace_processor_shell` subprocess invocation per (rank, query) -- never the merged
whole-run trace a real rocprof-sys output directory also carries alongside its per-rank files, and
never more than one rank's data out of a single `.proto` file. Locating `trace_processor_shell`
itself is this module's only "external tool" concern -- producing the `.proto` files in the first
place (running rocprof-sys in trace mode) is entirely out of scope, same as this project's other
tools never run the profiler that produces their own input.

Functions: discover_proto_ranks(), add_cli_argument(), resolve_trace_processor(),
write_unfiltered_csv(), write_partitioned_csvs(), convert_rank(), main().
"""

import argparse
import csv
import glob
import io
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage3_rocprofsys_trace import tag_for_category
import stage6_cli_common

HELP_BLURB = """\
Converts a rocprof-sys trace-mode run's raw per-rank Perfetto `.proto`
trace files into the flat trace-CSV files this project's
extract_trace_*.py tools expect (see the project README) -- the one step
between running rocprof-sys in trace mode and using this project's own
trace-based post-processing tools.

Reads every per-rank `*.proto` file in the given directory (rocprof-sys's
own naming convention: `<prefix>-<N>.proto`) and ignores a `merged.proto`
file if one is present alongside them -- that's the whole-run merged
trace, out of scope here; each rank's own trace is converted
independently, with no cross-rank consistency checking (a missing or
mismatched rank across files is not this tool's concern).

By default, writes the -gpu/-mpi/-other partitioned trio per rank -- the
shape needed for exact GPU-kernel-to-launch-site correlation downstream.
Pass --unfiltered to additionally write the plain, combined file per rank
(no GPU-arg columns at all) -- cheaper to read, but a downstream tool
loses exact kernel placement if that's the only file present for a rank.

Under the hood, this shells out to Perfetto's trace_processor_shell to
query each `.proto` file (see --trace-processor's own help text for how
this tool is located) -- see
https://perfetto.dev/docs/analysis/trace-processor for details on
trace_processor_shell itself.
"""

_PER_RANK_PROTO_RE = re.compile(r"-(\d+)\.proto$")

_BASE_SLICES_SQL = (
    "SELECT "
    "process.pid AS pid, process.name AS process_name, "
    "thread.tid AS tid, thread.name AS thread_name, "
    "slice.id AS slice_id, slice.parent_id AS parent_slice_id, slice.depth AS depth, "
    "slice.name AS name, slice.category AS category, slice.ts AS ts, slice.dur AS dur "
    "FROM slice "
    "LEFT JOIN thread_track ON slice.track_id = thread_track.id "
    "LEFT JOIN thread ON thread_track.utid = thread.utid "
    "LEFT JOIN process_track ON slice.track_id = process_track.id "
    "LEFT JOIN process ON COALESCE(thread.upid, process_track.upid) = process.upid "
    "ORDER BY slice.id"
)

_ARGS_SQL = (
    "SELECT slice.id AS slice_id, args.flat_key AS flat_key, "
    "args.int_value AS int_value, args.string_value AS string_value, args.real_value AS real_value "
    "FROM slice JOIN args ON slice.arg_set_id = args.arg_set_id"
)

_BASE_COLUMNS = [
    "pid", "process_name", "tid", "thread_name", "slice_id", "parent_slice_id", "depth", "name",
    "category", "ts", "dur",
]

# tag_for_category()'s own return values, bucketed into which of the 3 partitioned files a row
# goes to -- "mpi_territory" -> "-mpi", every gpu_*-prefixed tag -> "-gpu", everything else
# (the "other" tag itself, and None for a CPU-passthrough category) -> "-other".
_PARTITION_BINS = {
    "gpu_api": "gpu", "gpu_kernel": "gpu", "gpu_memcpy": "gpu",
    "mpi_territory": "mpi",
}


def discover_proto_ranks(proto_dir):
    """Scans proto_dir (recursively) for rocprof-sys's own per-rank `*.proto` naming convention
    (`<prefix>-<N>.proto`), skipping `merged.proto` -- it has no trailing "-<N>" so it never
    matches the regex below at all, but it's also skipped by one explicit, redundant name check,
    to document the exclusion as deliberate rather than incidental. Returns a list of
    (rank_key, proto_path) tuples sorted by NUMERIC rank (not string order, so rank 10 doesn't sort
    before rank 2). Raises SystemExit if nothing matches."""
    by_rank = {}
    for path in glob.glob(os.path.join(proto_dir, "**", "*.proto"), recursive=True):
        if os.path.basename(path) == "merged.proto":
            continue
        m = _PER_RANK_PROTO_RE.search(os.path.basename(path))
        if not m:
            continue
        by_rank[m.group(1)] = path

    if not by_rank:
        raise SystemExit(
            f"error: no per-rank trace-proto files matching '-<rank>.proto' found under "
            f"{proto_dir!r}",
        )
    return [(rank_key, by_rank[rank_key]) for rank_key in sorted(by_rank, key=int)]


def add_cli_argument(parser):
    """Adds --trace-processor to parser -- pair with resolve_trace_processor() once the tool's own
    main() has parsed args."""
    parser.add_argument(
        "--trace-processor", dest="trace_processor", default=None,
        help="path to Perfetto's trace_processor_shell (or its trace_processor wrapper script) -- "
             "falls back to $FRIENDLY_ROCPROF_TRACE_PROCESSOR, then PATH, if not given",
    )


def resolve_trace_processor(args):
    """--trace-processor -> $FRIENDLY_ROCPROF_TRACE_PROCESSOR -> PATH ("trace_processor_shell",
    then "trace_processor", the Python wrapper script some installs use instead -- it just execv's
    straight to the real binary, so it takes identical arguments). This project has no opinion on
    how the tool itself was obtained (see module docstring) -- it only needs a path to run. Raises
    SystemExit with an actionable message if none resolve."""
    candidate = args.trace_processor or os.environ.get("FRIENDLY_ROCPROF_TRACE_PROCESSOR")
    if candidate:
        return candidate
    found = shutil.which("trace_processor_shell") or shutil.which("trace_processor")
    if found:
        return found
    raise SystemExit(
        "error: couldn't find Perfetto's trace_processor_shell -- pass --trace-processor PATH, "
        "set $FRIENDLY_ROCPROF_TRACE_PROCESSOR, or put trace_processor_shell (or the "
        "trace_processor wrapper script) on PATH. See "
        "https://perfetto.dev/docs/analysis/trace-processor for how to obtain it.",
    )


def _run_trace_processor(trace_processor_path, proto_path, sql):
    """Runs one `trace_processor_shell -Q <sql> <proto_path>` invocation, returning its raw stdout
    text -- the one function tests mock, so no real trace_processor_shell binary or `.proto` file
    is needed to exercise the parsing/pivoting logic below. Raises SystemExit with the
    subprocess's own stderr on a non-zero exit, rather than a bare traceback."""
    result = subprocess.run(
        [trace_processor_path, "-Q", sql, proto_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"error: {trace_processor_path} failed on {proto_path!r}: {result.stderr.strip()}",
        )
    return result.stdout


def _parse_trace_processor_csv(stdout_text):
    """Parses one trace_processor_shell query's raw CSV stdout into a list of row dicts, fixing up
    its one real quirk: a SQL NULL prints as the literal string "[NULL]" (quoted), not an empty
    cell (confirmed directly against trace_processor_shell's own C++ source, shell/query.cc's
    ExtractQueryResult()) -- left as-is, it would come out as the truthy string "[NULL]" instead
    of an empty cell, silently breaking stage1_rocprofsys_trace.py's "v if v else None"
    blank-handling convention every other stage in this pipeline relies on. Every other value
    (quoted strings, unquoted numbers) is exactly what Python's own csv module already expects, no
    further handling needed."""
    reader = csv.DictReader(io.StringIO(stdout_text))
    return [{k: ("" if v == "[NULL]" else v) for k, v in raw_row.items()} for raw_row in reader]


def _pivot_args_by_slice_id(args_rows):
    """Turns _ARGS_SQL's long-format (slice_id, flat_key, value) rows into {slice_id: {flat_key:
    value}} -- one column per distinct key actually present, never a fixed/hardcoded key list,
    since the real set of GPU-arg keys is data-dependent and varies by ROCm/rocprof-sys version. A
    (slice_id, flat_key) pair seen more than once (an array-valued arg collapsing under one flat
    key) keeps its first value and is counted for one summary warning, matching
    stage4_rocprofsys_trace_aggregate._attach_kernels_by_corr_id()'s own "more than one match,
    don't guess, warn once" convention."""
    by_slice = {}
    duplicate_count = 0
    for row in args_rows:
        slot = by_slice.setdefault(row["slice_id"], {})
        key = row["flat_key"]
        if key in slot:
            duplicate_count += 1
            continue
        slot[key] = row["int_value"] or row["string_value"] or row["real_value"] or ""

    if duplicate_count:
        print(
            f"warning: {duplicate_count} duplicate (slice, arg-key) pair(s) found -- keeping "
            "each key's first value rather than guess which is correct",
        )
    return by_slice


def write_unfiltered_csv(base_rows, dest_path):
    """Writes the plain, no-args combined CSV for one rank -- every row, base 11 columns only,
    matching the real precedent's own "unfiltered" file shape exactly (see module docstring)."""
    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BASE_COLUMNS)
        writer.writeheader()
        for row in base_rows:
            writer.writerow({col: row.get(col, "") for col in _BASE_COLUMNS})


def _partition_bin_for(category):
    return _PARTITION_BINS.get(tag_for_category(category), "other")


def write_partitioned_csvs(base_rows, args_by_slice_id, dest_stem):
    """Writes <dest_stem>-gpu.csv/-mpi.csv/-other.csv -- base_rows bucketed by
    _partition_bin_for(), each file's own dynamic arg-column header (alphabetically ordered,
    matching real precedent) computed independently from only the rows actually written to it --
    never an assumed-empty or hardcoded set: real data shows only the GPU partition ever carries
    args in practice, but this doesn't assume that stays true. Every row across the three files
    sums back to exactly len(base_rows) -- the same row-for-row-partition invariant this project's
    discover_ranks() already relies on."""
    rows_by_bin = {"gpu": [], "mpi": [], "other": []}
    for row in base_rows:
        rows_by_bin[_partition_bin_for(row.get("category"))].append(row)

    for bin_name, rows in rows_by_bin.items():
        arg_keys = sorted({
            key for row in rows for key in args_by_slice_id.get(row["slice_id"], {})
        })
        fieldnames = _BASE_COLUMNS + arg_keys

        with open(f"{dest_stem}-{bin_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out = {col: row.get(col, "") for col in _BASE_COLUMNS}
                out.update(args_by_slice_id.get(row["slice_id"], {}))
                writer.writerow(out)


def convert_rank(proto_path, output_dir, trace_processor_path, emit_unfiltered):
    """Converts one rank's `.proto` file: always writes the -gpu/-mpi/-other trio, plus the plain
    unfiltered file too when emit_unfiltered. Returns the list of paths written, in write order."""
    dest_stem = os.path.join(output_dir, os.path.splitext(os.path.basename(proto_path))[0])

    base_rows = _parse_trace_processor_csv(
        _run_trace_processor(trace_processor_path, proto_path, _BASE_SLICES_SQL),
    )
    args_rows = _parse_trace_processor_csv(
        _run_trace_processor(trace_processor_path, proto_path, _ARGS_SQL),
    )
    args_by_slice_id = _pivot_args_by_slice_id(args_rows)

    write_partitioned_csvs(base_rows, args_by_slice_id, dest_stem)
    written = [f"{dest_stem}-{b}.csv" for b in ("gpu", "mpi", "other")]

    if emit_unfiltered:
        unfiltered_path = f"{dest_stem}.csv"
        write_unfiltered_csv(base_rows, unfiltered_path)
        written.append(unfiltered_path)

    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("proto_dir", help="rocprof-sys output directory containing per-rank .proto trace files")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None,
                         help="directory to write the converted CSV files into (default: proto_dir itself)")
    parser.add_argument("--unfiltered", action="store_true",
                         help="also write the plain, combined per-rank CSV (no GPU-arg columns) "
                              "alongside the default -gpu/-mpi/-other trio")
    add_cli_argument(parser)
    args = parser.parse_args(argv)

    stage6_cli_common.require_directory(args.proto_dir)
    trace_processor_path = resolve_trace_processor(args)
    output_dir = args.output_dir or args.proto_dir
    os.makedirs(output_dir, exist_ok=True)

    for _rank_key, proto_path in discover_proto_ranks(args.proto_dir):
        for path in convert_rank(proto_path, output_dir, trace_processor_path, args.unfiltered):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
