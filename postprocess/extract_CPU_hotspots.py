#!/usr/bin/env python3
"""Extract a short CPU-side hotspots report from rocprof-sys timemory text output.

Only reads the well-documented pipe-delimited "timemory" text tables
(e.g. wall_clock-<pid>.txt) that rocprof-sys writes for CPU-side timing.
GPU device kernel execution time is NOT present in this data -- see the
footer note this script writes into its own output.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

NON_TIMING_FILES = {"available.txt", "instrumented.txt", "excluded.txt", "overlapping.txt"}

GPU_API_PREFIXES = ("hip", "hsa", "roctx", "kfd", "rocdecode", "rocjpeg")
GPU_FILE_HINTS = ("roctracer", "hsa")

EXPECTED_HEADER_FIELDS = ["LABEL", "COUNT", "DEPTH", "METRIC", "UNITS", "SUM", "MEAN", "MIN", "MAX", "VAR", "STDDEV", "% SELF"]
# COUNT..% SELF -- everything after LABEL. Kept as a count, not the literal names,
# because under MPI the docs describe the row prefix as "|MM|NN>>>label" -- an
# embedded pipe that would otherwise misalign a naive fixed-column split.
FIXED_FIELDS_AFTER_LABEL = len(EXPECTED_HEADER_FIELDS) - 1

METADATA_FILENAME = "metadata.json"
# Field names are not documented anywhere -- these are best-effort guesses tried
# in order; if none match (or metadata.json is absent), the corresponding header
# field is just left blank, per this tool's "never error on missing metadata" rule.
EXECUTABLE_KEYS = ["command_line", "argv", "command", "exe", "executable"]
RUN_DATETIME_KEYS = ["start_time", "launch_time", "timestamp", "date", "time"]
TOTAL_RUNTIME_KEYS = ["elapsed", "duration", "wall_time", "total_time", "runtime"]
NUM_RANKS_KEYS = ["num_ranks", "world_size", "mpi_size", "ranks", "num_procs"]

# rocprof-sys's default ROCPROFSYS_TIME_OUTPUT subdirectory naming (documented default
# strftime pattern "%F_%H.%M", e.g. "2025-01-21_07.40") -- used as a fallback run
# date/time source when metadata.json doesn't have (or isn't) available.
TIME_OUTPUT_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}")
# rocprof-sys's default per-process file naming is "<component>-<pid>.txt" -- used as
# a fallback rank count (one file per process/rank) when metadata.json lacks a count.
PID_SUFFIX_RE = re.compile(r"-(\d+)\.txt$")

HELP_BLURB = """\
Reads the output of a profile_CPU_hotspots.sh run (or any rocprof-sys output
directory) and writes a short, ranked text report: which functions spend
the most time on the CPU, including time spent just waiting for the GPU.

Use this to find CPU-side work worth moving to the GPU ("offloading"), or
CPU code that's simply slow. It does NOT tell you which GPU kernels are
slow on the GPU itself -- for that, see extract_GPU_hotspots.py, or
extract_hotspots.py for both combined.

Numbers are percentages of total measured time -- good enough to spot your
top bottleneck, not a precise, reproducible benchmark.

Under the hood, this parses output written by AMD's rocprof-sys (ROCm
Systems Profiler) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def parse_table_file(path):
    """Parse one timemory pipe-delimited text table. Returns a list of dict rows, or None if this
    file doesn't look like a timemory table at all."""
    with open(path, "r", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    header_idx = None
    for i, line in enumerate(lines):
        fields = [c.strip() for c in line.strip("|").split("|")]
        if fields == EXPECTED_HEADER_FIELDS:
            header_idx = i
            break
    if header_idx is None:
        return None

    rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        fields = [c.strip() for c in stripped.strip("|").split("|")]
        if len(fields) < FIXED_FIELDS_AFTER_LABEL + 1:
            continue
        # Everything before the last FIXED_FIELDS_AFTER_LABEL fields is the label --
        # rejoined with "|" in case a rank prefix split it into more than one piece.
        raw_label = "|".join(fields[:-FIXED_FIELDS_AFTER_LABEL])
        count, depth, metric, units, total, mean, vmin, vmax, var, stddev, pct_self = fields[-FIXED_FIELDS_AFTER_LABEL:]
        label = clean_label(raw_label)
        try:
            total_f = float(total)
            rows.append({
                "label": label,
                "count": int(count),
                "sum": total_f,
                # This node's own (self) time, in seconds -- % SELF is only ever
                # meaningful per call-tree node, not once merged by label, so it's
                # converted here and accumulated as a plain sum from then on (see
                # aggregate()/aggregate_per_rank()); the raw % SELF isn't kept.
                "self_sum": total_f * float(pct_self) / 100.0,
            })
        except ValueError:
            continue
    return rows


def clean_label(raw_label):
    """Strip rocprof-sys's thread/rank prefix (|NN>>> or |MM|NN>>>) and hierarchy
    indentation (|_ repeated per call-stack depth) from a raw LABEL field."""
    label = raw_label
    if ">>>" in label:
        label = label.split(">>>", 1)[1]
    while label.startswith("|_"):
        label = label[2:]
    return label.strip()


def is_gpu_entry(label, filename):
    lname = label.lower()
    if lname.startswith(GPU_API_PREFIXES):
        return True
    fname = os.path.basename(filename).lower()
    return any(hint in fname for hint in GPU_FILE_HINTS)


def aggregate(output_dir):
    """Scan output_dir for timemory text tables and aggregate rows by clean function name.

    Returns (cpu_entries, gpu_entries, scanned_files, total_runtime) where each
    entries list is [{"label", "count", "sum", "self_sum", "pct_self", "pct_total"}],
    unsorted, and total_runtime is the denominator used for each entry's "% of
    total runtime": the sum, across all scanned files, of that file's own
    largest SUM value (a file's largest SUM is -- barring unusual
    instrumentation -- its outermost/root scope, since inclusive time only
    grows going up the call stack; this works whether the file is a
    hierarchical or a flattened profile, without needing to guess the root
    function's name).

    "sum" is inclusive time (this function plus everything it calls); "self_sum"
    is its own time only, summed across every call-tree node with this label --
    the metric select_entries() ranks by default, since it's the one that
    actually tells apart a real hotspot from a function that just calls the
    next thing (which is why a flat profile, where % SELF is always 100, can't
    support this distinction -- see scripts/profile_CPU_hotspots.sh). Entries'
    own "pct_total" here is self_sum-based (select_entries() recomputes it
    against whichever metric it's actually ranking by, so this is just a
    sensible default for callers that use aggregate()'s output directly).
    """
    scanned_files = []
    file_rows = []  # [(path, rows)]

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True))
    for path in candidates:
        if os.path.basename(path) in NON_TIMING_FILES:
            continue
        rows = parse_table_file(path)
        if rows is None:
            continue
        scanned_files.append(path)
        file_rows.append((path, rows))

    total_runtime = sum(max(row["sum"] for row in rows) for _, rows in file_rows if rows)

    totals = {}  # label -> {"count": int, "sum": float, "self_sum": float, "gpu": bool}
    for path, rows in file_rows:
        for row in rows:
            label = row["label"]
            gpu = is_gpu_entry(label, path)
            entry = totals.setdefault(label, {"count": 0, "sum": 0.0, "self_sum": 0.0, "gpu": gpu})
            entry["count"] += row["count"]
            entry["sum"] += row["sum"]
            entry["self_sum"] += row["self_sum"]
            entry["gpu"] = entry["gpu"] or gpu

    cpu_entries = []
    gpu_entries = []
    for label, entry in totals.items():
        pct_total = (entry["self_sum"] / total_runtime * 100.0) if total_runtime > 0 else None
        pct_self = (entry["self_sum"] / entry["sum"] * 100.0) if entry["sum"] > 0 else None
        item = {
            "label": label,
            "count": entry["count"],
            "sum": entry["sum"],
            "self_sum": entry["self_sum"],
            "pct_self": pct_self,
            "pct_total": pct_total,
        }
        (gpu_entries if entry["gpu"] else cpu_entries).append(item)

    return cpu_entries, gpu_entries, scanned_files, total_runtime


def aggregate_per_rank(output_dir, unfiltered=False):
    """Like aggregate(), but keeps each scanned file's CPU-only per-label
    totals separate instead of merging them into one global total -- one
    scanned file is treated as one rank's contribution (same file-per-
    process assumption guess_num_ranks() already relies on), which is what
    a load-imbalance-across-ranks computation needs as its input.

    Uses self-time by default (not inclusive) for the same reason aggregate()
    ranks by it by default -- keeps the load-imbalance table's ranking
    consistent with the main hotspots table in the same report; unfiltered=True
    switches to inclusive time, matching aggregate()'s own --unfiltered view.

    Returns (per_file_totals, scanned_files) where per_file_totals is a
    list of {label: value} dicts, one per scanned file, in the same order
    as scanned_files.
    """
    metric = "sum" if unfiltered else "self_sum"
    scanned_files = []
    per_file_totals = []

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True))
    for path in candidates:
        if os.path.basename(path) in NON_TIMING_FILES:
            continue
        rows = parse_table_file(path)
        if rows is None:
            continue
        scanned_files.append(path)
        file_totals = {}
        for row in rows:
            label = row["label"]
            if is_gpu_entry(label, path):
                continue
            file_totals[label] = file_totals.get(label, 0.0) + row[metric]
        per_file_totals.append(file_totals)

    return per_file_totals, scanned_files


def compute_load_imbalance(per_file_totals, top=None, threshold=None, show_all=False):
    """Per-label avg/std_dev/min/max of each rank's own total time in that
    label, across all ranks in per_file_totals. A rank that never shows up
    for a given label contributes 0.0 (it genuinely spent no time there),
    not a skipped/missing value -- a function that only runs on some ranks
    is real, extreme imbalance, not something to hide.

    Selection mirrors select_entries()'s top/threshold/show_all shape, but
    ranked by std_dev (not total time), and --threshold here means
    coefficient of variation (std_dev / avg, as a %) instead of % of total
    runtime -- a %-of-runtime cutoff has no equivalent meaning for a
    std_dev ranking. Returns (selected, description), same shape as
    select_entries().
    """
    labels = {label for ft in per_file_totals for label in ft}
    entries = []
    for label in labels:
        values = [ft.get(label, 0.0) for ft in per_file_totals]
        avg = statistics.mean(values)
        std_dev = statistics.pstdev(values)
        entries.append({
            "label": label,
            "avg": avg,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values),
            "cv_pct": (std_dev / avg * 100.0) if avg > 0 else None,
        })

    entries_sorted = sorted(entries, key=lambda e: e["std_dev"], reverse=True)
    total_count = len(entries_sorted)

    if show_all:
        return entries_sorted, f"all {total_count} entries"

    if threshold is not None:
        filtered = [e for e in entries_sorted if e["cv_pct"] is not None and e["cv_pct"] >= threshold]
        return filtered, f">= {threshold:g}% coefficient of variation ({len(filtered)} of {total_count} entries)"

    n = 20 if top is None else top
    return entries_sorted[:n], f"top {n} of {total_count} entries by std_dev"


def format_table_load_imbalance(entries):
    if not entries:
        return "  (none found)\n"
    lines = []
    lines.append(f"  {'#':>3}  {'avg(s)':>12}  {'std_dev':>10}  {'min(s)':>12}  {'max(s)':>12}  function")
    for i, e in enumerate(entries, 1):
        lines.append(f"  {i:>3}  {e['avg']:>12.6f}  {e['std_dev']:>10.6f}  {e['min']:>12.6f}  {e['max']:>12.6f}  {e['label']}")
    return "\n".join(lines) + "\n"


def select_entries(entries, total_runtime, top=None, threshold=None, show_all=False, rank_by="self"):
    """Pick which aggregated entries to report, sorted by rank_by descending
    ("self" -- self_sum, the default -- or "inclusive" -- sum, --unfiltered's
    view). Each entry's "pct_total" is (re)computed here against whichever
    metric is actually being ranked, so a % shown in the report always means
    "% of total runtime by the metric this table is sorted by" -- overwrites
    whatever aggregate() put there.

    Exactly one selection mode applies (show_all > threshold > top, in that
    precedence, though callers should only set one): show every entry, keep
    only entries at or above a %-of-total-runtime threshold, or keep the top N
    by the ranked metric. Returns (selected_entries, description_for_report_header).
    """
    key_field = "sum" if rank_by == "inclusive" else "self_sum"
    for e in entries:
        e["pct_total"] = (e[key_field] / total_runtime * 100.0) if total_runtime > 0 else None

    entries_sorted = sorted(entries, key=lambda e: e[key_field], reverse=True)
    total_count = len(entries_sorted)

    if show_all:
        return entries_sorted, f"all {total_count} entries"

    if threshold is not None:
        if total_runtime <= 0:
            return entries_sorted, f"all {total_count} entries (total runtime unknown, threshold ignored)"
        filtered = [e for e in entries_sorted if e["pct_total"] is not None and e["pct_total"] >= threshold]
        return filtered, f">= {threshold:g}% of total runtime ({len(filtered)} of {total_count} entries)"

    n = 20 if top is None else top
    return entries_sorted[:n], f"top {n} of {total_count} entries"


def format_table(entries):
    if not entries:
        return "  (none found)\n"
    lines = []
    lines.append(
        f"  {'#':>3}  {'self(s)':>12}  {'%total':>7}  {'total(s)':>12}  {'calls':>10}  {'%self':>7}  function"
    )
    for i, e in enumerate(entries, 1):
        pct_total_str = f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"
        pct_self_str = f"{e['pct_self']:.1f}" if e.get("pct_self") is not None else "n/a"
        lines.append(
            f"  {i:>3}  {e['self_sum']:>12.6f}  {pct_total_str:>7}  {e['sum']:>12.6f}  "
            f"{e['count']:>10}  {pct_self_str:>7}  {e['label']}"
        )
    return "\n".join(lines) + "\n"


def find_extra_artifacts(output_dir):
    proto_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.proto"), recursive=True))
    db_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.db"), recursive=True))
    return proto_files, db_files


def load_metadata(output_dir):
    candidates = sorted(glob.glob(os.path.join(output_dir, "**", METADATA_FILENAME), recursive=True))
    if not candidates:
        return {}
    try:
        with open(candidates[0]) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def find_first_key(d, candidate_keys, _depth=0):
    """Best-effort case-insensitive key search, one level of nested dicts deep --
    metadata.json's schema isn't documented, so this is a guess, not a parse."""
    if not isinstance(d, dict):
        return None
    lower_map = {k.lower(): v for k, v in d.items()}
    for key in candidate_keys:
        if key in lower_map and lower_map[key] not in (None, "", []):
            return lower_map[key]
    if _depth == 0:
        for v in d.values():
            if isinstance(v, dict):
                found = find_first_key(v, candidate_keys, _depth=1)
                if found is not None:
                    return found
    return None


def guess_executable(metadata):
    val = find_first_key(metadata, EXECUTABLE_KEYS)
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, str) and val.strip():
        return os.path.basename(val.split()[0])
    return None


def guess_run_datetime(metadata, output_dir, scanned_files=()):
    val = find_first_key(metadata, RUN_DATETIME_KEYS)
    if isinstance(val, str) and val.strip():
        return val.strip()
    m = TIME_OUTPUT_DIR_RE.search(output_dir)
    if m:
        return m.group(0)
    # rocprof-sys's default time-stamped subdirectory is found via a recursive glob rather
    # than being part of the output_dir path passed in -- look for it in each scanned file's
    # own directory instead.
    for path in scanned_files:
        m = TIME_OUTPUT_DIR_RE.search(os.path.dirname(path))
        if m:
            return m.group(0)
    return None


def guess_total_runtime(metadata):
    val = find_first_key(metadata, TOTAL_RUNTIME_KEYS)
    if isinstance(val, (int, float)):
        return f"{val:.6f} sec"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def guess_num_ranks(metadata, scanned_files):
    val = find_first_key(metadata, NUM_RANKS_KEYS)
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    pids = set()
    for path in scanned_files:
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        if m:
            pids.add(m.group(1))
    return len(pids) if pids else None


def gather_run_info(output_dir, scanned_files):
    metadata = load_metadata(output_dir)
    return {
        "executable": guess_executable(metadata),
        "run_datetime": guess_run_datetime(metadata, output_dir, scanned_files),
        "total_runtime": guess_total_runtime(metadata),
        "num_ranks": guess_num_ranks(metadata, scanned_files),
    }


def write_report(output_dir, dest_path, top=None, threshold=None, show_all=False, unfiltered=False):
    cpu_entries, gpu_entries, scanned_files, total_runtime = aggregate(output_dir)
    if not scanned_files:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found in {output_dir!r} "
            "(expected files like wall_clock-<pid>.txt with the documented "
            "LABEL|COUNT|DEPTH|METRIC|... header) -- nothing to report"
        )

    proto_files, db_files = find_extra_artifacts(output_dir)
    run_info = gather_run_info(output_dir, scanned_files)

    rank_by = "inclusive" if unfiltered else "self"
    cpu_selected, cpu_desc = select_entries(cpu_entries, total_runtime, top, threshold, show_all, rank_by)
    gpu_selected, gpu_desc = select_entries(gpu_entries, total_runtime, top, threshold, show_all, rank_by)

    parts = []
    parts.append("rocprof-sys hotspots report (CPU-side only)\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(output_dir)}\n")
    parts.append(f"executable: {run_info['executable'] or ''}\n")
    parts.append(f"run date/time: {run_info['run_datetime'] or ''}\n")
    parts.append(f"total runtime: {run_info['total_runtime'] or ''}\n")
    parts.append(f"MPI ranks: {run_info['num_ranks'] if run_info['num_ranks'] is not None else ''}\n")
    parts.append("files scanned:\n")
    for f in scanned_files:
        parts.append(f"  - {os.path.basename(f)}\n")
    parts.append("\n")

    if unfiltered:
        parts.append(
            "Ranked by inclusive (total) time -- a function that only calls other "
            "functions can still rank high here. Drop --unfiltered for the "
            "self-time view.\n"
        )
    else:
        parts.append(
            "Ranked by self time -- each function's own work, not counting time "
            "spent in whatever it calls, so pass-through functions (a function "
            "that just calls the next thing) fall out of the ranking on their "
            "own. Pass --unfiltered for the old inclusive/cumulative-time view.\n"
        )
    parts.append("\n")

    parts.append(f"CPU compute hotspots (candidates for GPU offload) -- showing {cpu_desc}\n")
    parts.append(format_table(cpu_selected))
    parts.append("\n")

    parts.append(f"GPU API / launch overhead -- showing {gpu_desc}\n")
    parts.append("(host-side call overhead only -- NOT device kernel execution time)\n")
    parts.append(format_table(gpu_selected))
    parts.append("\n")

    parts.append(
        "Note: true GPU kernel execution time is not present in this data. "
        "rocprof-sys's text/JSON output only captures host-side timing; "
        "GPU-launch-looking rows above are launch/API overhead, not device time. "
        "'%total' is each function's share of total measured time (summed across "
        "all scanned files); it will not add up to 100% across both tables.\n"
    )
    parts.append("\n")

    per_file_totals, imbalance_scanned = aggregate_per_rank(output_dir, unfiltered=unfiltered)
    if len(imbalance_scanned) < 2:
        parts.append(
            "CPU load imbalance across ranks -- skipped: only "
            f"{len(imbalance_scanned)} rank/file found, need at least 2 to compare.\n"
        )
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_file_totals, top, threshold, show_all)
        parts.append(
            f"CPU load imbalance across {len(imbalance_scanned)} ranks -- showing {imbalance_desc}\n"
        )
        parts.append(
            ("Each function's own inclusive" if unfiltered else "Each function's own self")
            + " time on each rank, compared across ranks -- a rank "
            "that never called a function counts as 0.0 for that rank, not omitted.\n"
        )
        parts.append(format_table_load_imbalance(imbalance_selected))
    parts.append("\n")

    if proto_files:
        parts.append("A Perfetto trace was also found (not parsed by this tool):\n")
        for f in proto_files:
            parts.append(f"  - {f}\n")
    if db_files:
        parts.append("A rocpd database was also found (ROCm 7.1+ only, not parsed by this tool):\n")
        for f in db_files:
            parts.append(f"  - {f}\n")
    parts.append(
        "For real GPU kernel hotspots, use scripts/profile_GPU_hotspots.sh "
        "(or run: rocprofv3 --kernel-trace --stats --output-format csv -- <app>)\n"
    )

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspots report (default: <output_dir>/hotspots.txt)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-n", "--top", dest="top", type=int, default=None,
                            help="number of hotspots to list per section (default: 20)")
    selection.add_argument("--threshold", dest="threshold", type=float, default=None,
                            help="only list entries at or above this %% of total runtime")
    selection.add_argument("--all", dest="show_all", action="store_true",
                            help="list every entry, no truncation")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="rank by inclusive (total) time instead of self time -- the old "
                              "behavior, where a function that just calls other functions can "
                              "still rank high")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "hotspots.txt")
    write_report(args.output_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 unfiltered=args.unfiltered)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
