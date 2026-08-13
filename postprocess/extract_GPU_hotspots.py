#!/usr/bin/env python3
"""Extract a short GPU kernel hotspots report from rocprofv3 kernel_stats.csv output.

Only reads `*_kernel_stats.csv` (produced by `rocprofv3 --kernel-trace --stats
--output-format csv`). This is real GPU device execution time -- unlike
rocprof-sys's text/JSON output, which only ever captures host-side timing.
For CPU-side hotspots, see scripts/profile_CPU_hotspots.sh instead.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

from stage1_rocprofv3 import parse_kernel_stats_csv
from rank_merge_math import stats_across_ranks

# rocprofv3's --output-config (-> <pid>_config.json) is a post-ROCm-7.0.2 feature;
# absent that file (the common case for our 7.0.2 compatibility target), header
# fields below just stay blank. No documented field-name schema was found for it
# either, so this is a best-effort guess, same philosophy as extract_CPU_hotspots.py.
CONFIG_EXECUTABLE_KEYS = ["command", "command_line", "argv", "cmd", "exe", "executable"]
CONFIG_DATETIME_KEYS = ["init_time", "start_time", "launch_time", "timestamp"]
CONFIG_RUNTIME_KEYS = ["elapsed", "duration", "wall_time", "total_time", "runtime"]

# rocprofv3's default naming is "<hostname>/<pid>_kernel_stats.csv" -- used as a
# fallback rank count (one file per process/rank) when config.json lacks one.
PID_SUFFIX_RE = re.compile(r"(\d+)_kernel_stats\.csv$")

HELP_BLURB = """\
Reads the output of a profile_GPU_hotspots.sh run (or any rocprofv3 output
directory) and writes a short, ranked text report: which GPU kernels
actually spend the most time executing on the GPU itself.

This is real device execution time, telling you which pieces of GPU work
are worth optimizing first. It does NOT show CPU-side hotspots (functions
still running on the CPU, possibly candidates for offloading to the GPU in
the first place) -- for that, see extract_CPU_hotspots.py, or
extract_hotspots.py for both combined.

Numbers are percentages of total measured GPU time -- good enough to spot
your top bottleneck, not a precise, reproducible benchmark.

Under the hood, this parses output written by AMD's rocprofv3 -- see
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
for details.
"""


def aggregate(output_dir):
    """Recursively find *_kernel_stats.csv under output_dir and aggregate by
    kernel name. Returns (entries, scanned_files, total_ns) where entries is
    [{"label", "count", "total_ns"}] (unsorted) and total_ns is the sum of
    every row's TotalDurationNs across every scanned file -- the denominator
    for each entry's "% of total" (rocprofv3 already aggregates duplicate
    kernel names *within* one file, so no per-file dedup is needed here).
    """
    scanned_files = []
    totals = {}  # label -> {"count": int, "total_ns": float}
    total_ns = 0.0

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*_kernel_stats.csv"), recursive=True))
    for path in candidates:
        rows = parse_kernel_stats_csv(path)
        if rows is None:
            continue
        scanned_files.append(path)
        for row in rows:
            entry = totals.setdefault(row["label"], {"count": 0, "total_ns": 0.0})
            entry["count"] += row["count"]
            entry["total_ns"] += row["total_ns"]
            total_ns += row["total_ns"]

    entries = []
    for label, entry in totals.items():
        pct_total = (entry["total_ns"] / total_ns * 100.0) if total_ns > 0 else None
        avg_us = (entry["total_ns"] / entry["count"] / 1000.0) if entry["count"] > 0 else None
        entries.append({
            "label": label,
            "count": entry["count"],
            "sum": entry["total_ns"] / 1e9,  # seconds, for consistent naming with extract_CPU_hotspots.py
            "avg_us": avg_us,
            "pct_total": pct_total,
        })

    return entries, scanned_files, total_ns


def aggregate_per_rank(output_dir):
    """Like aggregate(), but keeps each scanned file's per-kernel sums
    separate instead of merging them into one global total -- one scanned
    file is treated as one rank's contribution (same file-per-process
    assumption guess_num_ranks() already relies on). rocprofv3 already
    aggregates duplicate kernel names within one file internally, so each
    file's own rows are already a clean per-rank subtotal, same as
    aggregate() itself relies on.

    Returns (per_file_totals, scanned_files) where per_file_totals is a
    list of {kernel_name: total_seconds} dicts, one per scanned file, in
    the same order as scanned_files.
    """
    scanned_files = []
    per_file_totals = []

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*_kernel_stats.csv"), recursive=True))
    for path in candidates:
        rows = parse_kernel_stats_csv(path)
        if rows is None:
            continue
        scanned_files.append(path)
        file_totals = {}
        for row in rows:
            file_totals[row["label"]] = file_totals.get(row["label"], 0.0) + row["total_ns"] / 1e9
        per_file_totals.append(file_totals)

    return per_file_totals, scanned_files


def compute_load_imbalance(per_file_totals, top=None, threshold=None, show_all=False):
    """Per-kernel avg/std_dev/min/max of each rank's own total time in that kernel, across all
    ranks in per_file_totals -- the stats themselves come from the shared
    rank_merge_math.stats_across_ranks(); the surrounding per-label loop and the
    show_all/threshold/top selection below are still a standalone copy of
    extract_CPU_hotspots.py's function of the same name. A rank missing a kernel counts as
    0.0 for that rank, not omitted. --threshold here means coefficient of variation
    (std_dev / avg, as a %), not % of total GPU time.
    """
    labels = {label for ft in per_file_totals for label in ft}
    entries = []
    for label in labels:
        values = [ft.get(label, 0.0) for ft in per_file_totals]
        stats = stats_across_ranks(values)
        entries.append({
            "label": label,
            "avg": stats["avg"],
            "std_dev": stats["std_dev"],
            "min": stats["min"],
            "max": stats["max"],
            "cv_pct": (stats["std_dev"] / stats["avg"] * 100.0) if stats["avg"] > 0 else None,
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
    lines.append(f"  {'#':>3}  {'avg(s)':>12}  {'std_dev':>10}  {'min(s)':>12}  {'max(s)':>12}  kernel")
    for i, e in enumerate(entries, 1):
        lines.append(f"  {i:>3}  {e['avg']:>12.6f}  {e['std_dev']:>10.6f}  {e['min']:>12.6f}  {e['max']:>12.6f}  {e['label']}")
    return "\n".join(lines) + "\n"


def select_entries(entries, total_ns, top=None, threshold=None, show_all=False):
    """Same top/threshold/all selection semantics as extract_CPU_hotspots.py --
    duplicated rather than imported, since each extractor is meant to stand alone."""
    entries_sorted = sorted(entries, key=lambda e: e["sum"], reverse=True)
    total_count = len(entries_sorted)

    if show_all:
        return entries_sorted, f"all {total_count} entries"

    if threshold is not None:
        if total_ns <= 0:
            return entries_sorted, f"all {total_count} entries (total runtime unknown, threshold ignored)"
        filtered = [e for e in entries_sorted if e["pct_total"] is not None and e["pct_total"] >= threshold]
        return filtered, f">= {threshold:g}% of total runtime ({len(filtered)} of {total_count} entries)"

    n = 20 if top is None else top
    return entries_sorted[:n], f"top {n} of {total_count} entries"


def format_table(entries):
    if not entries:
        return "  (none found)\n"
    lines = []
    lines.append(f"  {'#':>3}  {'total(s)':>12}  {'%total':>7}  {'calls':>10}  {'avg(us)':>10}  kernel")
    for i, e in enumerate(entries, 1):
        pct_total_str = f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"
        avg_str = f"{e['avg_us']:.2f}" if e["avg_us"] is not None else "n/a"
        lines.append(f"  {i:>3}  {e['sum']:>12.6f}  {pct_total_str:>7}  {e['count']:>10}  {avg_str:>10}  {e['label']}")
    return "\n".join(lines) + "\n"


def load_config_json(output_dir):
    """Best-effort search for rocprofv3's optional <pid>_config.json (from
    --output-config). Returns the first one found, parsed, or {} if none exists
    or it fails to parse -- this is a post-ROCm-7.0.2 feature, so absence is
    the expected common case, not an error."""
    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*_config.json"), recursive=True))
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def find_first_key(d, candidate_keys, _depth=0):
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


def guess_executable(config):
    val = find_first_key(config, CONFIG_EXECUTABLE_KEYS)
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, str) and val.strip():
        return os.path.basename(val.split()[0])
    return None


def guess_run_datetime(config):
    val = find_first_key(config, CONFIG_DATETIME_KEYS)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def guess_total_runtime(config):
    val = find_first_key(config, CONFIG_RUNTIME_KEYS)
    if isinstance(val, (int, float)):
        return f"{val:.6f} sec"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def guess_num_ranks(config, scanned_files):
    pids = set()
    for path in scanned_files:
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        if m:
            pids.add(m.group(1))
    return len(pids) if pids else None


def gather_run_info(output_dir, scanned_files):
    config = load_config_json(output_dir)
    return {
        "executable": guess_executable(config),
        "run_datetime": guess_run_datetime(config),
        "total_runtime": guess_total_runtime(config),
        "num_ranks": guess_num_ranks(config, scanned_files),
    }


def write_report(output_dir, dest_path, top=None, threshold=None, show_all=False):
    entries, scanned_files, total_ns = aggregate(output_dir)
    if not scanned_files:
        raise SystemExit(
            f"error: no rocprofv3 kernel_stats.csv found under {output_dir!r} "
            "(expected files like <pid>_kernel_stats.csv, from "
            "'rocprofv3 --kernel-trace --stats --output-format csv') -- nothing to report"
        )

    run_info = gather_run_info(output_dir, scanned_files)
    selected, desc = select_entries(entries, total_ns, top, threshold, show_all)

    parts = []
    parts.append("rocprofv3 GPU kernel hotspots report\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(output_dir)}\n")
    parts.append(f"executable: {run_info['executable'] or ''}\n")
    parts.append(f"run date/time: {run_info['run_datetime'] or ''}\n")
    parts.append(f"total runtime: {run_info['total_runtime'] or ''}\n")
    parts.append(f"MPI ranks: {run_info['num_ranks'] if run_info['num_ranks'] is not None else ''}\n")
    parts.append("files scanned:\n")
    for f in scanned_files:
        parts.append(f"  - {os.path.relpath(f, output_dir)}\n")
    parts.append("\n")

    parts.append(f"GPU kernel hotspots -- showing {desc}\n")
    parts.append(format_table(selected))
    parts.append("\n")

    parts.append(
        "Note: this covers GPU kernel execution time only. '%total' is each "
        "kernel's share of total measured GPU time (summed across all scanned "
        "files); it will not add up to 100% if a --threshold/--top cut entries.\n"
        "For host-side (HIP API / launch overhead) hotspots, use scripts/profile_CPU_hotspots.sh.\n"
    )
    parts.append("\n")

    per_file_totals, imbalance_scanned = aggregate_per_rank(output_dir)
    if len(imbalance_scanned) < 2:
        parts.append(
            "GPU kernel load imbalance across ranks -- skipped: only "
            f"{len(imbalance_scanned)} rank/file found, need at least 2 to compare.\n"
        )
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_file_totals, top, threshold, show_all)
        parts.append(
            f"GPU kernel load imbalance across {len(imbalance_scanned)} ranks -- showing {imbalance_desc}\n"
        )
        parts.append(
            "Each kernel's own total time on each rank, compared across ranks -- a rank "
            "that never launched a kernel counts as 0.0 for that rank, not omitted.\n"
        )
        parts.append(format_table_load_imbalance(imbalance_selected))

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprofv3 output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspots report (default: <output_dir>/hotspots.txt)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-n", "--top", dest="top", type=int, default=None,
                            help="number of hotspots to list (default: 20)")
    selection.add_argument("--threshold", dest="threshold", type=float, default=None,
                            help="only list kernels at or above this %% of total GPU time")
    selection.add_argument("--all", dest="show_all", action="store_true",
                            help="list every kernel, no truncation")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "hotspots.txt")
    write_report(args.output_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
