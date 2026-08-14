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
from datetime import datetime

from stage1_rocprofsys import PID_SUFFIX_RE
from stage4_rocprofsys_flat import aggregate, aggregate_per_rank
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, load_imbalance_columns
from stage5_table_render import render_table, select_entries

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
    key_field = "sum" if rank_by == "inclusive" else "self_sum"

    def _set_pct_total(entries):
        for e in entries:
            e["pct_total"] = (e[key_field] / total_runtime * 100.0) if total_runtime > 0 else None

    cpu_selected, cpu_desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime", prepare=_set_pct_total,
    )
    gpu_selected, gpu_desc = select_entries(
        gpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime", prepare=_set_pct_total,
    )

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
    parts.append(render_table(CPU_HOTSPOTS_COLUMNS, cpu_selected))
    parts.append("\n")

    parts.append(f"GPU API / launch overhead -- showing {gpu_desc}\n")
    parts.append("(host-side call overhead only -- NOT device kernel execution time)\n")
    parts.append(render_table(CPU_HOTSPOTS_COLUMNS, gpu_selected))
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
        parts.append(render_table(load_imbalance_columns(), imbalance_selected))
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
