#!/usr/bin/env python3
"""Extract a short GPU kernel hotspots report from rocprofv3 kernel_stats.csv output.

Only reads *_kernel_stats.csv files rocprofv3 writes with --kernel-trace --stats
--output-format csv. This is REAL device kernel execution time, unlike
extract_CPU_hotspots.py's host-side timing -- see extract_hotspots.py to combine
both into one report.
"""

import argparse
import os
import re
from datetime import datetime

from stage4_rocprofv3 import aggregate, aggregate_per_rank
from stage5_gpu_hotspots_table import GPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, load_imbalance_columns
from stage5_table_render import render_table, select_entries
from stage6_report_builder import render_report, write_report_file
from stage6_run_metadata import guess_executable, guess_num_ranks, guess_run_datetime, guess_total_runtime, load_json_file

CONFIG_EXECUTABLE_KEYS = ["command", "command_line", "argv", "cmd", "exe", "executable"]
CONFIG_DATETIME_KEYS = ["init_time", "start_time", "launch_time", "timestamp"]
CONFIG_RUNTIME_KEYS = ["elapsed", "duration", "wall_time", "total_time", "runtime"]

PID_SUFFIX_RE = re.compile(r"(\d+)_kernel_stats\.csv$")

HELP_BLURB = """\
Reads the output of a profile_GPU_hotspots.sh run (or any rocprofv3 output
directory) and writes a short, ranked text report: which GPU kernels take
the most device execution time.

This is REAL GPU kernel execution time, measured on the device itself --
unlike extract_CPU_hotspots.py, which only sees host-side (CPU) timing and
can't tell you how long a kernel actually ran on the GPU. For host-side
launch/API overhead, or a combined CPU+GPU view, see extract_hotspots.py.

Under the hood, this parses output written by AMD's rocprofv3 -- see
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/ for details.
"""


def gather_run_info(output_dir, scanned_files):
    data = load_json_file(output_dir, "*_config.json")
    return {
        "executable": guess_executable(data, CONFIG_EXECUTABLE_KEYS),
        "run_datetime": guess_run_datetime(data, CONFIG_DATETIME_KEYS),
        "total_runtime": guess_total_runtime(data, CONFIG_RUNTIME_KEYS),
        "num_ranks": guess_num_ranks(data, PID_SUFFIX_RE, scanned_files),
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
    selected, desc = select_entries(
        entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
    )

    per_file_totals, imbalance_scanned = aggregate_per_rank(output_dir)
    if len(imbalance_scanned) < 2:
        imbalance_title, imbalance_body = None, (
            "GPU kernel load imbalance across ranks -- skipped: only "
            f"{len(imbalance_scanned)} rank/file found, need at least 2 to compare.\n"
        )
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_file_totals, top, threshold, show_all)
        imbalance_title = f"GPU kernel load imbalance across {len(imbalance_scanned)} ranks -- showing {imbalance_desc}\n"
        imbalance_body = (
            "Each kernel's own total time on each rank, compared across ranks -- a rank "
            "that never launched a kernel counts as 0.0 for that rank, not omitted.\n"
        ) + render_table(load_imbalance_columns(item_label="kernel"), imbalance_selected)

    header = (
        "rocprofv3 GPU kernel hotspots report\n"
        f"generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"source directory: {os.path.abspath(output_dir)}\n"
        f"executable: {run_info['executable'] or ''}\n"
        f"run date/time: {run_info['run_datetime'] or ''}\n"
        f"total runtime: {run_info['total_runtime'] or ''}\n"
        f"MPI ranks: {run_info['num_ranks'] if run_info['num_ranks'] is not None else ''}\n"
        "files scanned:\n"
        + "".join(f"  - {os.path.relpath(f, output_dir)}\n" for f in scanned_files)
        + "\n"
    )
    sections = [
        (f"GPU kernel hotspots -- showing {desc}\n", render_table(GPU_HOTSPOTS_COLUMNS, selected)),
        (None,
         "Note: this covers GPU kernel execution time only. '%total' is each "
         "kernel's share of total measured GPU time (summed across all scanned "
         "files); it will not add up to 100% if a --threshold/--top cut entries.\n"
         "For host-side (HIP API / launch overhead) hotspots, use scripts/profile_CPU_hotspots.sh.\n"),
        (imbalance_title, imbalance_body),
    ]

    return write_report_file(dest_path, render_report(header, sections))


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
