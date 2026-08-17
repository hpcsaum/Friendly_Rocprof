#!/usr/bin/env python3
"""Extract a short GPU kernel hotspots report from rocprofv3 kernel_stats.csv output.

Only reads *_kernel_stats.csv files rocprofv3 writes with --kernel-trace --stats
--output-format csv. This is REAL device kernel execution time, unlike
extract_CPU_hotspots.py's host-side timing -- see extract_hotspots.py to combine
both into one report.

Functions: gather_run_info(), write_report(), main().
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofv3 import aggregate, aggregate_per_rank
from stage5_gpu_hotspots_table import GPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, imbalance_note, load_imbalance_columns
from stage5_table_render import pct_total_note, render_table, select_entries
import stage6_cli_common
from stage6_report_builder import command_header, render_report, standard_header, write_report_file
from stage6_run_metadata import guess_executable, guess_num_ranks, guess_run_datetime, guess_total_runtime, load_json_file

CONFIG_EXECUTABLE_KEYS = ["command", "command_line", "argv", "cmd", "exe", "executable"]
CONFIG_DATETIME_KEYS = ["init_time", "start_time", "launch_time", "timestamp"]
CONFIG_RUNTIME_KEYS = ["elapsed", "duration", "wall_time", "total_time", "runtime"]

PID_SUFFIX_RE = re.compile(r"(\d+)_kernel_stats\.csv$")

SHORT_DESCRIPTION = "Ranks real GPU kernel execution time from rocprofv3 kernel-trace data.\n"

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


def write_report(output_dir, dest_path, top=None, threshold=None, show_all=False, command_line=""):
    entries, scanned_files, total_ns = aggregate(output_dir)
    if not scanned_files:
        raise SystemExit(
            f"error: no rocprofv3 kernel_stats.csv found under {output_dir!r} "
            "(expected files like <pid>_kernel_stats.csv, from "
            "'rocprofv3 --kernel-trace --stats --output-format csv') -- nothing to report"
        )

    run_info = gather_run_info(output_dir, scanned_files)
    threshold_unit = "of total measured GPU time (summed across all scanned files)"
    selected, desc = select_entries(
        entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=threshold_unit,
    )

    per_file_totals, imbalance_scanned = aggregate_per_rank(output_dir)
    if len(imbalance_scanned) < 2:
        imbalance_title = (
            f"GPU kernel load imbalance across ranks -- skipped: only {len(imbalance_scanned)} "
            "rank/file found, need at least 2 to compare\n"
        )
        imbalance_body = ""
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_file_totals, top, threshold, show_all)
        imbalance_title = f"GPU kernel load imbalance across {len(imbalance_scanned)} ranks -- showing {imbalance_desc}\n"
        imbalance_body = (
            render_table(load_imbalance_columns(item_label="kernel"), imbalance_selected) + "\n"
            + imbalance_note("kernel", "total")
        )

    header = standard_header("extract_GPU_hotspots.py", SHORT_DESCRIPTION, [{
        "directories": [("source directory", output_dir)], "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": run_info["num_ranks"], "scanned_files": scanned_files,
    }])
    sections = [
        (f"GPU kernel hotspots -- showing {desc}\n",
         render_table(GPU_HOTSPOTS_COLUMNS, selected) + "\n"
         + pct_total_note("kernel", threshold_unit)
         + "  - This won't add up to 100% if a --threshold/--top cut entries.\n"),
        (imbalance_title, imbalance_body),
    ]
    footer = (
        "For host-side (HIP API / launch overhead) hotspots, use scripts/profile_CPU_hotspots.sh.\n"
        + command_line
    )

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprofv3 output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspots report (default: <output_dir>/hotspots.txt)")
    stage6_cli_common.add_selection_args(parser, "kernels", "of total GPU time",
                                          top_noun="hotspots", top_help_suffix=" (default: 20)")
    args = parser.parse_args(argv)

    stage6_cli_common.require_directory(args.output_dir)

    dest = stage6_cli_common.resolve_dest(args.dest, args.output_dir, "hotspots.txt")
    tokens = [os.path.abspath(args.output_dir)]
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.top is not None:
        tokens += ["--top", str(args.top)]
    elif args.threshold is not None:
        tokens += ["--threshold", str(args.threshold)]
    elif args.show_all:
        tokens += ["--all"]
    command_line = command_header(sys.argv[0], tokens)

    write_report(args.output_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
