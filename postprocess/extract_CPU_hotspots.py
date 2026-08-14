#!/usr/bin/env python3
"""Extract a short CPU-side hotspots report from rocprof-sys timemory text output.

Only reads the well-documented pipe-delimited "timemory" text tables
(e.g. wall_clock-<pid>.txt) that rocprof-sys writes for CPU-side timing.
GPU device kernel execution time is NOT present in this data -- see the
footer note this script writes into its own output.
"""

import argparse
import glob
import os
import re
import sys

from stage1_rocprofsys import PID_SUFFIX_RE
from stage4_rocprofsys_flat import aggregate, aggregate_per_rank
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, imbalance_note, load_imbalance_columns
from stage5_table_render import pct_total_note, ranking_note, render_table, select_entries
from stage6_report_builder import command_header, render_report, standard_header, write_report_file
from stage6_run_metadata import guess_executable, guess_num_ranks, guess_run_datetime, guess_total_runtime, load_json_file

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

SHORT_DESCRIPTION = (
    "Ranks CPU-side hotspots from rocprof-sys timemory data -- functions worth offloading to\n"
    "the GPU, or worth optimizing on the CPU itself.\n"
)

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


def gather_run_info(output_dir, scanned_files):
    data = load_json_file(output_dir, METADATA_FILENAME)
    return {
        "executable": guess_executable(data, EXECUTABLE_KEYS),
        "run_datetime": guess_run_datetime(data, RUN_DATETIME_KEYS, output_dir=output_dir,
                                            scanned_files=scanned_files, dir_pattern=TIME_OUTPUT_DIR_RE),
        "total_runtime": guess_total_runtime(data, TOTAL_RUNTIME_KEYS),
        "num_ranks": guess_num_ranks(data, PID_SUFFIX_RE, scanned_files, keys=NUM_RANKS_KEYS),
    }


def write_report(output_dir, dest_path, top=None, threshold=None, show_all=False, unfiltered=False,
                  command_line=""):
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
    threshold_unit = "of total measured time (summed across all scanned files)"

    def _set_pct_total(entries):
        for e in entries:
            e["pct_total"] = (e[key_field] / total_runtime * 100.0) if total_runtime > 0 else None

    cpu_selected, cpu_desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=threshold_unit, prepare=_set_pct_total,
    )
    gpu_selected, gpu_desc = select_entries(
        gpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=threshold_unit, prepare=_set_pct_total,
    )

    per_file_totals, imbalance_scanned = aggregate_per_rank(output_dir, unfiltered=unfiltered)
    if len(imbalance_scanned) < 2:
        imbalance_title = (
            f"CPU load imbalance across ranks -- skipped: only {len(imbalance_scanned)} "
            "rank/file found, need at least 2 to compare\n"
        )
        imbalance_body = ""
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_file_totals, top, threshold, show_all)
        imbalance_title = f"CPU load imbalance across {len(imbalance_scanned)} ranks -- showing {imbalance_desc}\n"
        imbalance_body = (
            render_table(load_imbalance_columns(), imbalance_selected) + "\n"
            + imbalance_note("function", "inclusive" if unfiltered else "self")
        )

    header = standard_header("extract_CPU_hotspots.py", SHORT_DESCRIPTION, [{
        "directories": [("source directory", output_dir)], "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": run_info["num_ranks"], "scanned_files": scanned_files,
    }])
    sections = [
        (f"CPU compute hotspots (candidates for GPU offload) -- showing {cpu_desc}\n",
         render_table(CPU_HOTSPOTS_COLUMNS, cpu_selected) + "\n"
         + ranking_note(unfiltered)
         + pct_total_note("function", threshold_unit)
         + "  - This won't add up to 100% across both this table and the GPU API table below "
           "-- they share one denominator.\n"),
        (f"GPU API / launch overhead -- showing {gpu_desc}\n",
         render_table(CPU_HOTSPOTS_COLUMNS, gpu_selected) + "\n"
         + "  - Host-side call overhead only -- NOT device kernel execution time. rocprof-sys's "
           "text/JSON output only captures host-side timing; true GPU kernel execution time is "
           "not present in this data.\n"
         + ranking_note(unfiltered)
         + pct_total_note("function", threshold_unit)),
        (imbalance_title, imbalance_body),
    ]
    footer = "".join([
        *(["A Perfetto trace was also found (not parsed by this tool):\n"]
          + [f"  - {f}\n" for f in proto_files] if proto_files else []),
        *(["A rocpd database was also found (ROCm 7.1+ only, not parsed by this tool):\n"]
          + [f"  - {f}\n" for f in db_files] if db_files else []),
        "For real GPU kernel hotspots, use scripts/profile_GPU_hotspots.sh "
        "(or run: rocprofv3 --kernel-trace --stats --output-format csv -- <app>)\n",
        command_line,
    ])

    return write_report_file(dest_path, render_report(header, sections, footer))


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
    tokens = [os.path.abspath(args.output_dir)]
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.top is not None:
        tokens += ["--top", str(args.top)]
    elif args.threshold is not None:
        tokens += ["--threshold", str(args.threshold)]
    elif args.show_all:
        tokens += ["--all"]
    if args.unfiltered:
        tokens += ["--unfiltered"]
    command_line = command_header(sys.argv[0], tokens)

    write_report(args.output_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 unfiltered=args.unfiltered, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
