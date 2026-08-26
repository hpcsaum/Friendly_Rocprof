#!/usr/bin/env python3
"""Extract a combined CPU+GPU hotspots report from a rocprof-sys Perfetto trace-CSV export.

Only reads the flat trace-CSV files this project's own convention documents (see
stage4_rocprofsys_trace_ranks.py) -- a CSV export of a rocprof-sys trace-mode run, not the raw
Perfetto `.proto` trace itself (that conversion step is convert_trace_to_csv.py, a separate
invocation from this tool).

Functions: write_report(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofsys_trace_flat import aggregate, aggregate_per_rank
from stage4_rocprofsys_trace_ranks import discover_ranks
from stage5_fused_hotspots_table import FUSED_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, imbalance_note, load_imbalance_columns
from stage5_table_render import pct_total_note, ranking_note, render_table, select_entries
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, render_report, standard_header, write_report_file
import stage6_time_range_config

SHORT_DESCRIPTION = (
    "Ranks combined CPU+GPU hotspots from a rocprof-sys Perfetto trace-CSV export -- one\n"
    "table, since a trace already ties both domains to the same timeline.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys trace-mode run, already converted to the flat trace-CSV
files this project's tools expect (see the project README for the
conversion step), and writes a short, ranked text report: which functions
and GPU kernels spend the most time, across all ranks.

Unlike extract_CPU_hotspots.py/extract_GPU_hotspots.py, this is ONE combined
table -- a trace already ties CPU calls and GPU kernel execution to the same
timeline via an exact host-launch-to-device-dispatch correlation, so there's
no separate collection tools to reconcile. A second section shows load
imbalance across ranks, for CPU functions and GPU kernels alike.

Numbers are percentages of total measured time -- good enough to spot your
top bottleneck, not a precise, reproducible benchmark.

Under the hood, this reads a CSV export of a Perfetto trace produced by
AMD's rocprof-sys running in trace mode (ROCPROFSYS_TRACE=1) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for
rocprof-sys, and https://perfetto.dev/ for the trace format itself.
"""


def write_report(trace_dir, dest_path, top=None, threshold=None, show_all=False, unfiltered=False,
                  command_line=""):
    rank_inputs = discover_ranks(trace_dir)
    entries, total_runtime = aggregate(rank_inputs)

    key_field = "sum" if unfiltered else "self_sum"
    threshold_unit = "of total measured time (summed across all ranks)"

    def _set_pct_total(entries):
        for e in entries:
            e["pct_total"] = (e[key_field] / total_runtime * 100.0) if total_runtime > 0 else None

    selected, desc = select_entries(
        entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=threshold_unit, prepare=_set_pct_total,
    )

    per_rank_totals, rank_keys = aggregate_per_rank(rank_inputs, unfiltered=unfiltered)
    if len(rank_keys) < 2:
        imbalance_title = (
            f"Load imbalance across ranks -- skipped: only {len(rank_keys)} rank(s) found, need "
            "at least 2 to compare\n"
        )
        imbalance_body = ""
    else:
        imbalance_selected, imbalance_desc = compute_load_imbalance(per_rank_totals, top, threshold, show_all)
        imbalance_title = f"Load imbalance across {len(rank_keys)} ranks -- showing {imbalance_desc}\n"
        imbalance_body = (
            render_table(load_imbalance_columns(), imbalance_selected) + "\n"
            + imbalance_note("function", "inclusive" if unfiltered else "self")
        )

    header = standard_header("extract_trace_hotspots.py", SHORT_DESCRIPTION, [{
        "directories": [("trace directory", trace_dir)], "num_ranks": len(rank_keys),
        "extra_lines": [stage6_time_range_config.describe_time_range(rank_inputs)],
    }])
    sections = [
        (f"CPU+GPU hotspots -- showing {desc}\n",
         render_table(FUSED_HOTSPOTS_COLUMNS, selected) + "\n"
         + ranking_note(unfiltered)
         + pct_total_note("function/kernel", threshold_unit)),
        (imbalance_title, imbalance_body),
    ]
    footer = command_line

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", help="directory of trace-CSV files to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspots report (default: <trace_dir>/hotspots.txt)")
    stage6_cli_common.add_selection_args(parser, "entries", "of total runtime", singular_noun="entry",
                                          top_noun="hotspots", top_help_suffix=" per section (default: 20)")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="rank by inclusive (total) time instead of self time -- a function "
                              "that just calls other functions can still rank high this way")
    stage6_noise_config.add_cli_argument(parser)
    stage6_time_range_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    stage6_cli_common.require_directory(args.trace_dir)
    stage6_noise_config.configure_from_args(args)
    stage6_time_range_config.configure_from_args(args)

    dest = stage6_cli_common.resolve_dest(args.dest, args.trace_dir, "hotspots.txt")
    tokens = [os.path.abspath(args.trace_dir)]
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
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    if args.time_range:
        tokens += ["--time-range", args.time_range]
    command_line = command_header(sys.argv[0], tokens)

    write_report(args.trace_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 unfiltered=args.unfiltered, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
