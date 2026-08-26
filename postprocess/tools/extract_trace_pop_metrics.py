#!/usr/bin/env python3
"""Compute POP-inspired parallel-efficiency metrics from one or more rocprof-sys Perfetto
trace-CSV exports.

See docs/pop_metrics_reference.md for the full metric hierarchy and which POP metrics are/aren't
derivable from this data. In short: Load Balance, Communication Efficiency, and Parallel
Efficiency come from a single run; Computation Efficiency and Global Efficiency need a scaling
study (2+ runs, compared against the first as reference). Serialisation/Transfer Efficiency (need
Dimemas) and Instruction/IPC Scaling (need PAPI hardware counters) are NOT computed here -- see
that doc for why.

Functions: write_report(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofsys_trace_flat import gather_timing_summary_per_rank
from stage4_rocprofsys_trace_ranks import discover_ranks
from stage5_pop_metrics_table import compute_metrics_from_per_rank, format_metrics_table, metrics_legend, run_label
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file
import stage6_time_range_config

SHORT_DESCRIPTION = (
    "Computes POP-inspired parallel efficiency metrics (Load Balance, Communication/Parallel/\n"
    "Computation/Global Efficiency) from one or more Perfetto trace-CSV exports.\n"
)

HELP_BLURB = """\
Reads one or more rocprof-sys trace-mode runs, already converted to the flat
trace-CSV files this project's tools expect (see the project README for the
conversion step), and computes POP-inspired parallel efficiency metrics: Load
Balance, Communication Efficiency, and Parallel Efficiency from a single run;
Computation Efficiency and Global Efficiency when 2+ runs from a scaling
study (e.g. different -np counts) are given, compared against the first
directory as the reference.

This does NOT compute every POP metric -- Serialisation/Transfer Efficiency
need a Dimemas-style network simulation, and Instruction/IPC Scaling need
PAPI hardware counters; neither is available from rocprof-sys's own output.
See docs/pop_metrics_reference.md for the full picture.

Communication and GPU time are classified by a trace's own exact, per-event
category (not a function-name guess the way the text-table-based
extract_pop_metrics.py has to) -- and GPU visibility is always part of the
same trace, so the GPU-specific columns (GPU-Util/GPU-Off/GPU-LB, and
GPU-Eff for a scaling study) always appear here, unlike that tool's
paired-rocprofv3 case where they only show up when GPU data happens to be
paired in.

Under the hood, this reads a CSV export of a Perfetto trace produced by
AMD's rocprof-sys running in trace mode (ROCPROFSYS_TRACE=1) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for
rocprof-sys, and https://perfetto.dev/ for the trace format itself.
"""


def _compute_trace_run_metrics(trace_dir):
    """One trace directory's full metrics dict, shaped like
    stage5_pop_metrics_table.compute_run_metrics()'s own return minus "cpu_dir"/"gpu_dir" -- a
    trace has no such pairing concept, always both domains from one source."""
    rank_inputs = discover_ranks(trace_dir)
    per_rank = gather_timing_summary_per_rank(rank_inputs)
    metrics = compute_metrics_from_per_rank(per_rank)
    return {
        "run_dir": trace_dir,
        "num_ranks": len(per_rank),
        "per_rank": per_rank,
        "rank_inputs": rank_inputs,
        **metrics,
    }


def write_report(run_dirs, dest_path, scaling=None, command_line=""):
    all_metrics = [_compute_trace_run_metrics(d) for d in run_dirs]
    multi_run = len(all_metrics) > 1

    runs = []
    for i, m in enumerate(all_metrics):
        label = "reference run" if i == 0 else f"scaling run {i + 1} ({run_label(m['run_dir'])})"
        runs.append({
            "directories": [(label, m["run_dir"])], "num_ranks": m["num_ranks"],
            "extra_lines": [
                "  pool: CPU+GPU, single unified trace source\n",
                stage6_time_range_config.describe_time_range(m["rank_inputs"]),
            ],
        })
    header = standard_header("extract_trace_pop_metrics.py", SHORT_DESCRIPTION, runs)

    table_text, show_gpu_cols, show_gpu_eff = format_metrics_table(all_metrics, scaling)
    sections = [(None, table_text)]

    footer = (
        metrics_legend(show_gpu_cols, show_gpu_eff, multi_run, scaling)
        + "\n"
        + help_redirect("what's not computed and classification caveats", script_name="extract_trace_pop_metrics.py")
        + command_line
    )

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir", help="reference run's trace-CSV directory")
    parser.add_argument("scaled_dirs", nargs="*", help="additional runs from the same scaling study, compared against reference_dir")
    parser.add_argument("--scaling", choices=["strong", "weak"], default=None,
                         help="required when scaled_dirs is non-empty: 'strong' (fixed global problem "
                              "size) or 'weak' (fixed problem size per rank)")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the report (default: <reference_dir>/pop_metrics.txt)")
    stage6_noise_config.add_cli_argument(parser)
    stage6_time_range_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    run_dirs = [args.reference_dir] + args.scaled_dirs
    stage6_cli_common.require_directories(run_dirs)

    if args.scaled_dirs and args.scaling is None:
        raise SystemExit("error: --scaling {strong,weak} is required when scaled_dirs are given")

    stage6_noise_config.configure_from_args(args)
    stage6_time_range_config.configure_from_args(args)

    dest = stage6_cli_common.resolve_dest(args.dest, args.reference_dir, "pop_metrics.txt")
    tokens = [os.path.abspath(d) for d in run_dirs]
    if args.scaling:
        tokens += ["--scaling", args.scaling]
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    if args.time_range:
        tokens += ["--time-range", args.time_range]
    command_line = command_header(sys.argv[0], tokens)

    write_report(run_dirs, dest, scaling=args.scaling, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
