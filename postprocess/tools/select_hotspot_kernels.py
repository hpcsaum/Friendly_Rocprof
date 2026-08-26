#!/usr/bin/env python3
"""Resolve GPU hotspot kernel names into rocprof-compute "-k" input.

Read a hotspots report (written by extract_GPU_hotspots.py or extract_hotspots.py), a rocprofv3
output directory, or a rocprof-sys trace directory, and print one kernel name per line -- ready
to feed into AMD's rocprof-compute as "-k" (kernel) filter arguments.

Much simpler than select_instrumented_functions.py's CPU-side equivalent: rocprof-compute's "-k"
takes plain substrings, not a regex, so no escaping is needed here. There's also no
rocprof-sys-instrument-style rewrite step and no ground-truth "instrumented.json" to check
requested kernels against, so this module has no "--check-instrumented" mode either.

By default, a kernel dispatched only once in the whole run is excluded: scripts/
profile_hotspot_kernels.sh profiles each selected kernel's *second* call only (skipping the
unrepresentative first-touch/page-fault-affected first call), so a kernel with no second call
has nothing for that to target. Pass require_multiple_calls=False (the launcher's
--all-dispatches) to include those too.

Functions: labels_from_output_dir(), labels_from_report(), labels_from_trace_dir(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofv3 import aggregate
from stage4_rocprofsys_trace_ranks import discover_ranks
from stage4_rocprofsys_trace_flat import aggregate_gpu_kernels
from stage5_table_render import iter_table_rows, select_entries
import stage6_cli_common
import stage6_time_range_config

HELP_BLURB = """\
Turns a profile_hotspot_kernels.sh (or extract_GPU_hotspots.py/extract_hotspots.py) report, a
rocprofv3 output directory, or a rocprof-sys trace directory into a list of GPU hotspot kernel
names, ready to feed into AMD's rocprof-compute as "-k" (kernel filter) arguments. This is what
scripts/profile_hotspot_kernels.sh and scripts/profile_traced_hotspot_kernels.sh use to profile
only the biggest kernels in detail, instead of every kernel the application launches.

By default, a kernel that only ran once is left out -- there's no second call left to profile
once the first (unrepresentative, first-touch-affected) call is skipped. Pass --all-dispatches
to include those too.

--time-range only has an effect with --trace-dir (--report/--output-dir sources don't carry raw
timestamps to filter).

Under the hood, this prepares input for AMD's rocprof-compute -- see
https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
for details.
"""


def labels_from_output_dir(rocprofv3_dir, top=None, threshold=None, show_all=False,
                            require_multiple_calls=True):
    entries, scanned, total_ns = aggregate(rocprofv3_dir)
    if not scanned:
        raise SystemExit(
            f"error: no rocprofv3 kernel_stats.csv found in {rocprofv3_dir!r} -- "
            "nothing to select hotspot kernels from"
        )
    selected, _desc = select_entries(
        entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
    )
    if require_multiple_calls:
        selected = [e for e in selected if e["count"] >= 2]
    return sorted({e["label"] for e in selected})


def labels_from_report(report_path, require_multiple_calls=True):
    """Reads the 'GPU kernel hotspots' table from a report written by extract_GPU_hotspots.py
    or extract_hotspots.py -- both render it via the exact same GPU_HOTSPOTS_COLUMNS layout, so
    one parser covers both. Rows are taken as-is: whatever selection produced the report is
    trusted, except for the require_multiple_calls filter applied here from the 'calls' column."""
    with open(report_path, errors="replace") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "GPU kernel hotspots" in line:
            start = i
            break
    if start is None:
        raise SystemExit(
            f"error: no 'GPU kernel hotspots' section found in {report_path!r} -- "
            "expected a report written by extract_GPU_hotspots.py or extract_hotspots.py"
        )

    header = None
    for i in range(start, len(lines)):
        if lines[i].rstrip().endswith("kernel"):
            header = i
            break
    if header is None:
        raise SystemExit(
            f"error: found a 'GPU kernel hotspots' section in {report_path!r} but no table header after it"
        )

    labels = []
    # GPU_HOTSPOTS_COLUMNS: # total(s) %total calls avg(us) kernel -- 6 columns, the kernel name
    # (which may itself contain spaces, e.g. a templated C++ signature, and may itself span 2+
    # physical lines if wrap_trailing_label() hard-wrapped it) intact as the 6th and last token.
    for row in iter_table_rows(lines[header + 1:], num_columns=6):
        if len(row) < 6:
            continue
        try:
            calls = int(row[3])
        except ValueError:
            continue
        if require_multiple_calls and calls < 2:
            continue
        labels.append(row[5].strip())

    return sorted(set(labels))


def labels_from_trace_dir(trace_dir, top=None, threshold=None, show_all=False,
                           require_multiple_calls=True, cache_dir=None):
    """Reads GPU hotspot kernel names straight from a rocprof-sys trace directory's converted
    CSVs (see convert_trace_to_csv.py), independent of any rendered report -- needed because
    extract_trace_hotspots.py's own report format (a single CPU+GPU-fused table) isn't the
    GPU_HOTSPOTS_COLUMNS layout labels_from_report() parses. Respects whatever time range is
    currently configured via stage6_time_range_config (see its own module docstring) -- callers
    that want the whole trace should leave that unconfigured."""
    stage6_cli_common.require_directory(trace_dir)
    rank_inputs = discover_ranks(trace_dir)
    entries, _total_kernel_time = aggregate_gpu_kernels(rank_inputs, cache_dir=cache_dir)
    selected, _desc = select_entries(
        entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total GPU kernel time",
    )
    if require_multiple_calls:
        selected = [e for e in selected if e["count"] >= 2]
    return sorted({e["label"] for e in selected})


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="select_hotspot_kernels.py",
        description=HELP_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--report", dest="report", default=None,
                         help="hotspots.txt report to read GPU hotspot kernel names from")
    source.add_argument("--output-dir", dest="output_dir", default=None,
                         help="rocprofv3 output directory to read GPU hotspot kernel names from")
    source.add_argument("--trace-dir", dest="trace_dir", default=None,
                         help="rocprof-sys trace directory (converted CSVs) to read GPU hotspot "
                              "kernel names from")
    stage6_cli_common.add_selection_args(parser, "kernels", "of total device time",
                                          top_noun="hotspot kernels", top_help_suffix=" (default: 20)",
                                          verb="select")
    parser.add_argument("--all-dispatches", dest="all_dispatches", action="store_true", default=False,
                         help="also include kernels dispatched only once (excluded by default, "
                              "since the launcher profiles each kernel's 2nd call only)")
    stage6_time_range_config.add_cli_argument(parser)
    args = parser.parse_args(argv)
    stage6_time_range_config.configure_from_args(args)

    if not args.report and not args.output_dir and not args.trace_dir:
        raise SystemExit("error: one of --report, --output-dir, or --trace-dir is required")

    require_multiple_calls = not args.all_dispatches

    if args.report:
        labels = labels_from_report(args.report, require_multiple_calls=require_multiple_calls)
    elif args.trace_dir:
        labels = labels_from_trace_dir(
            args.trace_dir, top=args.top, threshold=args.threshold, show_all=args.show_all,
            require_multiple_calls=require_multiple_calls,
        )
    else:
        stage6_cli_common.require_directory(args.output_dir)
        labels = labels_from_output_dir(
            args.output_dir, top=args.top, threshold=args.threshold, show_all=args.show_all,
            require_multiple_calls=require_multiple_calls,
        )

    # Deliberately not an error if labels ends up empty here: scripts/profile_hotspot_kernels.sh
    # checks that itself and prints a more specific message (mentioning --all-dispatches).
    for label in labels:
        print(label)


if __name__ == "__main__":
    main()
