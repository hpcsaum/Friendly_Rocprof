#!/usr/bin/env python3
"""Render an indented call tree from a rocprof-sys Perfetto trace-CSV export, with real GPU
kernel data nested in at the exact CPU call site that launched it.

Only reads the flat trace-CSV files this project's own convention documents (see
stage4_rocprofsys_trace_ranks.py) -- a CSV export of a rocprof-sys trace-mode run, not the raw
Perfetto `.proto` trace itself (that conversion step is convert_trace_to_csv.py, a separate
invocation from this tool).

Unlike extract_calltree.py/extract_wallclock_calltree.py, GPU kernel placement here is exact, not
a name-match-then-structural-guess: a trace's `corr_id` links each kernel dispatch to the exact
host-side launch call that issued it, so there's no "no anchor found" fallback section at all --
every kernel dispatch is always placed exactly, or (if its corr_id genuinely doesn't resolve)
left visible as its own root rather than silently hidden.

Functions: write_report(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofsys_trace_ranks import discover_ranks
from stage5_trace_calltree_view import build_calltree_view
from stage5_tree_render import aggregation_note, tree_view_note
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Renders an aggregated call tree from a rocprof-sys Perfetto trace-CSV export, across\n"
    "every rank, with GPU kernel data nested in at its exact launch site.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys trace-mode run, already converted to the flat trace-CSV
files this project's tools expect (see the project README for the
conversion step), and writes an indented call tree -- actual function
nesting, not a flat ranked list -- aggregated into one global view across
every rank (not one tree per rank): each line's CALLS/TOTAL-AVG(s) are
averaged, and SELF gets a full avg/std_dev/min/max load-balance breakdown,
the same convention this codebase's other load-imbalance tables already use.

A trace records real per-event categories, so noise classification here is
exact, not guessed from a label string. Four independent flags reveal what's
hidden by default: --show-gpu-api, --show-rocprofsys-internals,
--show-mpi-internals, --show-compiler-runtime (or --show-all-internals for
all four at once). Wrapper frames are spliced out (children reparented, not
deleted); MPI internals are collapsed (the first real MPI frame is shown,
its own internals are not); GPU-API and compiler-runtime noise are pruned
(whole subtree hidden). Real GPU kernel execution and memory-copy time are
always shown -- they're actual device work, not noise, so there's no flag to
hide them.

GPU kernel dispatches are nested into the tree at the exact CPU call site
that launched them -- a trace's `corr_id` links each dispatch to its host
launch call directly, so this is never a guess the way it is for a text-table
profile (see extract_calltree.py) -- there is no "couldn't place this kernel"
fallback section here at all. When that exact call site turns out to be a
generic runtime entry point shared by every kernel launch in the whole
program (common under OMPT-based `omp target` instrumentation), a kernel
whose name embeds its owning function or subroutine (Cray Fortran's `$ck_`
marker, or the LLVM OpenMP-offloading kernel name shape every other
compiler this project targets uses) is instead reanchored onto the exact CPU
call instance -- found via that owning name and the trace's own timestamps,
not a guess -- that was actually running on the launching thread immediately
before the kernel dispatched. This is exact, not an estimate, but relies on
that one CPU thread's own recorded call frames never overlapping in time
(true for a normal call stack). A kernel whose owning name doesn't appear on
its launching thread at all, or whose name matches neither convention, still
shows up at its exact `corr_id` position, unchanged.

More generally, any CPU-side call -- not just a GPU kernel dispatch --
nests under the nearest ancestor this trace actually captured a frame for,
never a guess beyond that. rocprof-sys traces only the functions selected
for instrumentation (see the project README's hotspot-selection step), so a
call made from inside an uninstrumented function shows up nested directly
under whichever instrumented (or OMPT-internal, e.g. `ompt_implicit_task`)
frame was still open on that thread at the time -- which can look like it
was "called by" that frame even though it wasn't. Unlike the kernel case
above, there's no name or id to recover the true immediate caller from when
this happens; the fix is adding that function to the instrumentation
selection and re-profiling, not something this tool can reconstruct from
the data it's given.

Under the hood, this reads a CSV export of a Perfetto trace produced by
AMD's rocprof-sys running in trace mode (ROCPROFSYS_TRACE=1) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for
rocprof-sys, and https://perfetto.dev/ for the trace format itself.
"""


def write_report(trace_dir, dest_path, max_depth=None, show_gpu_api=False,
                  show_rocprofsys_internals=False, show_mpi_internals=False,
                  show_compiler_runtime=False, command_line=""):
    rank_inputs = discover_ranks(trace_dir)
    view = build_calltree_view(
        rank_inputs, max_depth=max_depth, show_gpu_api=show_gpu_api,
        show_rocprofsys_internals=show_rocprofsys_internals, show_mpi_internals=show_mpi_internals,
        show_compiler_runtime=show_compiler_runtime,
    )
    rank_keys = view["rank_keys"]

    header = standard_header("extract_trace_calltree.py", SHORT_DESCRIPTION, [{
        "directories": [("trace directory", trace_dir)], "num_ranks": len(rank_keys),
    }])
    tree_notes = aggregation_note() + tree_view_note(
        rank_keys, max_depth, show_gpu_api, show_rocprofsys_internals, show_mpi_internals,
        show_compiler_runtime,
    )
    sections = [(None, view["tree_text"] + "\n" + tree_notes)]

    footer = help_redirect(
        "noise filtering and kernel/call-placement caveats",
        script_name="extract_trace_calltree.py",
    ) + command_line

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", help="directory of trace-CSV files to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the call tree report (default: <trace_dir>/calltree.txt)")
    stage6_cli_common.add_max_depth_arg(parser)
    stage6_cli_common.add_noise_tier_args(
        parser, ["gpu_api", "rocprofsys_internals", "mpi_internals", "compiler_runtime"],
        all_shorthand=True,
    )
    stage6_noise_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    stage6_cli_common.require_directory(args.trace_dir)
    stage6_noise_config.configure_from_args(args)

    dest = stage6_cli_common.resolve_dest(args.dest, args.trace_dir, "calltree.txt")
    tokens = [os.path.abspath(args.trace_dir)]
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.max_depth is not None:
        tokens += ["--max-depth", str(args.max_depth)]
    if args.show_all_internals:
        tokens += ["--show-all-internals"]
    else:
        if args.show_gpu_api:
            tokens.append("--show-gpu-api")
        if args.show_rocprofsys_internals:
            tokens.append("--show-rocprofsys-internals")
        if args.show_mpi_internals:
            tokens.append("--show-mpi-internals")
        if args.show_compiler_runtime:
            tokens.append("--show-compiler-runtime")
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    command_line = command_header(sys.argv[0], tokens)

    write_report(
        args.trace_dir, dest, max_depth=args.max_depth,
        show_gpu_api=args.show_gpu_api or args.show_all_internals,
        show_rocprofsys_internals=args.show_rocprofsys_internals or args.show_all_internals,
        show_mpi_internals=args.show_mpi_internals or args.show_all_internals,
        show_compiler_runtime=args.show_compiler_runtime or args.show_all_internals,
        command_line=command_line,
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
