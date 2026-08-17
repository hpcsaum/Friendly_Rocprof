#!/usr/bin/env python3
"""Render an indented call tree from rocprof-sys sampled-stack text output,
optionally with rocprofv3 GPU kernel data nested in at the CPU call site(s)
that launched them.

This is the sampling-based variant: it prefers sampling_wall_clock-<pid>.txt
(a real unwound stack sample at every tick, regardless of what rocprof-sys
happened to instrument) per rank, falling back to wall_clock-<pid>.txt only
for a rank with no sampling file at all. Unlike extract_calltree_traced.py
(the wall_clock-preferred tool), this shows the TRUE call depth -- a real
intermediate frame that was never GOTCHA-instrumented still shows up here --
at the cost of two tradeoffs: (1) sampling's own timing is only statistically
approximate, and (2) the raw sampled stack is dominated by rocprof-sys's own
instrumentation/dynamic-linker machinery, the C/Fortran runtime's allocator
internals, and deep MPI-implementation internals, none of which are useful to
see by default -- see the four --show-* flags below.

Unlike extract_CPU_hotspots.py's scan_ranks()/aggregate(), this does NOT merge
same-label rows within one rank's own file -- a calltree needs every
individual call-tree node kept distinct (attach_ancestry()'s parent links
intact), not summed by label. It DOES merge structurally ACROSS ranks (see
stage4_rocprofsys_tree.merge_rank_trees()) into one aggregated tree -- a global view,
not one call tree per rank -- with each node's CALLS/TOTAL-AVG(s) columns
averaged, and SELF given a full avg/std_dev/min/max load-balance
breakdown, the same convention extract_CPU_hotspots.compute_load_imbalance()
already uses elsewhere in this codebase. Kernel placement (see stage5_calltree_view.py)
is never per-dispatch-exact: this toolchain's text/JSON output has no per-call
timestamps to correlate against, only the binary Perfetto trace does, and
there's no stdlib-friendly way to parse it -- a future capability, not
silently dropped.

Functions: write_report(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
from stage1_run_dirs import resolve_two_dirs
from stage5_calltree_view import build_calltree_view
from stage5_tree_render import aggregation_note, tree_view_note
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Renders an aggregated call tree (sampling-based, true call depth) across every rank,\n"
    "optionally with GPU kernel data nested in.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys output directory (optionally paired with rocprofv3, either
as a nested rocprofv3/ subdir or as a second, separately-run directory) and
writes an indented call tree -- actual function nesting, not a flat ranked
list -- aggregated into one global view across every rank (not one tree per
rank): each line's CALLS/TOTAL-AVG(s) are averaged, and SELF gets a full
avg/std_dev/min/max load-balance breakdown, the same convention this
codebase's other load-imbalance tables already use. Works on tool 1/3 output
(profile_CPU_hotspots.sh/profile_hotspots.sh) and tool 4's scan directory
(instrument_hotspots.sh trace) identically -- same underlying data format
either way.

This is the sampling-based variant: it shows the TRUE call depth (every real
stack frame at each sample tick), not just what rocprof-sys happened to
instrument -- see extract_calltree_traced.py for the faster, exact-where-
instrumented alternative if you don't need that. The tradeoff is that
sampling's own timing is only statistically approximate, and the raw sampled
stack is dominated by noise: rocprof-sys's own instrumentation/dynamic-linker
machinery, GPU/offload-runtime internals, the compiler runtime's allocator
internals, and deep MPI-implementation internals. Four independent flags
reveal each of these, hidden by default: --show-gpu-api,
--show-rocprofsys-internals, --show-mpi-internals, --show-compiler-runtime
(or --show-all-internals for all four at once). Wrapper frames are spliced
out (children reparented, not deleted); MPI internals are collapsed (the
first real MPI frame is shown, its own internals are not); GPU-API and
compiler-runtime noise are pruned (whole subtree hidden). The compiler-
runtime tier currently recognizes Cray's Fortran runtime allocator/intrinsic
helpers only -- other compilers' runtime noise is not filtered.

When a paired rocprofv3 directory is found, real GPU kernel data is nested
into the tree at the CPU subroutine that actually contains it -- Cray's
OpenACC/HIP-offload kernel naming embeds that subroutine's name directly in
the kernel name, so this is a precise match, not a guess, whenever that
subroutine shows up as its own node in the merged tree. A kernel that can't
be matched by name falls back to a structural estimate instead (nearest
launch-call ancestor, proportionally split by launch-call count when there's
more than one candidate site) -- neither approach is per-dispatch-exact:
this toolchain's text/JSON output has no per-call timestamps to correlate
against, only the binary Perfetto trace does, and that has no
stdlib-friendly Python parser (a future tool, not attempted here).

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def write_report(cpu_dir, gpu_dir, dest_path, max_depth=None, show_gpu_api=False,
                  show_rocprofsys_internals=False, show_mpi_internals=False,
                  show_compiler_runtime=False, command_line=""):
    view = build_calltree_view(
        cpu_dir, cpu_dir, gpu_dir, max_depth=max_depth, show_gpu_api=show_gpu_api,
        show_rocprofsys_internals=show_rocprofsys_internals, show_mpi_internals=show_mpi_internals,
        show_compiler_runtime=show_compiler_runtime,
    )
    rank_keys = view["rank_keys"]

    # One shared identity for the run being profiled -- CPU's own metadata is preferred; GPU's
    # (when a GPU directory was actually given) is used only for whichever field CPU didn't have.
    # MPI ranks comes from the tree merge itself (len(rank_keys)), ground truth rather than a guess.
    run_info = cpu_tool.gather_run_info(cpu_dir, [])
    directories = [("CPU run directory", cpu_dir)]
    if gpu_dir is not None:
        directories.append(("GPU run directory", gpu_dir))
        gpu_run_info = gpu_tool.gather_run_info(gpu_dir, [])
        run_info = {
            key: run_info[key] if run_info[key] is not None else gpu_run_info[key]
            for key in ("executable", "run_datetime", "total_runtime")
        }

    header = standard_header("extract_calltree.py", SHORT_DESCRIPTION, [{
        "directories": directories, "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": len(rank_keys),
    }])
    tree_notes = aggregation_note() + tree_view_note(
        rank_keys, max_depth, show_gpu_api, show_rocprofsys_internals, show_mpi_internals,
        show_compiler_runtime,
    )
    sections = [(None, view["tree_text"] + "\n" + tree_notes)]
    if view["fallback_text"]:
        sections.append((None, view["fallback_text"]))

    footer = help_redirect(
        "sampling behavior, noise filtering, and kernel-placement caveats",
        script_name="extract_calltree.py",
    ) + command_line

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys output directory to read, or a combined "
                                            "run directory containing both a rocprof-sys/ and a "
                                            "rocprofv3/ subdir")
    parser.add_argument("gpu_dir", nargs="?", default=None,
                         help="rocprofv3 output directory (GPU side) -- omit when output_dir "
                              "already contains both subdirs, or when there's no GPU data to pair")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the call tree report (default: <output_dir>/calltree.txt)")
    stage6_cli_common.add_max_depth_arg(parser)
    stage6_cli_common.add_noise_tier_args(
        parser, ["gpu_api", "rocprofsys_internals", "mpi_internals", "compiler_runtime"],
        all_shorthand=True,
    )
    stage6_noise_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    cpu_dir, gpu_dir = resolve_two_dirs(args.output_dir, args.gpu_dir)
    stage6_cli_common.require_directories([cpu_dir, gpu_dir])

    stage6_noise_config.configure_from_args(args)

    dest = stage6_cli_common.resolve_dest(args.dest, args.output_dir, "calltree.txt")
    tokens = [os.path.abspath(args.output_dir)]
    if args.gpu_dir:
        tokens.append(os.path.abspath(args.gpu_dir))
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
        cpu_dir, gpu_dir, dest, max_depth=args.max_depth,
        show_gpu_api=args.show_gpu_api or args.show_all_internals,
        show_rocprofsys_internals=args.show_rocprofsys_internals or args.show_all_internals,
        show_mpi_internals=args.show_mpi_internals or args.show_all_internals,
        show_compiler_runtime=args.show_compiler_runtime or args.show_all_internals,
        command_line=command_line,
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
