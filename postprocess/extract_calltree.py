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
already uses elsewhere in this codebase. See docs/plans/1.14-sampling-calltree-tool.md
for the full original design rationale, including why per-dispatch-exact
kernel placement isn't achievable from this toolchain's text/JSON output
(only the binary Perfetto trace has per-call timestamps, and there's no
stdlib-friendly way to parse it -- deferred future work, not silently
dropped).
"""

import argparse
import os
from datetime import datetime

from stage1_run_dirs import resolve_run_dirs
from stage5_calltree_view import build_calltree_view
from stage6_report_builder import render_report, write_report_file

HELP_BLURB = """\
Reads a rocprof-sys (optionally paired with rocprofv3) output directory and
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
runtime tier was built from Cray's Fortran runtime specifically -- not
necessarily complete for other compilers, since none have been observed in
this project's data so far.

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


def write_report(run_dir, dest_path, max_depth=None, show_gpu_api=False,
                  show_rocprofsys_internals=False, show_mpi_internals=False,
                  show_compiler_runtime=False):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    view = build_calltree_view(
        run_dir, cpu_dir, gpu_dir, max_depth=max_depth, show_gpu_api=show_gpu_api,
        show_rocprofsys_internals=show_rocprofsys_internals, show_mpi_internals=show_mpi_internals,
        show_compiler_runtime=show_compiler_runtime,
    )
    rank_keys = view["rank_keys"]

    header = (
        "Call tree report (sampling-based, aggregated across ranks)\n"
        f"generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"source directory: {os.path.abspath(run_dir)}\n"
        f"CPU data: {os.path.abspath(cpu_dir)}\n"
        f"GPU data: {os.path.abspath(gpu_dir) if view['gpu_paired'] else '(none)'}\n"
        f"ranks aggregated: {len(rank_keys)} (rank keys: {', '.join(rank_keys)})\n"
        + (
            "Showing: GPU-API/runtime noise "
            + ("included" if show_gpu_api else "hidden (--show-gpu-api to reveal)") + ", "
            "rocprof-sys internals "
            + ("included" if show_rocprofsys_internals else "spliced out (--show-rocprofsys-internals to reveal)") + ", "
            "MPI internals "
            + ("included" if show_mpi_internals else "collapsed (--show-mpi-internals to reveal)") + ", "
            "compiler-runtime helpers "
            + ("included" if show_compiler_runtime else "hidden (--show-compiler-runtime to reveal)") + "\n"
        )
        + f"max depth: {max_depth if max_depth is not None else 'unlimited'}\n"
        + "\n"
    )
    sections = [(None, view["tree_text"])]
    if view["fallback_text"]:
        sections.append((None, view["fallback_text"]))

    footer = (
        "Caveats:\n"
        "  - Every row is aggregated across all ranks (not one call tree per rank): CALLS and\n"
        "    TOTAL-AVG(s) are plain averages; SELF gets a full avg/std_dev/min/max load-balance\n"
        "    breakdown, the same convention extract_CPU_hotspots.py's own load-imbalance tables\n"
        "    use -- a rank that never reached a given node counts as 0 there, not omitted, so\n"
        "    real imbalance (e.g. a function only some ranks call) isn't hidden by averaging.\n"
        "  - sampling_wall_clock-<pid>.txt is used per rank when present (every real\n"
        "    unwound stack frame, not just instrumented boundaries); wall_clock-<pid>.txt is\n"
        "    used only as a whole-file fallback for a rank with no sampling file at all -- the\n"
        "    two are never spliced together. See extract_calltree_traced.py for the faster,\n"
        "    exact-where-instrumented alternative.\n"
        "  - SELF/TOTAL are this tool's own sampled measurement -- statistically approximate,\n"
        "    not exact.\n"
        "  - GPU-API/runtime noise (--show-gpu-api), rocprof-sys/GOTCHA wrapper frames\n"
        "    (--show-rocprofsys-internals), MPI library internals (--show-mpi-internals), and\n"
        "    compiler-runtime allocator/intrinsic helpers (--show-compiler-runtime) are all\n"
        "    hidden by default -- pass --show-all-internals for all four at once. Wrapper\n"
        "    frames are spliced out (children reparented, not deleted); MPI internals are\n"
        "    collapsed (the first real MPI frame is shown, its own internals are not); the\n"
        "    other two tiers are pruned (whole subtree hidden).\n"
        "  - The compiler-runtime tier was built from Cray's Fortran runtime specifically --\n"
        "    not necessarily complete for other compilers, since none have been observed in\n"
        "    this project's data so far.\n"
        "  - GPU kernel placement matches the kernel name's own compiler-embedded owner\n"
        "    subroutine against the merged tree when possible (precise, not a guess) -- a\n"
        "    kernel that can't be matched by name falls back to a structural estimate instead\n"
        "    (nearest launch-call ancestor, proportionally split by launch-call count when\n"
        "    multiple candidate sites exist). NEITHER approach is per-dispatch-exact --\n"
        "    this toolchain's text/JSON output has no per-call timestamps to correlate against;\n"
        "    only the binary Perfetto trace does, and that has no stdlib-friendly Python parser\n"
        "    (deferred future work, not attempted here).\n"
    )

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys (optionally + rocprofv3) output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the call tree report (default: <output_dir>/calltree.txt)")
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=None,
                         help="truncate the tree at this depth (default: unlimited, print the whole tree)")
    parser.add_argument("--show-gpu-api", action="store_true",
                         help="also show GPU-API/offload-runtime noise instead of hiding it")
    parser.add_argument("--show-rocprofsys-internals", action="store_true",
                         help="also show rocprof-sys's own instrumentation/GOTCHA/dynamic-linker "
                              "frames instead of splicing them out")
    parser.add_argument("--show-mpi-internals", action="store_true",
                         help="also show MPI library internals below the first MPI frame, instead "
                              "of collapsing them")
    parser.add_argument("--show-compiler-runtime", action="store_true",
                         help="also show compiler-runtime allocator/intrinsic helper noise instead "
                              "of hiding it")
    parser.add_argument("--show-all-internals", action="store_true",
                         help="shorthand for all four --show-* flags above at once")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "calltree.txt")
    write_report(
        args.output_dir, dest, max_depth=args.max_depth,
        show_gpu_api=args.show_gpu_api or args.show_all_internals,
        show_rocprofsys_internals=args.show_rocprofsys_internals or args.show_all_internals,
        show_mpi_internals=args.show_mpi_internals or args.show_all_internals,
        show_compiler_runtime=args.show_compiler_runtime or args.show_all_internals,
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
