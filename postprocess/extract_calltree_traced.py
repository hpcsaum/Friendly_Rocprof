#!/usr/bin/env python3
"""Render an indented call tree from rocprof-sys timemory text output, optionally
with rocprofv3 GPU kernel data nested in at the CPU call site(s) that launched them.

This is the fast, exact-where-instrumented variant: it prefers wall_clock-<pid>.txt
(GOTCHA-instrumented -- exact timing for every intercepted call) per rank, falling
back to sampling_wall_clock-<pid>.txt only for a rank with no wall_clock file at all.
Its tree is therefore only as deep as rocprof-sys's own instrumentation boundaries --
a real intermediate frame that was never instrumented is invisible, so a child can
appear to hang directly off a much higher ancestor than it really does. See
`extract_calltree.py` (the sampling-based tool) for the true, deeper call structure,
at the cost of needing noise filtering and only statistically-approximate timing.

Unlike extract_CPU_hotspots.py's scan_ranks()/aggregate(), this does NOT merge
same-label rows within one rank's own file -- a calltree needs every individual
call-tree node kept distinct (attach_ancestry()'s parent links intact), not summed
by label. It DOES merge structurally ACROSS ranks (see stage4_rocprofsys_tree.merge_rank_trees())
into one aggregated tree -- a global view, not one call tree per rank -- with each
node's CALLS/SELF/TOTAL columns averaged (and self-time's load imbalance shown via
std_dev/min/max) across every rank, the same avg/std_dev/min/max convention
extract_CPU_hotspots.compute_load_imbalance() already uses elsewhere in this codebase.
See docs/plans/1.13-calltree-tool.md for the full original design rationale,
including why per-dispatch-exact kernel placement isn't achievable from this
toolchain's text/JSON output (only the binary Perfetto trace has per-call
timestamps, and there's no stdlib-friendly way to parse it -- deferred future
work, not silently dropped).
"""

import argparse
import os
from datetime import datetime

from stage1_run_dirs import resolve_run_dirs
from stage5_calltree_traced_view import build_calltree_view

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

This is the fast, exact-where-instrumented variant: it prefers
wall_clock-<pid>.txt (GOTCHA-instrumented) per rank, so timing is exact for
every intercepted call, but the tree is only as deep as rocprof-sys's own
instrumentation boundaries -- a real but never-instrumented intermediate
frame is invisible, so a child can appear to hang directly off a much higher
ancestor than it really does. See extract_calltree.py (the sampling-based
tool) for the true, deeper call structure instead, at the cost of needing
noise filtering and only statistically-approximate timing.

Filtering matches tool 3's "CPU compute hotspots" bucket: GPU-API/runtime
noise (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-prefixed calls, plus
kernel-descriptor sampling artifacts -- labels ending in ".kd") is hidden by
default, so what's left is your own code plus MPI calls. Pass
--show-gpu-api to see the hidden chain too.

When a paired rocprofv3 directory is found, real GPU kernel data is nested
into the tree at the CPU subroutine that actually contains it, matched by
name when the compiler's own kernel naming allows it, or a structural
estimate (nearest launch-call ancestor, proportionally split by launch-call
count when there's more than one candidate site) otherwise. NEITHER is
per-dispatch-exact: this toolchain's text/JSON output has no per-call
timestamps to correlate against, only the binary Perfetto trace does, and
that has no stdlib-friendly Python parser (a future tool, not attempted
here).

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def write_report(run_dir, dest_path, max_depth=None, show_gpu_api=False):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    view = build_calltree_view(run_dir, cpu_dir, gpu_dir, max_depth=max_depth, show_gpu_api=show_gpu_api)
    rank_keys = view["rank_keys"]

    parts = []
    parts.append("Call tree report (traced/wall_clock-based, aggregated across ranks)\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(run_dir)}\n")
    parts.append(f"CPU data: {os.path.abspath(cpu_dir)}\n")
    parts.append(f"GPU data: {os.path.abspath(gpu_dir) if view['gpu_paired'] else '(none)'}\n")
    parts.append(f"ranks aggregated: {len(rank_keys)} (rank keys: {', '.join(rank_keys)})\n")
    parts.append(
        "Showing user code + MPI calls only"
        + (", GPU-API/runtime calls included\n" if show_gpu_api else " (pass --show-gpu-api to also show GPU-API/runtime calls)\n")
    )
    parts.append(f"max depth: {max_depth if max_depth is not None else 'unlimited'}\n")
    parts.append("\n")

    parts.append(view["tree_text"])
    parts.append(view["fallback_text"])

    parts.append(
        "Caveats:\n"
        "  - Every row is aggregated across all ranks (not one call tree per rank): CALLS and\n"
        "    TOTAL-AVG(s) are plain averages; SELF gets a full avg/std_dev/min/max load-balance\n"
        "    breakdown, the same convention extract_CPU_hotspots.py's own load-imbalance tables\n"
        "    use -- a rank that never reached a given node counts as 0 there, not omitted, so\n"
        "    real imbalance (e.g. a function only some ranks call) isn't hidden by averaging.\n"
        "  - GPU-API/runtime rows are hidden by default (pass --show-gpu-api to see them) --\n"
        "    same classification as extract_CPU_hotspots.py's GPU-API/overhead bucket.\n"
        "  - Kernel-descriptor sampling artifacts (labels ending in \".kd\") are hidden by\n"
        "    default too -- rocprof-sys's own sampling sometimes attributes a GPU kernel launch\n"
        "    to its compiled kernel-descriptor symbol directly in the CPU call tree, at\n"
        "    near-zero duration, duplicating the same kernel's real device time already shown\n"
        "    under \"[GPU kernels -- rocprofv3]\" below. Pass --show-gpu-api to see them too.\n"
        "  - GPU kernel placement matches the kernel name's own compiler-embedded owner\n"
        "    subroutine against the merged tree when possible (precise, not a guess) -- a\n"
        "    kernel that can't be matched by name falls back to a structural estimate instead\n"
        "    (nearest launch-call ancestor, proportionally split by launch-call count when\n"
        "    multiple candidate sites exist). NEITHER approach is per-dispatch-exact --\n"
        "    this toolchain's text/JSON output has no per-call timestamps to correlate against;\n"
        "    only the binary Perfetto trace does, and that has no stdlib-friendly Python parser\n"
        "    (deferred future work, not attempted here).\n"
        "  - wall_clock-<pid>.txt is used per rank when present (exact for GOTCHA-intercepted\n"
        "    MPI calls); sampling_wall_clock-<pid>.txt is used only as a whole-file fallback\n"
        "    for a rank with no wall_clock file at all -- the two are never spliced together.\n"
        "    This also means the tree is only as deep as rocprof-sys's instrumentation --\n"
        "    see extract_calltree.py for the sampling-based tool that shows real call depth.\n"
    )

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys (optionally + rocprofv3) output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the call tree report (default: <output_dir>/calltree_traced.txt)")
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=None,
                         help="truncate the tree at this depth (default: unlimited, print the whole tree)")
    parser.add_argument("--show-gpu-api", dest="show_gpu_api", action="store_true",
                         help="also show GPU-API/runtime calls (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-"
                              "prefixed) instead of hiding them")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "calltree_traced.txt")
    write_report(args.output_dir, dest, max_depth=args.max_depth, show_gpu_api=args.show_gpu_api)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
