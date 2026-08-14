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
import glob
import os
from datetime import datetime

import extract_GPU_hotspots as gpu_tool
from stage1_rocprofsys import PID_SUFFIX_RE, parse_table_file
from stage1_rocprofv3 import parse_kernel_stats_csv
from stage1_run_dirs import resolve_run_dirs
from stage2_rocprofsys import attach_ancestry
from stage3_rocprofsys import load_default_patterns, make_is_pruned, tag_rows
from stage4_rocprofsys_tree import (
    attach_kernel_summaries,
    flatten_tree,
    make_kernel_node,
    make_node_values,
    merge_rank_trees,
    unattached_kernel_per_rank,
)
from tree_render import (
    REPORT_HEADERS,
    build_children_map,
    format_aligned_rows,
    render_forest,
)

TAG_DEFS = load_default_patterns()

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


def load_rank_trees(cpu_dir):
    """Per rank: parse_table_file() + attach_ancestry() + tag_rows() directly
    (NOT scan_ranks(), which merges same-label rows and would destroy tree
    identity). wall_clock-<pid>.txt wins when present; sampling_wall_clock-<pid>.txt
    is used only for a rank that has no wall_clock file at all -- the two are
    never spliced together into one tree, since their parent-links come from two
    independently-reconstructed call orders.

    Returns a list of (rank_key, rows, roots) tuples, one per rank, in sorted
    order. rows is every parsed row (parent/depth/thread_id/tags all set); roots
    is the subset with parent is None -- every row with parent is None starts
    its own tree, which correctly separates multiple OS threads' subtrees
    within one rank regardless of which of the two DEPTH-numbering shapes the
    file uses (see docs/plans/1.13-calltree-tool.md point 2 -- is_thread_root
    isn't reliably set in the DEPTH-resets-to-0 case, but parent is None always is).
    This tool only ever acts on the gpu_api tag (see write_report()) -- tag_rows()
    also computes wrapper_noise/mpi_territory/compiler_runtime_noise on every row,
    harmlessly unused, keeping this tool's own "exactly one filter tier" scope.
    """
    paths_by_rank = {}
    order = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "wall_clock-*.txt"), recursive=True)):
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        paths_by_rank[rank_key] = path
        order.append(rank_key)
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "sampling_wall_clock-*.txt"), recursive=True)):
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        if rank_key not in paths_by_rank:
            paths_by_rank[rank_key] = path
            order.append(rank_key)

    result = []
    for rank_key in order:
        path = paths_by_rank[rank_key]
        rows = parse_table_file(path)
        if not rows:
            continue
        attach_ancestry(rows)
        tag_rows(rows, TAG_DEFS, filename=path)
        roots = [r for r in rows if r["parent"] is None]
        result.append((rank_key, rows, roots))
    return result


def write_report(run_dir, dest_path, max_depth=None, show_gpu_api=False):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    ranks = load_rank_trees(cpu_dir)
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {cpu_dir!r} "
            "(expected files like wall_clock-<pid>.txt) -- nothing to render"
        )

    is_pruned = make_is_pruned(set() if show_gpu_api else {"gpu_api"})
    rank_keys = [rank_key for rank_key, _rows, _roots in ranks]
    node_values = make_node_values(rank_keys)

    merged_roots = merge_rank_trees(ranks)
    flat = flatten_tree(merged_roots)

    gpu_per_rank = None
    if gpu_dir is not None:
        gpu_totals, gpu_scanned = gpu_tool.aggregate_per_rank(gpu_dir)
        if gpu_scanned and len(gpu_totals) == len(ranks):
            gpu_per_rank = gpu_totals
        elif gpu_scanned:
            print(
                f"warning: {run_dir!r}: rocprof-sys reports {len(ranks)} rank(s) but "
                f"rocprofv3 reports {len(gpu_totals)} -- skipping GPU kernel integration "
                "rather than risk pairing mismatched ranks",
            )

    parts = []
    parts.append("Call tree report (traced/wall_clock-based, aggregated across ranks)\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(run_dir)}\n")
    parts.append(f"CPU data: {os.path.abspath(cpu_dir)}\n")
    parts.append(f"GPU data: {os.path.abspath(gpu_dir) if gpu_per_rank is not None else '(none)'}\n")
    parts.append(f"ranks aggregated: {len(ranks)} (rank keys: {', '.join(rank_keys)})\n")
    parts.append(
        "Showing user code + MPI calls only"
        + (", GPU-API/runtime calls included\n" if show_gpu_api else " (pass --show-gpu-api to also show GPU-API/runtime calls)\n")
    )
    parts.append(f"max depth: {max_depth if max_depth is not None else 'unlimited'}\n")
    parts.append("\n")

    unattached = set()
    if gpu_per_rank is not None:
        # aggregate_per_rank() only gives total seconds, not call counts --
        # re-derive counts from each rank's own paired kernel_stats.csv directly.
        gpu_kernel_by_rank = {
            rank_key: _kernel_totals_with_counts(gpu_dir, i) for i, rank_key in enumerate(rank_keys)
        }
        unattached = attach_kernel_summaries(flat, gpu_kernel_by_rank, is_pruned)

    children_map = build_children_map(flat)
    parts.append(format_aligned_rows(
        render_forest(merged_roots, children_map, max_depth, is_pruned, node_values), REPORT_HEADERS,
    ))
    parts.append("\n")

    if unattached:
        parts.append("=== GPU kernels (rocprofv3) -- no owning subroutine or launch call site found in CPU tree ===\n")
        fallback_rows = [
            (f"  {kernel_name}", node_values(make_kernel_node(kernel_name, per_rank)))
            for kernel_name, per_rank in unattached_kernel_per_rank(unattached, gpu_kernel_by_rank).items()
        ]
        fallback_rows.sort(key=lambda r: -r[1][1])
        parts.append(format_aligned_rows(fallback_rows, REPORT_HEADERS))
        parts.append("\n")

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


def _kernel_totals_with_counts(gpu_dir, rank_index):
    """gpu_tool.aggregate_per_rank() only returns {kernel_name: total_seconds}
    (no call count). Re-parse that rank's own kernel_stats.csv directly for
    the Calls column too, rather than duplicating aggregate_per_rank()'s file
    discovery -- same file, read twice, cheap for text this small."""
    candidates = sorted(glob.glob(os.path.join(gpu_dir, "**", "*_kernel_stats.csv"), recursive=True))
    path = candidates[rank_index]
    rows = parse_kernel_stats_csv(path)
    totals = {}
    for row in rows:
        entry = totals.setdefault(row["label"], [0, 0.0])
        entry[0] += row["count"]
        entry[1] += row["total_ns"] / 1e9
    return {k: tuple(v) for k, v in totals.items()}


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
