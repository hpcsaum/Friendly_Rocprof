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
import glob
import os
from datetime import datetime

import extract_GPU_hotspots as gpu_tool
from stage1_rocprofsys import PID_SUFFIX_RE, parse_table_file
from stage1_rocprofv3 import parse_kernel_stats_csv
from stage1_run_dirs import resolve_run_dirs
from stage2_rocprofsys import attach_ancestry
from stage3_rocprofsys import (
    load_default_patterns,
    make_collapses_children,
    make_is_pruned,
    remove_tagged_subtrees,
    splice_by_tag,
    tag_rows,
)
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
(or --show-all-internals for all four at once).

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


def load_rank_trees(cpu_dir, show_rocprofsys_internals):
    """Per rank: parse_table_file() + attach_ancestry() directly (NOT
    scan_ranks(), which merges same-label rows and would destroy tree
    identity). sampling_wall_clock-<pid>.txt wins when present (the whole
    point of this tool -- every real unwound stack frame, not just
    instrumented boundaries); wall_clock-<pid>.txt is used only as a
    whole-file fallback for a rank with no sampling file at all.

    Returns a list of (rank_key, rows, roots) tuples, one per rank.
    tag_rows() classifies every row in one pass (including inheriting
    gpu_api/mpi_territory onto a background/event-loop thread that samples as
    its own untethered root, no parent link at all -- see
    stage3_rocprofsys.py's first-real-descendant scope); then, unless
    show_rocprofsys_internals, the whole wrapper_branch_noise-tagged sibling
    subtree is dropped before the wrapper-splice pass runs, so a rank whose
    whole top of stack was wrapper frames correctly surfaces "main" (or
    whatever real code sits under them) as its own root.
    """
    paths_by_rank = {}
    order = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "sampling_wall_clock-*.txt"), recursive=True)):
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        paths_by_rank[rank_key] = path
        order.append(rank_key)
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "wall_clock-*.txt"), recursive=True)):
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
        if not show_rocprofsys_internals:
            # Order matters: remove_tagged_subtrees() needs wrapper_noise-matching
            # descendants (e.g. get_library) still present to find contamination in
            # the first place -- splice_by_tag() would have already removed them.
            rows = remove_tagged_subtrees(rows, {"wrapper_branch_noise"})
            rows = splice_by_tag(rows, "wrapper_noise", fold=False)  # fold=False preserves
                                                                        # today's discard-self-
                                                                        # time behavior
        roots = [r for r in rows if r["parent"] is None]
        result.append((rank_key, rows, roots))
    return result


def write_report(run_dir, dest_path, max_depth=None, show_gpu_api=False,
                  show_rocprofsys_internals=False, show_mpi_internals=False,
                  show_compiler_runtime=False):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    ranks = load_rank_trees(cpu_dir, show_rocprofsys_internals)
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {cpu_dir!r} "
            "(expected files like sampling_wall_clock-<pid>.txt) -- nothing to render"
        )

    prune_tags = (
        ({"gpu_api"} if not show_gpu_api else set())
        | ({"compiler_runtime_noise"} if not show_compiler_runtime else set())
    )
    is_pruned = make_is_pruned(prune_tags)
    collapses_children = make_collapses_children({"mpi_territory"} if not show_mpi_internals else set())

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
    parts.append("Call tree report (sampling-based, aggregated across ranks)\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(run_dir)}\n")
    parts.append(f"CPU data: {os.path.abspath(cpu_dir)}\n")
    parts.append(f"GPU data: {os.path.abspath(gpu_dir) if gpu_per_rank is not None else '(none)'}\n")
    parts.append(f"ranks aggregated: {len(ranks)} (rank keys: {', '.join(rank_keys)})\n")
    parts.append(
        "Showing: GPU-API/runtime noise "
        + ("included" if show_gpu_api else "hidden (--show-gpu-api to reveal)") + ", "
        "rocprof-sys internals "
        + ("included" if show_rocprofsys_internals else "spliced out (--show-rocprofsys-internals to reveal)") + ", "
        "MPI internals "
        + ("included" if show_mpi_internals else "collapsed (--show-mpi-internals to reveal)") + ", "
        "compiler-runtime helpers "
        + ("included" if show_compiler_runtime else "hidden (--show-compiler-runtime to reveal)") + "\n"
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

    children_map = build_children_map(flat, collapses_children=collapses_children)
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

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def _kernel_totals_with_counts(gpu_dir, rank_index):
    """See extract_calltree_traced.py's identical function: re-parse that
    rank's own kernel_stats.csv directly for its Calls column, since
    gpu_tool.aggregate_per_rank() only returns total seconds."""
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
