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
calltree_common.merge_rank_trees()) into one aggregated tree -- a global view,
not one call tree per rank -- with each node's CALLS/TOTAL-AVG(s) columns
averaged, and SELF given a full avg/std_dev/min/max load-balance
breakdown, the same convention extract_CPU_hotspots.compute_load_imbalance()
already uses elsewhere in this codebase. See docs/plans/14-sampling-calltree-tool.md
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

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
from calltree_common import (
    REPORT_HEADERS,
    attach_kernel_summaries,
    build_children_map,
    flatten_tree,
    format_aligned_rows,
    make_kernel_node,
    make_node_values,
    merge_rank_trees,
    render_forest,
    resolve_run_dirs,
    unattached_kernel_per_rank,
)

# GPU-API/runtime noise -- broadened from extract_CPU_hotspots.py's
# GPU_API_PREFIXES (startswith-only) to substring/`in` matching, to also catch
# namespace-qualified C++ symbols the prefix check misses (e.g.
# "rocprofiler::hip::..."), plus GPU/offload-runtime noise observed in real
# sampled data that classify_gpu() has no prefix for at all.
GPU_NOISE_SUBSTRINGS = cpu_tool.GPU_API_PREFIXES + (
    "rocprofiler::", "cray_acc", "hiphardwaredevice", "present_table",
)

# rocprof-sys's own instrumentation/GOTCHA plumbing and dynamic-linker
# bootstrap frames. Real code (eventually main() and everything in it) sits
# *inside* these wrapper frames, not beside them -- so these are SPLICED out
# (node removed, children reparented to its own parent), never pruned:
# pruning them would delete the whole program along with them.
ROCPROFSYS_WRAPPER_SUBSTRINGS = (
    "tim::", "gotcha", "rocprofsys", "__libc_start", "lookup_hashtable",
    "lookup.constprop", "lib_bindings", "library_gots",
)

# MPI library internals -- COLLAPSED (the first real MPI frame hit while
# descending is shown, its own further internals are not), not pruned: the
# MPI call itself is real application-relevant information, only its
# multi-level implementation internals underneath are noise.
MPI_PREFIXES = ("mpi_", "pmpi_", "mpir_", "mpid_", "mpidi_")
MPI_FORTRAN_SHIM_SUFFIXES = ("_f08_", "_f08ts_")

# Compiler-runtime helper noise -- observed on Cray's Fortran runtime
# specifically (string/array intrinsics, the allocator chain behind
# ALLOCATE/DEALLOCATE, Fortran formatted I/O internals). The LEAST universal
# of the four tiers: not necessarily present with other compilers, since none
# have been observed in this project's data so far. PRUNED (nothing real is
# expected nested inside a compiler runtime's own allocator machinery).
COMPILER_RUNTIME_SUBSTRINGS = (
    "_f90_", "__allocate", "_dealloc", "posix_memalign", "_mid_memalign",
    "_int_memalign", "_int_malloc", "_int_free", "sysmalloc",
    "__default_morecore", "sbrk", "_fwf", "_xfer_iolist",
)

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


def is_kernel_descriptor_artifact(label):
    """See extract_calltree_traced.py's identical function: rocprof-sys's
    sampling sometimes attributes a GPU kernel launch to its compiled
    kernel-descriptor ELF symbol (the ".kd" suffix) directly, at near-zero
    duration, duplicating the same kernel's real device time already reported
    by rocprofv3's kernel_stats.csv. Treated as GPU/offload-runtime noise."""
    return label.endswith(".kd")


def is_gpu_api_entry(label):
    lname = label.lower()
    return any(s in lname for s in GPU_NOISE_SUBSTRINGS)


def is_rocprofsys_wrapper(label):
    lname = label.lower()
    return any(s in lname for s in ROCPROFSYS_WRAPPER_SUBSTRINGS)


def is_mpi_territory(label):
    lname = label.lower()
    return lname.startswith(MPI_PREFIXES) or lname.endswith(MPI_FORTRAN_SHIM_SUFFIXES)


def is_compiler_runtime_noise(label):
    lname = label.lower()
    return any(s in lname for s in COMPILER_RUNTIME_SUBSTRINGS)


def classify_gpu_broad(row):
    """Same ancestry-propagation idea as extract_CPU_hotspots.classify_gpu()
    (a thread-root row with no GPU label of its own still inherits GPU
    classification if any ancestor has one -- e.g. a background thread the
    HIP runtime spawns), but matching is_gpu_api_entry()/
    is_kernel_descriptor_artifact() (substring-based) instead of
    classify_gpu()'s startswith-only check. Filename hints
    (extract_CPU_hotspots.GPU_FILE_HINTS) are skipped entirely here -- this
    tool only ever reads wall_clock-<pid>.txt/sampling_wall_clock-<pid>.txt,
    whose names never carry those hints anyway.
    """
    if is_gpu_api_entry(row["label"]) or is_kernel_descriptor_artifact(row["label"]):
        return True
    if row.get("is_thread_root"):
        ancestor = row["parent"]
        while ancestor is not None:
            if is_gpu_api_entry(ancestor["label"]) or is_kernel_descriptor_artifact(ancestor["label"]):
                return True
            ancestor = ancestor["parent"]
    return False


def propagate_gpu_to_untethered_thread_roots(rows):
    """A background/event-loop thread rocprof-sys's own pthread_create_gotcha
    wrapper spawns (e.g. HIP/ROCr's async-completion or event-polling
    threads) samples as its OWN independent root: its DEPTH resets to 0 for
    the new OS thread, so attach_ancestry() gives it parent=None -- the same
    as a real top-level thread spawned directly by application code, and
    unlike the nested case classify_gpu_broad() already handles (a
    thread-root row still linked to its spawning ancestor via a real parent
    chain, DEPTH nesting one level deeper instead of resetting to 0). With no
    parent to walk upward through at all, the only signal that such a root is
    driver-internal noise rather than real application work is what's
    immediately inside it: real data confirmed the exact shape --
    "start_thread" -> rocprofsys's own pthread_create_gotcha wrapper (every
    spawned thread gets this hop, regardless of what it does) ->
    "rocr::os::ThreadTrampoline" and deeper ROCm-runtime internals. This
    walks down through any wrapper hop to the first real content; if that's
    GPU/ROCm-runtime noise, the untethered root itself is reclassified too,
    so the whole subtree hides as one block under --show-gpu-api like any
    other GPU noise -- regardless of whether --show-rocprofsys-internals
    also reveals the wrapper hop's own label. A root already GPU-classified,
    with no children, or whose real content is genuine application code
    (e.g. a worker thread the application itself spawns, also wrapped by the
    same rocprof-sys instrumentation but leading to real code instead) is
    left untouched. Mutates rows in place.
    """
    children_by_parent_id = {}
    for row in rows:
        parent = row["parent"]
        if parent is not None:
            children_by_parent_id.setdefault(id(parent), []).append(row)

    for row in rows:
        if row["parent"] is not None or row["gpu"]:
            continue
        node = row
        seen = set()
        while True:
            kids = children_by_parent_id.get(id(node))
            if not kids:
                break
            child = kids[0]
            if id(child) in seen:
                break  # defensive: a real cycle should never happen here
            seen.add(id(child))
            if not is_rocprofsys_wrapper(child["label"]):
                if child["gpu"]:
                    row["gpu"] = True
                break
            node = child


def splice_out_wrapper_nodes(rows):
    """Reassigns every row's parent pointer to skip past any chain of
    is_rocprofsys_wrapper() ancestors, then drops wrapper rows from the list
    entirely -- a row whose whole ancestor chain was wrapper frames ends up
    with parent=None, correctly becoming a new root (e.g. "main", once
    __libc_start_main/__libc_start_call_main/rocprofsys_main are spliced out
    from above it). Splicing (not pruning) is essential here: real code sits
    *inside* these wrapper frames, not beside them, so pruning them like GPU
    noise would delete the whole program along with them.
    """
    for row in rows:
        parent = row["parent"]
        while parent is not None and is_rocprofsys_wrapper(parent["label"]):
            parent = parent["parent"]
        row["parent"] = parent
    return [r for r in rows if not is_rocprofsys_wrapper(r["label"])]


def load_rank_trees(cpu_dir, show_rocprofsys_internals):
    """Per rank: parse_table_file() + attach_ancestry() directly (NOT
    scan_ranks(), which merges same-label rows and would destroy tree
    identity). sampling_wall_clock-<pid>.txt wins when present (the whole
    point of this tool -- every real unwound stack frame, not just
    instrumented boundaries); wall_clock-<pid>.txt is used only as a
    whole-file fallback for a rank with no sampling file at all.

    Returns a list of (rank_key, rows, roots) tuples, one per rank. Before
    roots are computed: propagate_gpu_to_untethered_thread_roots() catches a
    background/event-loop thread that samples as its own untethered root (no
    parent link at all -- see its own docstring); then, unless
    show_rocprofsys_internals, the wrapper-splice pass runs, so a rank whose
    whole top of stack was wrapper frames correctly surfaces "main" (or
    whatever real code sits under them) as its own root.
    """
    paths_by_rank = {}
    order = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "sampling_wall_clock-*.txt"), recursive=True)):
        m = cpu_tool.PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        paths_by_rank[rank_key] = path
        order.append(rank_key)
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "wall_clock-*.txt"), recursive=True)):
        m = cpu_tool.PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        if rank_key not in paths_by_rank:
            paths_by_rank[rank_key] = path
            order.append(rank_key)

    result = []
    for rank_key in order:
        path = paths_by_rank[rank_key]
        rows = cpu_tool.parse_table_file(path)
        if not rows:
            continue
        cpu_tool.attach_ancestry(rows)
        for row in rows:
            row["gpu"] = classify_gpu_broad(row)
            row["compiler_runtime"] = is_compiler_runtime_noise(row["label"])
            row["mpi_territory"] = is_mpi_territory(row["label"])
        propagate_gpu_to_untethered_thread_roots(rows)
        if not show_rocprofsys_internals:
            rows = splice_out_wrapper_nodes(rows)
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

    def is_pruned(node):
        return (node["gpu"] and not show_gpu_api) or (node.get("compiler_runtime") and not show_compiler_runtime)

    def collapses_children(row):
        return row.get("mpi_territory", False) and not show_mpi_internals

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
    rows = gpu_tool.parse_kernel_stats_csv(path)
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
