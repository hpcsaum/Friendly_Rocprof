#!/usr/bin/env python3
"""Render an indented call tree from rocprof-sys timemory text output, optionally
with rocprofv3 GPU kernel data nested in at the CPU call site(s) that launched them.

Unlike extract_CPU_hotspots.py's scan_ranks()/aggregate(), this does NOT merge
same-label rows across a file -- a calltree needs every individual call-tree
node kept distinct (attach_ancestry()'s parent links intact), not summed by
label. See docs/plans/13-calltree-tool.md for the full design rationale,
including why per-dispatch-exact kernel placement isn't achievable from this
toolchain's text/JSON output (only the binary Perfetto trace has per-call
timestamps, and there's no stdlib-friendly way to parse it -- deferred future
work, not silently dropped).
"""

import argparse
import glob
import os
import statistics
from datetime import datetime

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool

# Best-effort list of known HIP kernel-launch entry points -- not exhaustive (same
# "best-effort guess" spirit as extract_CPU_hotspots.py's EXECUTABLE_KEYS etc.).
# Matched by startswith(), not equality, since a label can carry a demangled
# parameter signature suffix (e.g. "hipLaunchKernel(void const*, ...)").
KERNEL_LAUNCH_LABELS = (
    "hipLaunchKernel",
    "hipModuleLaunchKernel",
    "hipExtLaunchKernel",
    "hipLaunchKernelGGL",
    "hipExtModuleLaunchKernel",
    "hipGraphLaunch",
)

HELP_BLURB = """\
Reads a rocprof-sys (optionally paired with rocprofv3) output directory and
writes an indented call tree -- actual function nesting, not a flat ranked
list. Works on tool 1/3 output (profile_CPU_hotspots.sh/profile_hotspots.sh)
and tool 4's scan directory (instrument_hotspots.sh trace) identically --
same underlying data format either way.

Filtering matches tool 3's "CPU compute hotspots" bucket: GPU-API/runtime
noise (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-prefixed calls, plus
kernel-descriptor sampling artifacts -- labels ending in ".kd") is hidden by
default, so what's left is your own code plus MPI calls. Pass
--show-gpu-api to see the hidden chain too.

When a paired rocprofv3 directory is found, real GPU kernel data is nested
into the tree at the CPU call site(s) that launched kernels -- this is a
structural estimate (nearest launch-call ancestor, proportionally split by
launch-call count when there's more than one candidate site), NOT a
per-dispatch-exact placement: this toolchain's text/JSON output has no
per-call timestamps to correlate against, only the binary Perfetto trace
does, and that has no stdlib-friendly Python parser (a future tool, not
attempted here).

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def resolve_run_dirs(run_dir):
    """Same auto-detection as extract_pop_metrics.py's resolve_run_dirs() --
    duplicated locally per this codebase's existing "small helpers are
    duplicated across standalone tools" convention (see extract_GPU_hotspots.py's
    compute_load_imbalance docstring)."""
    cpu_subdir = os.path.join(run_dir, "rocprof-sys")
    gpu_subdir = os.path.join(run_dir, "rocprofv3")
    cpu_dir = cpu_subdir if os.path.isdir(cpu_subdir) else run_dir
    gpu_dir = gpu_subdir if os.path.isdir(gpu_subdir) else None
    return cpu_dir, gpu_dir


def load_rank_trees(cpu_dir):
    """Per rank: parse_table_file() + attach_ancestry() + classify_gpu() directly
    (NOT scan_ranks(), which merges same-label rows and would destroy tree
    identity). wall_clock-<pid>.txt wins when present; sampling_wall_clock-<pid>.txt
    is used only for a rank that has no wall_clock file at all -- the two are
    never spliced together into one tree, since their parent-links come from two
    independently-reconstructed call orders.

    Returns a list of (rank_key, rows, roots) tuples, one per rank, in sorted
    order. rows is every parsed row (parent/depth/thread_id/gpu all set); roots
    is the subset with parent is None -- every row with parent is None starts
    its own tree, which correctly separates multiple OS threads' subtrees
    within one rank regardless of which of the two DEPTH-numbering shapes the
    file uses (see docs/plans/13-calltree-tool.md point 2 -- is_thread_root
    isn't reliably set in the DEPTH-resets-to-0 case, but parent is None always is).
    """
    paths_by_rank = {}
    order = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "wall_clock-*.txt"), recursive=True)):
        m = cpu_tool.PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        paths_by_rank[rank_key] = path
        order.append(rank_key)
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", "sampling_wall_clock-*.txt"), recursive=True)):
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
            row["gpu"] = cpu_tool.classify_gpu(row, path) or is_kernel_descriptor_artifact(row["label"])
        roots = [r for r in rows if r["parent"] is None]
        result.append((rank_key, rows, roots))
    return result


def is_kernel_descriptor_artifact(label):
    """rocprof-sys's default sampling sometimes attributes a GPU kernel launch to
    its compiled kernel-descriptor ELF symbol (the ".kd" suffix -- standard
    AMDGPU convention) directly inside a wall_clock-<pid>.txt row, at near-zero
    duration. Confirmed real (not a wall_clock/sampling_wall_clock merge issue --
    these rows are already present in wall_clock-<pid>.txt on its own): the same
    kernel's real device time is already reported by rocprofv3's kernel_stats.csv
    and surfaces correctly via attach_kernel_summaries(); left visible, this shows
    up as a near-duplicate, near-zero-duration entry under whatever CPU call site
    happened to be sampled at launch time. Treated the same as GPU-API/runtime
    noise: hidden by default, visible with --show-gpu-api.
    """
    return label.endswith(".kd")


def nearest_visible_ancestor(row, show_gpu_api):
    """Walk parent links up from row past any GPU-API-classified node (unless
    --show-gpu-api), to the ancestor that will actually be rendered -- the
    correct kernel-attachment point either way."""
    node = row["parent"]
    while node is not None and node["gpu"] and not show_gpu_api:
        node = node["parent"]
    return node


def find_kernel_anchors(rows, show_gpu_api):
    """Returns {id(anchor_row): [anchor_row, launch_call_weight]} for every
    distinct visible ancestor a KERNEL_LAUNCH_LABELS row's nearest_visible_ancestor()
    resolves to, weighted by that row's own call count. A launch-family row with
    no visible ancestor at all (vanishingly rare -- would mean the launch call
    is itself an un-rooted node) contributes to neither this dict nor the
    no-anchor fallback; skipped rather than mis-attributed.
    """
    anchors = {}
    for row in rows:
        if not row["label"].startswith(KERNEL_LAUNCH_LABELS):
            continue
        anchor = nearest_visible_ancestor(row, show_gpu_api)
        if anchor is None:
            continue
        key = id(anchor)
        if key not in anchors:
            anchors[key] = [anchor, 0]
        anchors[key][1] += row["count"]
    return anchors


def make_kernel_node(label, count, total_sec):
    """A synthetic (not from parse_table_file()) tree node -- same shape as a
    real row so render_node() can treat it identically, plus "static_children"
    for its own kernel-name breakdown (real rows never have this key)."""
    return {
        "label": label, "count": count, "sum": total_sec, "self_sum": total_sec,
        "gpu": False, "static_children": [],
    }


def attach_kernel_summaries(rows, roots, gpu_kernel_totals, show_gpu_api):
    """Mutates rows in place: inserts a synthetic "[GPU kernels -- rocprofv3]"
    node (see make_kernel_node()) as a static_children entry of the right
    anchor(s), per docs/plans/13-calltree-tool.md's placement rules. Returns
    True if at least one anchor was found (False means this rank's kernel data
    belongs in the top-level no-anchor fallback section instead).
    """
    anchors = find_kernel_anchors(rows, show_gpu_api)
    if not anchors:
        return False

    total_kernel_time = sum(t for _c, t in gpu_kernel_totals.values())
    total_kernel_calls = sum(c for c, _t in gpu_kernel_totals.values())
    total_weight = sum(w for _a, w in anchors.values())

    for anchor, weight in anchors.values():
        if total_weight > 0:
            fraction = weight / total_weight
        else:
            # Every candidate site had a launch-family row with count=0 --
            # shouldn't normally happen, but avoid dividing by zero rather
            # than crash: fall back to an even split.
            fraction = 1.0 / len(anchors)

        if len(anchors) == 1:
            label = "[GPU kernels -- rocprofv3]"
        else:
            pct = fraction * 100.0
            label = (
                f"[GPU kernels -- rocprofv3, ~{pct:.0f}% estimate: this site issued "
                f"{weight}/{total_weight} observed launch calls]"
            )

        node = make_kernel_node(label, round(total_kernel_calls * fraction), total_kernel_time * fraction)
        for kernel_name, (count, total_sec) in sorted(gpu_kernel_totals.items(), key=lambda kv: -kv[1][1]):
            node["static_children"].append(make_kernel_node(kernel_name, round(count * fraction), total_sec * fraction))
        anchor.setdefault("static_children", []).append(node)

    return True


def get_children(node, children_map):
    kids = list(children_map.get(id(node), []))
    kids.extend(node.get("static_children", []))
    return kids


def build_children_map(rows):
    children = {}
    for row in rows:
        parent = row["parent"]
        if parent is not None:
            children.setdefault(id(parent), []).append(row)
    return children


def render_node(node, prefix, is_last, show_connector, children_map, level, max_depth, show_gpu_api, out):
    """Appends (label_text, count, self_sec, total_sec) tuples to out -- one per
    rendered line; count/self_sec/total_sec are None for the "N more node(s)
    hidden" marker line, which has no metrics of its own. Real columns (not a
    "[calls=.../self=.../total=...]" string repeated on every line) are built by
    format_aligned_rows() from these tuples. Uses tree-drawing connectors
    (├── / └── / │) like the `tree` command, so a node's nesting level is
    unambiguous without counting indent spaces -- roots themselves (show_connector
    =False) print flush, since a rank's multiple independent roots (e.g. separate
    OS threads) aren't true siblings under one shared parent; connectors start
    from each root's own children downward.
    """
    label_text = f"{prefix}{'└── ' if is_last else '├── '}{node['label']}" if show_connector else node["label"]
    out.append((label_text, node["count"], node["self_sum"], node["sum"]))

    kids = [k for k in get_children(node, children_map) if show_gpu_api or not k["gpu"]]
    if not kids:
        return
    child_prefix = prefix + ("    " if is_last else "│   ") if show_connector else ""
    if max_depth is not None and level >= max_depth:
        hidden = count_all_descendants(kids, children_map, show_gpu_api)
        marker = f"{child_prefix}└── ... ({hidden} more node(s) hidden below this point, raise --max-depth to see them)"
        out.append((marker, None, None, None))
        return
    for i, kid in enumerate(kids):
        render_node(kid, child_prefix, i == len(kids) - 1, True, children_map, level + 1, max_depth, show_gpu_api, out)


def render_forest(roots, children_map, max_depth, show_gpu_api):
    """One rank's whole forest (every root tree, e.g. one per OS thread) as a
    list of (label_text, count, self_sec, total_sec) tuples, ready for
    format_aligned_rows()."""
    out = []
    for root in roots:
        if show_gpu_api or not root["gpu"]:
            render_node(root, "", False, False, children_map, 0, max_depth, show_gpu_api, out)
    return out


def format_aligned_rows(rows):
    """Real right-aligned CALLS/SELF(s)/TOTAL(s) columns under one header, sized
    to this block's longest label -- not a "[calls=.../self=.../total=...]"
    string repeated on every line. A marker row (count is None, e.g. the "N more
    node(s) hidden" line) has no metrics of its own and is printed as plain text.
    Returns "" for an empty block (no header printed with nothing under it)."""
    data_rows = [r for r in rows if r[1] is not None]
    if not data_rows:
        return ""

    label_width = max(len(text) for text, _count, _self, _total in data_rows)
    lines = [f"{'':<{label_width}}  {'CALLS':>8}  {'SELF(s)':>12}  {'TOTAL(s)':>12}"]
    for text, count, self_sec, total_sec in rows:
        if count is None:
            lines.append(text)
        else:
            lines.append(f"{text:<{label_width}}  {count:>8}  {self_sec:>12.6f}  {total_sec:>12.6f}")
    return "\n".join(lines) + "\n"


def count_all_descendants(nodes, children_map, show_gpu_api):
    total = 0
    for node in nodes:
        total += 1
        kids = [k for k in get_children(node, children_map) if show_gpu_api or not k["gpu"]]
        total += count_all_descendants(kids, children_map, show_gpu_api)
    return total


def write_report(run_dir, dest_path, max_depth=None, show_gpu_api=False):
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    ranks = load_rank_trees(cpu_dir)
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {cpu_dir!r} "
            "(expected files like wall_clock-<pid>.txt) -- nothing to render"
        )

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
    parts.append("Call tree report\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(run_dir)}\n")
    parts.append(f"CPU data: {os.path.abspath(cpu_dir)}\n")
    parts.append(f"GPU data: {os.path.abspath(gpu_dir) if gpu_per_rank is not None else '(none)'}\n")
    parts.append(
        "Showing user code + MPI calls only"
        + (", GPU-API/runtime calls included\n" if show_gpu_api else " (pass --show-gpu-api to also show GPU-API/runtime calls)\n")
    )
    parts.append(f"max depth: {max_depth if max_depth is not None else 'unlimited'}\n")
    parts.append("\n")

    no_anchor_ranks = []
    for i, (rank_key, rows, roots) in enumerate(ranks):
        parts.append(f"=== Rank {rank_key} ===\n")
        if gpu_per_rank is not None:
            # aggregate_per_rank() only gives total seconds, not call counts --
            # re-derive counts from the paired kernel_stats.csv directly.
            gpu_kernel_totals = _kernel_totals_with_counts(gpu_dir, i)
            found_anchor = attach_kernel_summaries(rows, roots, gpu_kernel_totals, show_gpu_api)
            if not found_anchor:
                no_anchor_ranks.append((rank_key, gpu_kernel_totals))

        children_map = build_children_map(rows)
        parts.append(format_aligned_rows(render_forest(roots, children_map, max_depth, show_gpu_api)))
        parts.append("\n")

    if no_anchor_ranks:
        parts.append("=== GPU kernels (rocprofv3) -- no launch call site found in CPU tree ===\n")
        for rank_key, gpu_kernel_totals in no_anchor_ranks:
            parts.append(f"  Rank {rank_key}:\n")
            fallback_rows = [
                (f"    {kernel_name}", count, total_sec, total_sec)  # leaf: self == total
                for kernel_name, (count, total_sec) in sorted(gpu_kernel_totals.items(), key=lambda kv: -kv[1][1])
            ]
            parts.append(format_aligned_rows(fallback_rows))
        parts.append("\n")

    parts.append(
        "Caveats:\n"
        "  - GPU-API/runtime rows are hidden by default (pass --show-gpu-api to see them) --\n"
        "    same classification as extract_CPU_hotspots.py's GPU-API/overhead bucket.\n"
        "  - Kernel-descriptor sampling artifacts (labels ending in \".kd\") are hidden by\n"
        "    default too -- rocprof-sys's own sampling sometimes attributes a GPU kernel launch\n"
        "    to its compiled kernel-descriptor symbol directly in the CPU call tree, at\n"
        "    near-zero duration, duplicating the same kernel's real device time already shown\n"
        "    under \"[GPU kernels -- rocprofv3]\" below. Pass --show-gpu-api to see them too.\n"
        "  - GPU kernel placement is a structural estimate (nearest launch-call ancestor,\n"
        "    proportionally split by launch-call count when multiple candidate sites exist),\n"
        "    NOT per-dispatch-exact -- this toolchain's text/JSON output has no per-call\n"
        "    timestamps to correlate against; only the binary Perfetto trace does, and that\n"
        "    has no stdlib-friendly Python parser (deferred future work, not attempted here).\n"
        "  - wall_clock-<pid>.txt is used per rank when present (exact for GOTCHA-intercepted\n"
        "    MPI calls); sampling_wall_clock-<pid>.txt is used only as a whole-file fallback\n"
        "    for a rank with no wall_clock file at all -- the two are never spliced together.\n"
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
    parser.add_argument("--show-gpu-api", dest="show_gpu_api", action="store_true",
                         help="also show GPU-API/runtime calls (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-"
                              "prefixed) instead of hiding them")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "calltree.txt")
    write_report(args.output_dir, dest, max_depth=args.max_depth, show_gpu_api=args.show_gpu_api)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
