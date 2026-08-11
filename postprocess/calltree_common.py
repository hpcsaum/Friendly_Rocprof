"""Shared building blocks for both calltree tools (`extract_calltree.py`, the
sampling-based one, and `extract_calltree_traced.py`, the wall_clock-based one).

Kept here rather than duplicated (unlike this codebase's usual "small
constants are duplicated across standalone tools" convention -- see
`extract_GPU_hotspots.py`'s `compute_load_imbalance` docstring) because this
is real, load-bearing tree-rendering, cross-rank merging, and kernel-
attribution math, not a handful of constants; a bug fixed in one copy but not
the other would silently diverge between the two tools. Neither tool's own
filtering rules (which categories of node get pruned/spliced/collapsed) are
hardcoded here -- each tool injects its own `is_pruned(node)`/
`collapses_children(node)` callables, so this module has no opinion on any
one tool's visibility rules. See docs/plans/14-sampling-calltree-tool.md and
docs/DEVELOPMENT_HISTORY.md for the cross-rank aggregation round.
"""

import os
import statistics

# Best-effort list of known GPU-kernel-launch entry points -- not exhaustive.
# Matched via substring/`in` (case-insensitive), not startswith/equality,
# since real symbols carry namespace qualification and demangled parameter
# signatures (e.g. "hip::hipModuleLaunchKernel(ihipModuleSymbol_t*, ...)").
# "__cray_start_acc_kernel" is Cray's compiler-generated OpenACC/offload
# launch entry point, observed in real sampled data -- other compilers'
# equivalents (e.g. LLVM OpenMP target offload) aren't included yet, only
# because none have been observed in this project's data so far.
KERNEL_LAUNCH_LABEL_SUBSTRINGS = (
    "hiplaunchkernel",
    "hipmodulelaunchkernel",
    "hipextlaunchkernel",
    "hiplaunchkernelggl",
    "hipextmodulelaunchkernel",
    "hipgraphlaunch",
    "__cray_start_acc_kernel",
)

# Real right-aligned columns for both tools' rendered trees -- one row per
# merge_rank_trees() node, averaged/load-balance-summarized across every rank
# (see aggregate_node_stats()), not per-rank. CALLS is a plain average (a
# function called a wildly different number of times per rank is unusual and
# would show up in SELF-AVG/SELF-MAX anyway); SELF gets the full
# avg/std_dev/min/max load-balance treatment, matching this codebase's
# established convention (extract_CPU_hotspots.compute_load_imbalance() does
# the same for self-time, not inclusive time); TOTAL is a plain average --
# inclusive time is dominated by children's own load imbalance, which their
# own rows already show individually, so a second full breakdown here would
# mostly restate deeper rows rather than add information.
REPORT_HEADERS = [
    ("CALLS", 8, ".1f"),
    ("SELF-AVG(s)", 12, ".6f"),
    ("SELF-STD(s)", 12, ".6f"),
    ("SELF-MIN(s)", 12, ".6f"),
    ("SELF-MAX(s)", 12, ".6f"),
    ("TOTAL-AVG(s)", 13, ".6f"),
]


def resolve_run_dirs(run_dir):
    """A "run" is one directory that may contain a rocprof-sys/ subdir (CPU
    timing) and/or a rocprofv3/ subdir (GPU kernel timing). Falls back to
    treating run_dir itself as the CPU dir when there's no rocprof-sys/
    subdir (profile_CPU_hotspots.sh's un-nested layout)."""
    cpu_subdir = os.path.join(run_dir, "rocprof-sys")
    gpu_subdir = os.path.join(run_dir, "rocprofv3")
    cpu_dir = cpu_subdir if os.path.isdir(cpu_subdir) else run_dir
    gpu_dir = gpu_subdir if os.path.isdir(gpu_subdir) else None
    return cpu_dir, gpu_dir


def is_kernel_launch(label):
    lname = label.lower()
    return any(s in lname for s in KERNEL_LAUNCH_LABEL_SUBSTRINGS)


def make_kernel_node(label, per_rank):
    """A synthetic (not from parse_table_file()/merge_rank_trees()) tree
    node -- same "per_rank"-keyed shape as a merged real node (see
    merge_rank_trees()) so render_node()/aggregate_node_stats() can treat it
    identically, plus "static_children" for its own kernel-name breakdown
    (a merged real node never has populated static_children until
    attach_kernel_summaries() adds one)."""
    return {"label": label, "per_rank": per_rank, "gpu": False, "static_children": []}


def get_children(node, children_map):
    kids = list(children_map.get(id(node), []))
    kids.extend(node.get("static_children", []))
    return kids


def nearest_visible_ancestor(row, is_pruned):
    """Walk parent links up from row past any node is_pruned() flags, to the
    ancestor that will actually be rendered -- the correct kernel-attachment
    point either way."""
    node = row["parent"]
    while node is not None and is_pruned(node):
        node = node["parent"]
    return node


def merge_rank_trees(ranks):
    """Merges N per-rank call trees (as returned by a tool's own
    load_rank_trees(): a list of (rank_key, rows, roots) tuples, each rows
    entry carrying "parent"/"label"/"count"/"self_sum"/"sum"/"gpu"/etc.) into
    ONE call tree -- a global view instead of one tree per rank.

    Matching is purely structural, by label at each tree level, walked
    top-down per rank: the same binary produces the same call structure on
    every rank, so "the node with this label under this already-matched
    parent" is a reliable identity across ranks, even though each rank's row
    objects are completely independent (parsed from separate files, never
    the same Python object). A rank missing a subtree entirely (e.g. an
    error path only one rank hit) simply contributes no per_rank entry
    there -- aggregate_node_stats() then correctly treats that rank as 0 for
    every metric at that node, the same "missing is real, not skipped" rule
    extract_CPU_hotspots.compute_load_imbalance() already uses elsewhere in
    this codebase, not reinvented here.

    Returns a list of merged root nodes. Each merged node has: "label",
    "parent" (a merged node or None -- same shape real rows use, so
    nearest_visible_ancestor() works unchanged), "children" ({label: merged
    child}, insertion-ordered by first-seen rank), "gpu"/"compiler_runtime"/
    "mpi_territory" (True if ANY contributing rank classified it as such --
    these are properties of a code location, not really rank-dependent, so
    OR-ing is a safe, conservative merge), and "per_rank"
    ({rank_key: {"count", "self_sum", "sum"}}, one entry per rank that had a
    row at this exact tree position).
    """
    merged_roots = {}

    for rank_key, rows, roots in ranks:
        children_by_parent_id = {}
        for row in rows:
            parent = row["parent"]
            if parent is not None:
                children_by_parent_id.setdefault(id(parent), []).append(row)

        def walk(row, merged_parent, merged_siblings):
            merged_node = merged_siblings.get(row["label"])
            if merged_node is None:
                merged_node = {
                    "label": row["label"], "parent": merged_parent, "children": {},
                    "gpu": False, "compiler_runtime": False, "mpi_territory": False,
                    "per_rank": {}, "static_children": [],
                }
                merged_siblings[row["label"]] = merged_node

            merged_node["gpu"] = merged_node["gpu"] or bool(row.get("gpu"))
            merged_node["compiler_runtime"] = merged_node["compiler_runtime"] or bool(row.get("compiler_runtime"))
            merged_node["mpi_territory"] = merged_node["mpi_territory"] or bool(row.get("mpi_territory"))

            entry = merged_node["per_rank"].setdefault(rank_key, {"count": 0, "self_sum": 0.0, "sum": 0.0})
            entry["count"] += row["count"]
            entry["self_sum"] += row["self_sum"]
            entry["sum"] += row["sum"]

            for child in children_by_parent_id.get(id(row), []):
                walk(child, merged_node, merged_node["children"])

        for root in roots:
            walk(root, None, merged_roots)

    return list(merged_roots.values())


def flatten_tree(roots):
    """Every merged node reachable via "children", pre-order -- the flat,
    parent-linked list build_children_map()/find_kernel_anchors() expect,
    the same shape parse_table_file() + attach_ancestry() give a single
    rank's own rows (merge_rank_trees()'s nodes carry "parent" too, for
    exactly this reason)."""
    flat = []

    def visit(node):
        flat.append(node)
        for child in node["children"].values():
            visit(child)

    for root in roots:
        visit(root)
    return flat


def aggregate_node_stats(per_rank, rank_keys):
    """{"calls_avg", "self_avg", "self_std", "self_min", "self_max",
    "total_avg"} for one merged (or synthetic kernel) node, across every
    rank in rank_keys -- NOT just the ranks present in per_rank: a rank
    missing from per_rank genuinely spent 0 time/0 calls here, and counts as
    0 in every statistic (see merge_rank_trees()'s docstring) rather than
    being omitted, which would understate real load imbalance. std_dev uses
    statistics.pstdev() (population, not sample) -- same choice
    extract_CPU_hotspots.compute_load_imbalance() already made, kept
    consistent here rather than reinvented.
    """
    counts = [per_rank.get(rk, {}).get("count", 0) for rk in rank_keys]
    selfs = [per_rank.get(rk, {}).get("self_sum", 0.0) for rk in rank_keys]
    totals = [per_rank.get(rk, {}).get("sum", 0.0) for rk in rank_keys]
    return {
        "calls_avg": statistics.mean(counts) if counts else 0.0,
        "self_avg": statistics.mean(selfs) if selfs else 0.0,
        "self_std": statistics.pstdev(selfs) if selfs else 0.0,
        "self_min": min(selfs) if selfs else 0.0,
        "self_max": max(selfs) if selfs else 0.0,
        "total_avg": statistics.mean(totals) if totals else 0.0,
    }


def make_node_values(rank_keys):
    """Returns a node_values(node) callable (see render_node()) bound to a
    fixed list of rank keys -- the same function works for a real merged
    node or a synthetic kernel node (make_kernel_node()), since both carry
    the same "per_rank" shape."""
    def node_values(node):
        stats = aggregate_node_stats(node["per_rank"], rank_keys)
        return (
            stats["calls_avg"], stats["self_avg"], stats["self_std"],
            stats["self_min"], stats["self_max"], stats["total_avg"],
        )
    return node_values


def find_kernel_anchors(rows, is_pruned):
    """Returns {id(anchor_row): [anchor_row, launch_call_weight]} for every
    distinct visible ancestor a kernel-launch row's nearest_visible_ancestor()
    resolves to, weighted by that row's own call count (summed across every
    rank that has it -- rows here are merge_rank_trees() output). A
    launch-family row with no visible ancestor at all (vanishingly rare --
    would mean the launch call is itself an un-rooted node) contributes to
    neither this dict nor the no-anchor fallback; skipped rather than
    mis-attributed.

    This is the coarse, STRUCTURAL fallback used only for a kernel that
    kernel_owner_label() couldn't place by name (see attach_kernel_summaries())
    -- guessing from proximity in the call tree, not from what the compiler
    actually named the kernel after.
    """
    anchors = {}
    for row in rows:
        if not is_kernel_launch(row["label"]):
            continue
        anchor = nearest_visible_ancestor(row, is_pruned)
        if anchor is None:
            continue
        key = id(anchor)
        if key not in anchors:
            anchors[key] = [anchor, 0]
        anchors[key][1] += sum(v["count"] for v in row["per_rank"].values())
    return anchors


def kernel_owner_label(kernel_name):
    """Cray's OpenACC/HIP-offload kernel naming embeds the enclosing Fortran
    subroutine -- the kernel's real "caller", from the compiler's own
    perspective -- directly in the kernel name:
    "<subroutine>$<module>_mod_$ck_L<line>_<n>[_cce$noloop$form]". The part
    before "$ck_" is exactly the same "<subroutine>$<module>_mod_" label a
    real CPU call-tree node for that subroutine carries, confirmed against
    real data (every kernel's owner subroutine showed up as its own sampled
    CPU node) -- so it can be matched directly against the CPU tree instead
    of guessed via nearest launch-call ancestor. A kernel name with no
    "$ck_" marker (a different naming scheme, or a non-Cray compiler) is
    returned unchanged -- unmatchable by name, falls through to
    find_kernel_anchors()'s structural heuristic instead.
    """
    return kernel_name.split("$ck_", 1)[0]


def _attach_kernel_group(anchor_weights, kernel_names, gpu_kernel_by_rank):
    """Shared attachment step for both placement strategies below: mutates
    each anchor in anchor_weights ({id(anchor): [anchor_row, weight]}) with a
    synthetic "[GPU kernels -- rocprofv3]" static_children entry, scoped to
    just this one group of kernel names (not necessarily the whole kernel
    pool -- see attach_kernel_summaries()). Exactly one anchor -> full
    attribution. Multiple anchors -> each one's own weight is used as a
    proportional split of both the group's aggregate per-rank time and each
    individual kernel name's per-rank count/time, preserving per-rank
    granularity so the resulting synthetic nodes get real load-balance
    columns too, not just a single combined number. Falls back to an even
    split only if every site's weight is zero (shouldn't normally happen,
    avoids dividing by zero).
    """
    total_weight = sum(w for _a, w in anchor_weights.values())

    for anchor, weight in anchor_weights.values():
        fraction = (weight / total_weight) if total_weight > 0 else (1.0 / len(anchor_weights))

        if len(anchor_weights) == 1:
            label = "[GPU kernels -- rocprofv3]"
        else:
            pct = fraction * 100.0
            label = (
                f"[GPU kernels -- rocprofv3, ~{pct:.0f}% estimate: this site issued "
                f"{weight:g}/{total_weight:g} observed launch calls]"
            )

        group_per_rank = {}
        kernel_nodes = []
        for kernel_name in kernel_names:
            kernel_per_rank = {}
            for rank_key, totals in gpu_kernel_by_rank.items():
                if kernel_name not in totals:
                    continue
                count, total_sec = totals[kernel_name]
                scaled_count = count * fraction
                scaled_total = total_sec * fraction
                kernel_per_rank[rank_key] = {"count": scaled_count, "self_sum": scaled_total, "sum": scaled_total}
                group_entry = group_per_rank.setdefault(rank_key, {"count": 0.0, "self_sum": 0.0, "sum": 0.0})
                group_entry["count"] += scaled_count
                group_entry["self_sum"] += scaled_total
                group_entry["sum"] += scaled_total
            kernel_nodes.append(make_kernel_node(kernel_name, kernel_per_rank))

        kernel_nodes.sort(key=lambda n: -sum(v["sum"] for v in n["per_rank"].values()))
        group_node = make_kernel_node(label, group_per_rank)
        group_node["static_children"] = kernel_nodes
        anchor.setdefault("static_children", []).append(group_node)


def attach_kernel_summaries(rows, gpu_kernel_by_rank, is_pruned):
    """Mutates rows in place: inserts synthetic "[GPU kernels -- rocprofv3]"
    node(s) at the right place(s) in the (merged) tree. gpu_kernel_by_rank is
    {rank_key: {kernel_name: (count, total_seconds)}} -- per-rank, so the
    synthetic nodes this creates carry real per-rank breakdowns and get
    proper load-balance columns too, not a single pre-averaged number.
    Returns the set of kernel names that could NOT be placed anywhere (an
    empty set means everything was attached) -- the caller shows this
    remainder in its own top-level "no anchor" fallback section, instead of
    an all-or-nothing boolean.

    Two-tier placement, name match preferred over structural guessing:
    1. kernel_owner_label() extracts the enclosing subroutine the compiler
       named this kernel after. When a CPU tree node with that EXACT label
       exists in this (merged) tree, the kernel attaches there directly --
       weighted by each candidate node's own call count (summed across every
       rank) when more than one shares that label (the same subroutine
       called from multiple sites). This is precise, not a guess: the
       compiler put that name there because that's literally the subroutine
       whose source the kernel came from, confirmed against real data (every
       kernel's owner subroutine showed up as its own sampled CPU node,
       while the OLD nearest-launch-ancestor approach was scattering every
       kernel across every launch site in proportion to call counts,
       regardless of which subroutine actually contained it).
    2. Any kernel that doesn't name-match anything in this tree (no "$ck_"
       marker at all, or its owner subroutine wasn't sampled as its own
       distinct frame on any rank) falls back to find_kernel_anchors()'s
       coarser nearest-launch-call-ancestor heuristic instead.
    """
    all_kernel_names = set()
    for totals in gpu_kernel_by_rank.values():
        all_kernel_names.update(totals.keys())

    owner_groups = {}
    unmatched = set()
    for kernel_name in all_kernel_names:
        owner = kernel_owner_label(kernel_name)
        if owner != kernel_name:
            owner_groups.setdefault(owner, set()).add(kernel_name)
        else:
            unmatched.add(kernel_name)

    label_to_rows = {}
    for row in rows:
        label_to_rows.setdefault(row["label"], []).append(row)

    still_unattached = set(unmatched)
    for owner, kernel_names in owner_groups.items():
        anchors = {}
        for candidate in label_to_rows.get(owner, []):
            anchor = candidate if not is_pruned(candidate) else nearest_visible_ancestor(candidate, is_pruned)
            if anchor is None:
                continue
            key = id(anchor)
            weight = sum(v["count"] for v in candidate["per_rank"].values())
            if key not in anchors:
                anchors[key] = [anchor, 0]
            anchors[key][1] += weight

        if anchors:
            _attach_kernel_group(anchors, kernel_names, gpu_kernel_by_rank)
        else:
            still_unattached.update(kernel_names)

    if still_unattached:
        anchors = find_kernel_anchors(rows, is_pruned)
        if anchors:
            _attach_kernel_group(anchors, still_unattached, gpu_kernel_by_rank)
            still_unattached = set()

    return still_unattached


def unattached_kernel_per_rank(kernel_names, gpu_kernel_by_rank):
    """{kernel_name: per_rank} for a set of kernel names attach_kernel_summaries()
    couldn't place anywhere -- built the same way _attach_kernel_group() builds
    a real anchor's per-rank breakdown, so the caller's own "no anchor" fallback
    section gets full load-balance columns too, via make_kernel_node() +
    make_node_values(), instead of a single pre-averaged number."""
    result = {}
    for kernel_name in kernel_names:
        per_rank = {}
        for rank_key, totals in gpu_kernel_by_rank.items():
            if kernel_name not in totals:
                continue
            count, total_sec = totals[kernel_name]
            per_rank[rank_key] = {"count": count, "self_sum": total_sec, "sum": total_sec}
        result[kernel_name] = per_rank
    return result


def count_all_descendants(nodes, children_map, is_pruned):
    total = 0
    for node in nodes:
        total += 1
        kids = [k for k in get_children(node, children_map) if not is_pruned(k)]
        total += count_all_descendants(kids, children_map, is_pruned)
    return total


def render_node(node, prefix, is_last, show_connector, children_map, level, max_depth, is_pruned, node_values, out):
    """Appends (label_text, values_or_None) tuples to out -- one per rendered
    line. values_or_None is None for the "N more node(s) hidden" marker line
    (no metrics of its own). `node_values(node)` returns the CALLING tool's
    own tuple of numeric column values for a real node (e.g.
    (calls_avg, self_avg, self_std, self_min, self_max, total_avg) -- see
    format_aligned_rows()). Uses tree-drawing connectors (├── / └── / │)
    like the `tree` command; roots (show_connector=False) print flush, since
    a rank's multiple independent roots (e.g. separate OS threads) aren't
    true siblings under one shared parent -- connectors start from each
    root's own children downward.
    """
    label_text = f"{prefix}{'└── ' if is_last else '├── '}{node['label']}" if show_connector else node["label"]
    out.append((label_text, node_values(node)))

    kids = [k for k in get_children(node, children_map) if not is_pruned(k)]
    if not kids:
        return
    child_prefix = prefix + ("    " if is_last else "│   ") if show_connector else ""
    if max_depth is not None and level >= max_depth:
        hidden = count_all_descendants(kids, children_map, is_pruned)
        marker = f"{child_prefix}└── ... ({hidden} more node(s) hidden below this point, raise --max-depth to see them)"
        out.append((marker, None))
        return
    for i, kid in enumerate(kids):
        render_node(kid, child_prefix, i == len(kids) - 1, True, children_map, level + 1, max_depth, is_pruned, node_values, out)


def render_forest(roots, children_map, max_depth, is_pruned, node_values):
    """The whole forest (every root tree) as a list of (label_text,
    values_or_None) tuples, ready for format_aligned_rows()."""
    out = []
    for root in roots:
        if not is_pruned(root):
            render_node(root, "", False, False, children_map, 0, max_depth, is_pruned, node_values, out)
    return out


def build_children_map(rows, collapses_children=lambda row: False):
    """{id(parent): [child rows]}. A parent for which collapses_children()
    returns True (e.g. a rank's first MPI-library frame, when MPI internals
    are being collapsed) contributes no children of its own -- its real
    subtree still exists in `rows`, just never reachable via this map."""
    children = {}
    for row in rows:
        parent = row["parent"]
        if parent is not None and not collapses_children(parent):
            children.setdefault(id(parent), []).append(row)
    return children


def format_aligned_rows(rows, headers):
    """Real right-aligned numeric columns under one header, sized to this
    block's longest label -- not a "[calls=.../self=...]" string repeated on
    every line. `headers` is a list of (name, width, format_spec) tuples,
    e.g. [("CALLS", 8, ".1f")]. Each row is (label_text, values) where
    values is a tuple with one entry per header (a number, or None to render
    that single cell as "-"), or values is `None` entirely for a marker row
    with no metrics at all (printed as plain text, e.g. the "N more node(s)
    hidden" line). Returns "" for an empty block (no header printed with
    nothing under it)."""
    data_rows = [r for r in rows if r[1] is not None]
    if not data_rows:
        return ""

    label_width = max(len(text) for text, _values in data_rows)
    header_line = f"{'':<{label_width}}" + "".join(f"  {name:>{width}}" for name, width, _fmt in headers)
    lines = [header_line]
    for text, values in rows:
        if values is None:
            lines.append(text)
            continue
        cells = []
        for value, (_name, width, fmt) in zip(values, headers):
            cell = f"{value:{fmt}}" if value is not None else "-"
            cells.append(f"{cell:>{width}}")
        lines.append(f"{text:<{label_width}}  " + "  ".join(cells))
    return "\n".join(lines) + "\n"
