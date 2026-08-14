"""Stage 4 (merge ranks / load-balance) for the tree-shaped calltree tools, plus GPU
kernel attachment.

Scope: merging N per-rank call trees into one by tree position (not by label -- see
stage4_rank_merge_math.py, used here and by the flat, by-label merge the hotspots tools use
instead), computing avg/std_dev/min/max/calls statistics per node across ranks, and
attaching real GPU kernel data (from rocprofv3) onto the CPU subroutine that actually
launched it. Has no opinion on which nodes get rendered or how, or on any one tool's
own pruning/collapsing rules -- each tool injects its own is_pruned()/
collapses_children() callables; see tree_render.py for the rendering side.

Functions: merge_rank_trees(), flatten_tree(), aggregate_node_stats(),
make_node_values(), attach_kernel_summaries(), kernel_owner_label(),
find_kernel_anchors(), unattached_kernel_per_rank(), make_kernel_node(),
nearest_visible_ancestor(), is_kernel_launch().
"""

from stage4_rank_merge_math import stats_across_ranks

# Best-effort list of known GPU-kernel-launch entry points -- not exhaustive.
# Matched via substring/`in` (case-insensitive), not startswith/equality,
# since real symbols carry namespace qualification and demangled parameter
# signatures (e.g. "hip::hipModuleLaunchKernel(ihipModuleSymbol_t*, ...)").
# "__cray_start_acc_kernel" is Cray's compiler-generated OpenACC/offload
# launch entry point, observed in real sampled data. "__tgt_target_kernel" is
# LLVM libomptarget's OpenMP-target-offload launch entry point -- confirmed in
# real test_apps HPC data under BOTH amdclang++ and Cray CCE (both link the
# same libomptarget entry point for `omp target` in this environment), sitting
# directly beneath the real launch_omp_kernel() call site in the sampled tree.
KERNEL_LAUNCH_LABEL_SUBSTRINGS = (
    "hiplaunchkernel",
    "hipmodulelaunchkernel",
    "hipextlaunchkernel",
    "hiplaunchkernelggl",
    "hipextmodulelaunchkernel",
    "hipgraphlaunch",
    "__cray_start_acc_kernel",
    "__tgt_target_kernel",
)


def is_kernel_launch(label):
    lname = label.lower()
    return any(s in lname for s in KERNEL_LAUNCH_LABEL_SUBSTRINGS)


def make_kernel_node(label, per_rank):
    """A synthetic (not from parse_table_file()/merge_rank_trees()) tree
    node -- same "per_rank"-keyed shape as a merged real node (see
    merge_rank_trees()) so tree_render.render_node()/aggregate_node_stats() can treat
    it identically, plus "static_children" for its own kernel-name breakdown
    (a merged real node never has populated static_children until
    attach_kernel_summaries() adds one). Empty "tags"/"structural_drop_tags" --
    a synthetic kernel-summary node is never itself subject to stage3_rocprofsys
    noise tagging, so it's never pruned/collapsed."""
    return {"label": label, "per_rank": per_rank, "tags": set(), "structural_drop_tags": set(), "static_children": []}


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
    entry carrying "parent"/"label"/"count"/"self_sum"/"sum"/"tags"/etc.) into
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
    child}, insertion-ordered by first-seen rank), "tags"/"structural_drop_tags"
    (the union of every contributing rank's own stage3_rocprofsys tag sets for
    this code location -- these are properties of a code location, not really
    rank-dependent, so union is a safe, conservative merge), and "per_rank"
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
                    "tags": set(), "structural_drop_tags": set(),
                    "per_rank": {}, "static_children": [],
                }
                merged_siblings[row["label"]] = merged_node

            merged_node["tags"] |= row.get("tags", set())
            merged_node["structural_drop_tags"] |= row.get("structural_drop_tags", set())

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
    parent-linked list tree_render.build_children_map()/find_kernel_anchors() expect,
    the same shape stage1_rocprofsys.parse_table_file() + stage2_rocprofsys.attach_ancestry()
    give a single rank's own rows (merge_rank_trees()'s nodes carry "parent" too, for
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
    being omitted, which would understate real load imbalance.
    """
    counts = [per_rank.get(rk, {}).get("count", 0) for rk in rank_keys]
    selfs = [per_rank.get(rk, {}).get("self_sum", 0.0) for rk in rank_keys]
    totals = [per_rank.get(rk, {}).get("sum", 0.0) for rk in rank_keys]
    self_stats = stats_across_ranks(selfs)
    return {
        "calls_avg": stats_across_ranks(counts)["avg"],
        "self_avg": self_stats["avg"],
        "self_std": self_stats["std_dev"],
        "self_min": self_stats["min"],
        "self_max": self_stats["max"],
        "total_avg": stats_across_ranks(totals)["avg"],
    }


def make_node_values(rank_keys):
    """Returns a node_values(node) callable (see tree_render.render_node()) bound to a
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
