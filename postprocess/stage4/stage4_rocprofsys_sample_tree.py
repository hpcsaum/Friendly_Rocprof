"""Stage 4 (rank loading / load-balance) for the tree-shaped calltree tools, plus GPU-kernel-data
pairing and GPU kernel attachment specific to the sample (timemory text-table) pipeline.

Scope: per-rank loading (parse + ancestry + noise-tagging, load_rank_trees()) and attaching real
GPU kernel data (from rocprofv3) onto the CPU subroutine that actually launched it, via a
name-match-then-structural-proximity heuristic -- rocprofv3's kernel_stats.csv carries no tree
position of its own, so this has to guess. (The trace-CSV pipeline's own build_rank_aggregate() in
stage4_rocprofsys_trace_aggregate.py does the equivalent attachment via an exact `corr_id` join
instead -- no heuristic needed there.) The generic tree-merge/flatten/stats engine this file used
to also hold (merge_rank_trees(), flatten_tree(), caller_chains_for_label(),
aggregate_node_stats(), make_node_values()) moved to stage4_rocprofsys_common.py, since none of it
was actually sample-format-specific and the trace pipeline needs it too. Has no opinion on which
nodes get rendered or how, or on any one tool's own pruning/collapsing rules -- each tool injects
its own is_pruned()/collapses_children() callables; see stage5_tree_render.py for the rendering
side (including render_gpu_kernel_fallback(), the rendering half of what used to be one mixed
attach-and-render function here).

Functions: load_rank_trees(), kernel_totals_with_counts(), pair_gpu_per_rank(),
attach_gpu_kernels(), attach_kernel_summaries(), kernel_owner_label(), find_kernel_anchors(),
unattached_kernel_per_rank(), make_kernel_node(), nearest_visible_ancestor(), is_kernel_launch().
"""

import glob
import os

from stage1_rocprofsys_sample import PID_SUFFIX_RE, parse_table_file
from stage1_rocprofv3 import parse_kernel_stats_csv
from stage2_rocprofsys_sample import attach_ancestry
from stage3_rocprofsys_common import tag_rows
from stage4_rocprofv3 import aggregate_per_rank


def load_rank_trees(cpu_dir, primary_pattern, fallback_pattern, postprocess=None):
    """Per rank: parse_table_file() + attach_ancestry() + tag_rows() directly (NOT a by-label
    merge like stage4_rocprofsys_sample_flat.scan_ranks(), which would destroy tree identity).
    primary_pattern's file wins per rank when present; fallback_pattern is used only for a rank
    with no primary_pattern file at all -- the two are never spliced together into one tree, since
    their parent-links come from two independently-reconstructed call orders. After
    parsing/ancestry-linking/tagging, postprocess(rows) is called if given (returning the edited
    row list), for any further tool-specific row-set editing -- e.g. the sampling calltree tool's
    conditional wrapper-noise splice; it defaults to None (a no-op).

    Returns a list of (rank_key, rows, roots) tuples, one per rank, in sorted order. rows is every
    parsed row (parent/depth/thread_id/tags all set); roots is the subset with parent is None --
    every row with parent is None starts its own tree, which correctly separates multiple OS
    threads' subtrees within one rank regardless of which of the two DEPTH-numbering shapes the
    file uses (is_thread_root isn't reliably set in the DEPTH-resets-to-0 case, but parent is None
    always is).
    """
    paths_by_rank = {}
    order = []
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", primary_pattern), recursive=True)):
        m = PID_SUFFIX_RE.search(os.path.basename(path))
        rank_key = m.group(1) if m else path
        paths_by_rank[rank_key] = path
        order.append(rank_key)
    for path in sorted(glob.glob(os.path.join(cpu_dir, "**", fallback_pattern), recursive=True)):
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
        tag_rows(rows, filename=path)
        if postprocess is not None:
            rows = postprocess(rows)
        roots = [r for r in rows if r["parent"] is None]
        result.append((rank_key, rows, roots))
    return result


def kernel_totals_with_counts(gpu_dir, rank_index):
    """Re-parses one rank's own kernel_stats.csv directly for its Calls column, since
    aggregate_per_rank() only returns total seconds, not call counts."""
    candidates = sorted(glob.glob(os.path.join(gpu_dir, "**", "*_kernel_stats.csv"), recursive=True))
    path = candidates[rank_index]
    rows = parse_kernel_stats_csv(path)
    totals = {}
    for row in rows:
        entry = totals.setdefault(row["label"], [0, 0.0])
        entry[0] += row["count"]
        entry[1] += row["total_ns"] / 1e9
    return {k: tuple(v) for k, v in totals.items()}


def pair_gpu_per_rank(gpu_dir, run_dir, rank_keys):
    """Returns the per-rank GPU kernel total-seconds dicts if gpu_dir is given and its rank count
    matches rank_keys, else None -- printing a warning (not raising) on a mismatched rank count,
    rather than risk pairing mismatched ranks."""
    if gpu_dir is None:
        return None
    gpu_totals, gpu_scanned = aggregate_per_rank(gpu_dir)
    if gpu_scanned and len(gpu_totals) == len(rank_keys):
        return gpu_totals
    if gpu_scanned:
        print(
            f"warning: {run_dir!r}: rocprof-sys reports {len(rank_keys)} rank(s) but "
            f"rocprofv3 reports {len(gpu_totals)} -- skipping GPU kernel integration "
            "rather than risk pairing mismatched ranks",
        )
    return None

# Best-effort list of known GPU-kernel-launch entry points -- not exhaustive.
# Matched via substring/`in` (case-insensitive), not startswith/equality,
# since real symbols carry namespace qualification and demangled parameter
# signatures (e.g. "hip::hipModuleLaunchKernel(ihipModuleSymbol_t*, ...)").
# "__cray_start_acc_kernel" is Cray's compiler-generated OpenACC/offload
# launch entry point. "__tgt_target_kernel" is LLVM libomptarget's
# OpenMP-target-offload launch entry point -- the same entry point amdclang++
# and Cray CCE both link for `omp target` in this environment, sitting
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


def make_kernel_node(label, per_rank, parent=None):
    """A synthetic (not from parse_table_file()/merge_rank_trees()) tree
    node -- same "per_rank"-keyed shape as a merged real node (see
    merge_rank_trees()) so tree_render.render_node()/aggregate_node_stats() can treat
    it identically, plus "static_children" for its own kernel-name breakdown
    (a merged real node never has populated static_children until
    attach_kernel_summaries() adds one). Empty "tags"/"structural_drop_tags" --
    a synthetic kernel-summary node is never itself subject to stage3_rocprofsys_common
    noise tagging, so it's never pruned/collapsed. "parent" defaults to None (the
    shape every downward-only renderer has used until now); passing the real
    attachment point lets caller_chains_for_label() walk up through this node like
    any other row, which downward rendering itself never needed and still ignores."""
    return {
        "label": label, "parent": parent, "per_rank": per_rank,
        "tags": set(), "structural_drop_tags": set(), "static_children": [],
    }


def nearest_visible_ancestor(row, is_pruned):
    """Walk parent links up from row past any node is_pruned() flags, to the
    ancestor that will actually be rendered -- the correct kernel-attachment
    point either way."""
    node = row["parent"]
    while node is not None and is_pruned(node):
        node = node["parent"]
    return node


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
    before "$ck_" is exactly the same "<subroutine>$<module>_mod_" label the
    real CPU call-tree node for that subroutine carries -- so it can be
    matched directly against the CPU tree instead of guessed via nearest
    launch-call ancestor. A kernel name with no
    "$ck_" marker (a different naming scheme, or a non-Cray compiler) is
    returned unchanged -- unmatchable by name, falls through to
    find_kernel_anchors()'s structural heuristic instead.
    """
    return kernel_name.split("$ck_", 1)[0]


def _attach_kernel_group(anchor_weights, kernel_names, gpu_kernel_by_rank, collect_into=None):
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
    avoids dividing by zero). If collect_into is given (a list), every newly
    created node (the group wrapper plus each per-kernel-name leaf) is appended
    to it, in addition to being wired into the tree via static_children -- see
    attach_kernel_summaries()'s own collect_into for why.
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
        group_node = make_kernel_node(label, group_per_rank, parent=anchor)
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
            kernel_nodes.append(make_kernel_node(kernel_name, kernel_per_rank, parent=group_node))

        kernel_nodes.sort(key=lambda n: -sum(v["sum"] for v in n["per_rank"].values()))
        group_node["static_children"] = kernel_nodes
        anchor.setdefault("static_children", []).append(group_node)
        if collect_into is not None:
            collect_into.append(group_node)
            collect_into.extend(kernel_nodes)


def attach_kernel_summaries(rows, gpu_kernel_by_rank, is_pruned, collect_into=None):
    """Mutates rows in place: inserts synthetic "[GPU kernels -- rocprofv3]"
    node(s) at the right place(s) in the (merged) tree. gpu_kernel_by_rank is
    {rank_key: {kernel_name: (count, total_seconds)}} -- per-rank, so the
    synthetic nodes this creates carry real per-rank breakdowns and get
    proper load-balance columns too, not a single pre-averaged number.
    Returns the set of kernel names that could NOT be placed anywhere (an
    empty set means everything was attached) -- the caller shows this
    remainder in its own top-level "no anchor" fallback section, instead of
    an all-or-nothing boolean.

    collect_into, if given (a list), receives every synthetic node created this
    call (each group wrapper plus each per-kernel-name leaf) -- these nodes are
    otherwise reachable only via static_children, never members of `rows` itself,
    so a caller wanting to find one later by label (e.g.
    caller_chains_for_label(), which searches a flat row list) needs them added to
    its own pool explicitly. Omit it (the default) for every existing use, which
    only ever renders downward via static_children and never needs to.

    Two-tier placement, name match preferred over structural guessing:
    1. kernel_owner_label() extracts the enclosing subroutine the compiler
       named this kernel after. When a CPU tree node with that EXACT label
       exists in this (merged) tree, the kernel attaches there directly --
       weighted by each candidate node's own call count (summed across every
       rank) when more than one shares that label (the same subroutine
       called from multiple sites). This is precise, not a guess: the
       compiler put that name there because that's literally the subroutine
       whose source the kernel came from, unlike a structural guess by
       nearest launch-call ancestor, which has no way to know which
       subroutine actually contained the kernel and can only distribute it
       proportionally across every launch site instead.
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
            _attach_kernel_group(anchors, kernel_names, gpu_kernel_by_rank, collect_into=collect_into)
        else:
            still_unattached.update(kernel_names)

    if still_unattached:
        anchors = find_kernel_anchors(rows, is_pruned)
        if anchors:
            _attach_kernel_group(anchors, still_unattached, gpu_kernel_by_rank, collect_into=collect_into)
            still_unattached = set()

    return still_unattached


def attach_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, is_pruned, collect_into=None):
    """Mutates flat's tree in place to attach real GPU kernel data onto the CPU subroutine that
    launched it (attach_kernel_summaries()), re-deriving per-rank call counts from each rank's own
    kernel_stats.csv along the way (kernel_totals_with_counts(), since aggregate_per_rank() only
    returns total seconds, not call counts). Returns (unattached, gpu_kernel_by_rank): unattached
    is the set of kernel names attach_kernel_summaries() couldn't place anywhere (empty if
    gpu_per_rank is None or everything attached); gpu_kernel_by_rank is
    {rank_key: {kernel_name: (count, total_seconds)}}, handed back so a caller building its own
    "no anchor" fallback section (see stage5_tree_render.render_gpu_kernel_fallback()) has the
    per-rank data it needs without re-parsing kernel_stats.csv a second time.

    collect_into is passed straight through to attach_kernel_summaries() -- omit it (the default)
    for a caller that only renders downward via static_children; pass a list for a caller that
    also needs to find an attached kernel's own node later by label (e.g. caller_chains_for_label(),
    which searches a flat row list)."""
    if gpu_per_rank is None:
        return set(), {}

    gpu_kernel_by_rank = {
        rank_key: kernel_totals_with_counts(gpu_dir, i) for i, rank_key in enumerate(rank_keys)
    }
    unattached = attach_kernel_summaries(flat, gpu_kernel_by_rank, is_pruned, collect_into=collect_into)
    return unattached, gpu_kernel_by_rank


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
