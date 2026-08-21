"""Shared tree-merge engine for rocprof-sys call-tree tools -- generic across both the sample
(timemory text-table) and trace (Perfetto trace-CSV) pipelines this project supports.

Scope: merging N per-rank call trees into one by tree position, flattening a merged tree back into
a parent-linked list, walking caller chains, and computing avg/std_dev/min/max/calls statistics per
node across ranks. Every function here operates purely on rows/nodes carrying
"label"/"parent"/"count"/"self_sum"/"sum"/"tags"/"structural_drop_tags" (or, for a merged node,
"children"/"per_rank"/"static_children") -- nothing reads a file, and nothing assumes which stage1
produced the rows (`stage4_rocprofsys_sample_tree.py`'s `load_rank_trees()` and
`stage4_rocprofsys_trace_aggregate.py`'s `build_rank_aggregate()` are both current consumers). Has
no opinion on which nodes get rendered or how (see `stage5_tree_render.py`), or on how a merged
node's real GPU-kernel/launch-site attachment is computed (each pipeline does that its own way
before or independently of merging here).

Also owns two small, generic label-selection helpers (expand_labels_with_ancestors(),
resolve_kernel_owners()) built directly on top of the primitives above -- reusable by any future
tool that needs "these labels plus enough of their real ancestry" or "these GPU kernel names'
real CPU owners," not specific to any one selection tool's own reason for wanting them.

Functions: merge_rank_trees(), flatten_tree(), caller_chains_for_label(), aggregate_node_stats(),
make_node_values(), kernel_owner_label(), expand_labels_with_ancestors(), resolve_kernel_owners().
"""

import re

from stage4_rank_merge_math import stats_across_ranks

# Cray's Fortran frontend embeds the enclosing subroutine directly in a kernel's name:
# "<subroutine>$<module>_mod_$ck_L<line>_<n>[_cce$noloop$form]".
_CRAY_CK_MARKER = "$ck_"

# Every other compiler this project targets (AMD's amdclang/amdflang, Cray's own Clang-based C/C++
# frontend) instead uses the standard LLVM OpenMP-offloading kernel name shape:
# "__omp_offloading_<hex>_<hex>_<mangled-or-plain-name>_l<line>[_cce$noloop$form]".
_OMP_OFFLOAD_RE = re.compile(
    r"^__omp_offloading_[0-9a-f]+_[0-9a-f]+_(?P<mangled>.+)_l\d+(?:_cce\$noloop\$form)?$",
)
# Itanium C++ mangling of a simple (non-namespaced, non-templated) free function: "_Z<len><name>...".
_ITANIUM_RE = re.compile(r"^_Z(\d+)")
# Flang's Fortran module-procedure / bare-procedure mangling: "_QM<module>P<name>" / "_QP<name>".
_FLANG_MODULE_RE = re.compile(r"^_QM\w+?P(\w+)$")
_FLANG_BARE_RE = re.compile(r"^_QP(\w+)$")


def _demangle_omp_offload_name(mangled):
    """Best-effort recovery of the plain source function/subroutine name from the middle segment
    of an "__omp_offloading_..." kernel name. Only decodes three simple shapes -- a plain C name, a
    non-namespaced/non-templated Itanium-mangled C++ name, or a Flang-mangled Fortran name --
    anything else (a namespaced or templated C++ symbol, for instance) is returned unchanged rather
    than guessed."""
    m = _ITANIUM_RE.match(mangled)
    if m:
        n = int(m.group(1))
        rest = mangled[m.end():]
        if len(rest) >= n:
            return rest[:n]
        return mangled

    m = _FLANG_MODULE_RE.match(mangled)
    if m:
        return m.group(1)

    m = _FLANG_BARE_RE.match(mangled)
    if m:
        return m.group(1)

    return mangled


def kernel_owner_label(kernel_name):
    """Recovers the real CPU-side owning function/subroutine name embedded in a GPU kernel's own
    name, so it can be matched directly against a CPU call-tree node carrying that same label,
    instead of guessed via nearest launch-call ancestor. Decodes two conventions, tried in order:

    1. Cray Fortran's "$ck_" marker: "<subroutine>$<module>_mod_$ck_L<line>_<n>[_cce$noloop$form]"
       -- the part before "$ck_" is exactly the "<subroutine>$<module>_mod_" label the real CPU
       tree node carries.
    2. The generic LLVM OpenMP-offloading shape every other compiler this project targets uses:
       "__omp_offloading_<hex>_<hex>_<mangled-or-plain-name>_l<line>[_cce$noloop$form]" -- the
       middle segment is demangled (see _demangle_omp_offload_name()) to recover the plain name.

    A kernel name matching neither convention is returned unchanged -- unmatchable by name, left
    for whatever structural fallback the calling pipeline uses instead."""
    if _CRAY_CK_MARKER in kernel_name:
        return kernel_name.split(_CRAY_CK_MARKER, 1)[0]

    m = _OMP_OFFLOAD_RE.match(kernel_name)
    if m:
        return _demangle_omp_offload_name(m.group("mangled"))

    return kernel_name


def expand_labels_with_ancestors(flat, labels, depth):
    """Given flat (any parent-linked row list, e.g. flatten_tree()'s output) and a set of target
    labels, returns the set of ADDITIONAL labels found within `depth` real ancestor levels of any
    of them -- the immediate parent at depth 1, the parent's parent at depth 2, and so on. Uses
    caller_chains_for_label() per label (every distinct root-to-target chain, since the same label
    can be called from more than one real position), taking each chain's nearest `depth` entries
    before the target itself. `depth` <= 0 returns an empty set -- no ancestors pulled in. A label
    with no matching row anywhere contributes nothing, same "absence is a real fact, not an error"
    precedent caller_chains_for_label() already establishes.

    Generic across every consumer of this codebase's flat/parent-linked row shape -- no
    hotspot/instrumentation concept here, just "give me real, nearby ancestors of these labels."
    """
    if depth <= 0:
        return set()

    added = set()
    for label in labels:
        for chain in caller_chains_for_label(flat, label):
            # chain is root-first, ending with the target itself -- its nearest `depth`
            # ancestors are the `depth` entries immediately before that last one.
            for node in chain[:-1][-depth:]:
                if node["label"] not in labels:
                    added.add(node["label"])
    return added


def resolve_kernel_owners(kernel_labels):
    """Given an iterable of GPU kernel names, returns the set of real CPU owner names
    successfully decoded via kernel_owner_label() -- a kernel name matching neither convention
    that function recognizes contributes nothing (kernel_owner_label() returns it unchanged in
    that case, which is exactly how "no owner found" is represented; filtered out here rather
    than added as a bogus self-referential "owner")."""
    owners = set()
    for kernel_label in kernel_labels:
        owner = kernel_owner_label(kernel_label)
        if owner != kernel_label:
            owners.add(owner)
    return owners


def merge_rank_trees(ranks, label_key="label"):
    """Merges N per-rank call trees (as returned by a tool's own
    load_rank_trees(): a list of (rank_key, rows, roots) tuples, each rows
    entry carrying "parent"/"label"/"count"/"self_sum"/"sum"/"tags"/etc.) into
    ONE call tree -- a global view instead of one tree per rank.

    label_key names the key each row's display name lives under -- "label" (the default) for the
    sample pipeline's rows, "name" for the trace pipeline's (see
    stage1_rocprofsys_trace.LABEL_KEY) -- the same parameterization
    stage3_rocprofsys_common.tag_rows() already uses for the same reason: which key holds a row's
    display name is a fact about which stage1 produced it, not something this function should
    hardcode. A merged node's own "label" key is always called "label" regardless -- it's this
    function's own output shape, not a passthrough of the input key name.

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
    (the union of every contributing rank's own tag sets for this code
    location -- these are properties of a code location, not really
    rank-dependent, so union is a safe, conservative merge), and "per_rank"
    ({rank_key: {"count", "self_sum", "sum"}}, one entry per rank that had a
    row at this exact tree position).

    A single rank (len(ranks) == 1) is not a special case -- this is exactly
    the mechanism that collapses repeated same-position calls within one
    rank too (e.g. many identical-label sibling rows under one shared parent
    object), since matching is by label-under-already-matched-parent
    regardless of how many (rank_key, rows, roots) tuples are given.
    """
    merged_roots = {}

    for rank_key, rows, roots in ranks:
        children_by_parent_id = {}
        for row in rows:
            parent = row["parent"]
            if parent is not None:
                children_by_parent_id.setdefault(id(parent), []).append(row)

        def walk(row, merged_parent, merged_siblings):
            label = row[label_key]
            merged_node = merged_siblings.get(label)
            if merged_node is None:
                merged_node = {
                    "label": label, "parent": merged_parent, "children": {},
                    "tags": set(), "structural_drop_tags": set(),
                    "per_rank": {}, "static_children": [],
                }
                merged_siblings[label] = merged_node

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
    the same shape stage1_rocprofsys_sample.parse_table_file() + stage2_rocprofsys_sample.attach_ancestry()
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


def caller_chains_for_label(rows, target_label):
    """Every distinct root-to-target ancestor chain for rows matching target_label -- the
    inverse walk of flatten_tree()/render_forest() (both go root-to-descendants): a function
    called from N different call sites returns N chains, each a list of rows from the real root
    down to (and including) the matching row itself, walked via "parent" links alone (rows is any
    flat list where each entry's "parent" is another row in the same list, or None -- normally
    flatten_tree()'s output). A label with no matching row anywhere (never sampled/instrumented,
    or spliced away as noise before merging) returns an empty list, not an error -- absence is a
    real, reportable fact for the caller, not a bug here."""
    chains = []
    for row in rows:
        if row["label"] != target_label:
            continue
        chain = [row]
        node = row["parent"]
        while node is not None:
            chain.append(node)
            node = node["parent"]
        chain.reverse()
        chains.append(chain)
    return chains


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
