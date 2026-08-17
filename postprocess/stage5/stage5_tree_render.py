"""Stage 5 tree rendering, shared by both calltree tools.

Scope: drawing an indented, tree-connector-style call tree (render_forest()/
render_node()) and right-aligning it into fixed-width text columns
(format_aligned_rows()). Has no opinion on what a node's own numeric values mean or
how they were computed -- callers supply a node_values(node) callable (see
stage4_rocprofsys_tree.py's make_node_values()) and their own is_pruned()/
collapses_children() predicates. Every node render_forest() walks is expected to
carry a "static_children" list -- get_children() reads that key generically,
without importing anything from stage4_rocprofsys_tree.py, but it's that module's
merge_rank_trees()/make_kernel_node()/_attach_kernel_group() that actually populate
it; a real but indirect (data-shape, not call-graph) coupling worth knowing about
when changing either side.

Also owns the rank-loading and GPU-kernel-pairing/attachment logic shared by
extract_calltree.py and extract_calltree_traced.py -- identical (or near-identical,
modulo which glob pattern each tool prefers), so it lives here once rather than twice.
Each tool's own
stage5_calltree_view.py/stage5_calltree_traced_view.py companion module supplies only
what's genuinely tool-specific (its own prune/collapse predicates, and, for the
sampling tool, its wrapper-noise postprocess step) and calls these functions to do
the rest.

Functions: render_forest(), render_node(), get_children(), count_all_descendants(),
build_children_map(), wrap_leading_labels(), format_aligned_rows(), load_rank_trees(),
kernel_totals_with_counts(), pair_gpu_per_rank(), attach_and_render_gpu_kernels(),
render_calltree_text(), aggregation_note(), tree_view_note().
"""

import glob
import os

from stage1_rocprofsys import PID_SUFFIX_RE, parse_table_file
from stage1_rocprofv3 import parse_kernel_stats_csv
from stage2_rocprofsys import attach_ancestry
from stage3_rocprofsys import tag_rows
from stage4_rocprofsys_tree import attach_kernel_summaries, make_kernel_node, unattached_kernel_per_rank
from stage4_rocprofv3 import aggregate_per_rank

# Real right-aligned columns for both calltree tools' rendered trees -- one row per
# stage4_rocprofsys_tree.merge_rank_trees() node, averaged/load-balance-summarized
# across every rank (see stage4_rocprofsys_tree.aggregate_node_stats()), not per-rank.
# calls is a plain average (a function called a wildly different number of times per
# rank is unusual and would show up in self-avg/self-max anyway); self gets the full
# avg/std_dev/min/max load-balance treatment, matching this codebase's established
# convention (extract_CPU_hotspots.compute_load_imbalance() does the same for
# self-time, not inclusive time); total is a plain average -- inclusive time is
# dominated by children's own load imbalance, which their own rows already show
# individually, so a second full breakdown here would mostly restate deeper rows
# rather than add information.
REPORT_HEADERS = [
    ("calls", 8, ".1f"),
    ("self-avg(s)", 12, ".6f"),
    ("self-std(s)", 12, ".6f"),
    ("self-min(s)", 12, ".6f"),
    ("self-max(s)", 12, ".6f"),
    ("total-avg(s)", 13, ".6f"),
]


def get_children(node, children_map):
    kids = list(children_map.get(id(node), []))
    kids.extend(node.get("static_children", []))
    return kids


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


def wrap_leading_labels(rows, suffix_width, width=120):
    """Caps and hard-wraps the leading label column for a block whose numeric columns come AFTER
    it (suffix_width = the fixed width of everything printed after the label, e.g.
    sum(2 + col_width for each numeric column)). Returns (label_width, wrapped_rows):
    label_width is the one column width every row's label is padded to -- the longest real label,
    capped so label + suffix_width never exceeds `width` columns, which is what stops a single
    pathologically long label from dragging every other row's alignment wider (the 20-column floor
    is the same defensive minimum as stage5_table_render.wrap_trailing_label()'s). wrapped_rows is
    `rows` with each data row's text replaced by a list of 1+ label fragments -- a character-level
    cut, no word-boundary search, so concatenating every fragment in order reconstructs the
    original label exactly -- only the first of which carries that row's `values`; marker rows
    (values is None) pass through with their text untouched (still a plain string, not a list).
    Reusable as-is by any future tree-style renderer sharing this leading-label-then-numeric-
    columns layout, not just format_aligned_rows().
    """
    data_rows = [r for r in rows if r[1] is not None]
    available = max(width - suffix_width, 20)
    label_width = min(max((len(text) for text, _values in data_rows), default=0), available)
    wrapped = []
    for text, values in rows:
        if values is None or len(text) <= available:
            wrapped.append((text if values is None else [text], values))
            continue
        chunks = [text[i:i + available] for i in range(0, len(text), available)]
        wrapped.append((chunks, values))
    return label_width, wrapped


def format_aligned_rows(rows, headers):
    """Real right-aligned numeric columns under one header, sized to this
    block's longest label (capped -- see wrap_leading_labels()) -- not a
    "[calls=.../self=...]" string repeated on every line. `headers` is a
    list of (name, width, format_spec) tuples, e.g. [("CALLS", 8, ".1f")].
    Each row is (label_text, values) where values is a tuple with one entry
    per header (a number, or None to render that single cell as "-"), or
    values is `None` entirely for a marker row with no metrics at all
    (printed as plain text, e.g. the "N more node(s) hidden" line). Returns
    "" for an empty block (no header printed with nothing under it)."""
    data_rows = [r for r in rows if r[1] is not None]
    if not data_rows:
        return ""

    suffix_width = sum(2 + width for _name, width, _fmt in headers)
    label_width, wrapped_rows = wrap_leading_labels(rows, suffix_width)
    header_line = f"{'':<{label_width}}" + "".join(f"  {name:>{width}}" for name, width, _fmt in headers)
    lines = [header_line]
    for text_or_chunks, values in wrapped_rows:
        if values is None:
            lines.append(text_or_chunks)
            continue
        chunks = text_or_chunks
        cells = []
        for value, (_name, width, fmt) in zip(values, headers):
            cell = f"{value:{fmt}}" if value is not None else "-"
            cells.append(f"{cell:>{width}}")
        lines.append(f"{chunks[0]:<{label_width}}  " + "  ".join(cells))
        lines.extend(chunks[1:])
    return "\n".join(lines) + "\n"


def load_rank_trees(cpu_dir, primary_pattern, fallback_pattern, postprocess=None):
    """Per rank: parse_table_file() + attach_ancestry() + tag_rows() directly (NOT a by-label
    merge like stage4_rocprofsys_flat.scan_ranks(), which would destroy tree identity).
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


def attach_and_render_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, is_pruned, node_values,
                                   collect_into=None):
    """If gpu_per_rank is given, re-derives per-rank call counts from each rank's own
    kernel_stats.csv (kernel_totals_with_counts()), mutates flat in place to attach matched
    kernels onto the tree (stage4_rocprofsys_tree.attach_kernel_summaries()), and renders the
    '=== GPU kernels ... ===' fallback table for anything that couldn't be attached. Returns
    fallback text (empty string if gpu_per_rank is None or nothing was left unattached), ending in
    exactly its own content's newline and no more -- the caller supplies any blank line.

    collect_into is passed straight through to attach_kernel_summaries() -- omit it (the default)
    for a caller that only renders downward via static_children; pass a list for a caller that
    also needs to find an attached kernel's own node later by label (e.g. via
    stage4_rocprofsys_tree.caller_chains_for_label(), which searches a flat row list)."""
    if gpu_per_rank is None:
        return ""

    gpu_kernel_by_rank = {
        rank_key: kernel_totals_with_counts(gpu_dir, i) for i, rank_key in enumerate(rank_keys)
    }
    unattached = attach_kernel_summaries(flat, gpu_kernel_by_rank, is_pruned, collect_into=collect_into)
    if not unattached:
        return ""

    fallback_rows = [
        (f"  {kernel_name}", node_values(make_kernel_node(kernel_name, per_rank)))
        for kernel_name, per_rank in unattached_kernel_per_rank(unattached, gpu_kernel_by_rank).items()
    ]
    fallback_rows.sort(key=lambda r: -r[1][1])
    return (
        "=== GPU kernels (rocprofv3) -- no owning subroutine or launch call site found in CPU tree ===\n"
        + format_aligned_rows(fallback_rows, REPORT_HEADERS)
    )


def render_calltree_text(merged_roots, flat, max_depth, is_pruned, node_values,
                          collapses_children=lambda row: False):
    """Renders the merged tree as text via build_children_map()/render_forest()/
    format_aligned_rows() -- the one call both calltree tools make identically.
    collapses_children defaults to a no-op; only the sampling tool passes a real one (for
    MPI-internals collapsing). Ends in exactly its own content's newline and no more -- the
    caller supplies any blank line."""
    children_map = build_children_map(flat, collapses_children=collapses_children)
    return format_aligned_rows(
        render_forest(merged_roots, children_map, max_depth, is_pruned, node_values), REPORT_HEADERS,
    )


def aggregation_note():
    """Bulleted note explaining the per-rank aggregation convention every rendered tree in this
    codebase shares -- identical for both calltree tools."""
    return (
        "  - Every row is aggregated across all ranks (not one call tree per rank): calls and\n"
        "    total-avg(s) are plain averages; self gets a full avg/std_dev/min/max load-balance\n"
        "    breakdown, the same convention this codebase's other load-imbalance tables use -- a\n"
        "    rank that never reached a given node counts as 0 there, not omitted, so real\n"
        "    imbalance (e.g. a function only some ranks call) isn't hidden by averaging.\n"
    )


def tree_view_note(rank_keys, max_depth, show_gpu_api, show_rocprofsys_internals=None,
                    show_mpi_internals=None, show_compiler_runtime=None):
    """Bulleted note summarizing how this specific tree was filtered/truncated. The 3 internals
    flags default to None (omitted from the summary entirely) for extract_calltree_traced.py,
    which only has show_gpu_api; extract_calltree.py passes all 4."""
    tiers = [("GPU-API/runtime noise", show_gpu_api)]
    if show_rocprofsys_internals is not None:
        tiers.append(("rocprof-sys internals", show_rocprofsys_internals))
    if show_mpi_internals is not None:
        tiers.append(("MPI internals", show_mpi_internals))
    if show_compiler_runtime is not None:
        tiers.append(("compiler-runtime helpers", show_compiler_runtime))
    shown = ", ".join(f"{name} ({'shown' if flag else 'hidden'})" for name, flag in tiers)
    return (
        f"  - Ranks aggregated: {', '.join(rank_keys)}\n"
        f"  - Showing: {shown}\n"
        f"  - Max depth: {max_depth if max_depth is not None else 'unlimited'}\n"
    )
