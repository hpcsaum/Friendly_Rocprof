"""Stage 5 tree rendering, shared by both calltree tools.

Scope: drawing an indented, tree-connector-style call tree (render_forest()/
render_node()) and right-aligning it into fixed-width text columns
(format_aligned_rows()). Has no opinion on what a node's own numeric values mean or
how they were computed -- callers supply a node_values(node) callable (see
stage4_rocprofsys_sample_tree.py's make_node_values()) and their own is_pruned()/
collapses_children() predicates. Every node render_forest() walks is expected to
carry a "static_children" list -- get_children() reads that key generically,
without importing anything from stage4_rocprofsys_sample_tree.py, but it's that module's
merge_rank_trees()/make_kernel_node()/_attach_kernel_group() that actually populate
it; a real but indirect (data-shape, not call-graph) coupling worth knowing about
when changing either side.

render_gpu_kernel_fallback() is the rendering half of GPU-kernel attachment: the actual
attachment (mutating the tree, re-deriving per-rank kernel counts) is
stage4_rocprofsys_sample_tree.attach_gpu_kernels() -- this module only turns whatever that
couldn't place anywhere into the "=== GPU kernels ... ===" fallback table text, the same
rendering-only role every other function here has. Each calltree tool's own
stage5_calltree_view.py/stage5_wallclock_calltree_view.py companion module supplies only
what's genuinely tool-specific (its own prune/collapse predicates, and, for the
sampling tool, its wrapper-noise postprocess step) and calls stage4's loading/merging/
attachment functions plus this module's rendering functions to do the rest.

Functions: render_forest(), render_node(), get_children(), count_all_descendants(),
build_children_map(), wrap_leading_labels(), format_aligned_rows(),
render_gpu_kernel_fallback(), render_calltree_text(), aggregation_note(), tree_view_note().
"""

from stage4_rocprofsys_sample_tree import make_kernel_node, unattached_kernel_per_rank

# Real right-aligned columns for both calltree tools' rendered trees -- one row per
# stage4_rocprofsys_sample_tree.merge_rank_trees() node, averaged/load-balance-summarized
# across every rank (see stage4_rocprofsys_sample_tree.aggregate_node_stats()), not per-rank.
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


def render_gpu_kernel_fallback(unattached, gpu_kernel_by_rank, node_values):
    """Renders the '=== GPU kernels ... ===' fallback table for whatever
    stage4_rocprofsys_sample_tree.attach_gpu_kernels() couldn't place anywhere in the tree --
    the rendering half of what used to be one mixed attach-and-render function here. Returns ""
    when unattached is empty (nothing left to show), ending in exactly its own content's trailing
    newline and no more otherwise -- the caller supplies any blank line."""
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
    flags default to None (omitted from the summary entirely) for extract_wallclock_calltree.py,
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
