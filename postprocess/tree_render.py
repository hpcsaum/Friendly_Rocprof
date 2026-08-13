"""Tree rendering and text-table formatting for the calltree tools.

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

Functions: render_forest(), render_node(), get_children(), count_all_descendants(),
build_children_map(), format_aligned_rows().
"""

# Real right-aligned columns for both calltree tools' rendered trees -- one row per
# stage4_rocprofsys_tree.merge_rank_trees() node, averaged/load-balance-summarized
# across every rank (see stage4_rocprofsys_tree.aggregate_node_stats()), not per-rank.
# CALLS is a plain average (a function called a wildly different number of times per
# rank is unusual and would show up in SELF-AVG/SELF-MAX anyway); SELF gets the full
# avg/std_dev/min/max load-balance treatment, matching this codebase's established
# convention (extract_CPU_hotspots.compute_load_imbalance() does the same for
# self-time, not inclusive time); TOTAL is a plain average -- inclusive time is
# dominated by children's own load imbalance, which their own rows already show
# individually, so a second full breakdown here would mostly restate deeper rows
# rather than add information.
REPORT_HEADERS = [
    ("CALLS", 8, ".1f"),
    ("SELF-AVG(s)", 12, ".6f"),
    ("SELF-STD(s)", 12, ".6f"),
    ("SELF-MIN(s)", 12, ".6f"),
    ("SELF-MAX(s)", 12, ".6f"),
    ("TOTAL-AVG(s)", 13, ".6f"),
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
