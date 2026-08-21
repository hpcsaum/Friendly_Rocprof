"""Reconstructs a flat, parent-linked row list -- the same shape
stage4_rocprofsys_common.caller_chains_for_label() already consumes -- directly from a *rendered*
calltree.txt's text, for a tool that only has a saved report on disk, not the raw profiling
directory that produced it.

Scope: parsing only. Has no opinion on what the reconstructed rows are used for (ancestor-chain
lookups, or anything else a future tool needs from a rendered tree) -- that's the caller's job.
Exploits stage5_tree_render.py's own exact, fixed-width rendering rules directly (REPORT_HEADERS'
column widths, format_aligned_rows()'s label-then-numeric-columns layout,
wrap_leading_labels()'s character-level line-wrapping) rather than a generic/fuzzy text scan, so
this only ever needs to be as forgiving as that renderer's own output actually is.

Functions: flat_rows_from_calltree_text().
"""

import re

from stage5_tree_render import REPORT_HEADERS

_SUFFIX_WIDTH = sum(2 + width for _name, width, _fmt in REPORT_HEADERS)
_MARKER_SUFFIX = "more node(s) hidden below this point, raise --max-depth to see them)"
# One connector level (render_node()'s own `prefix` unit) is exactly 4 characters, either a real
# ancestor's vertical bar ("│   ") or blank continuation space ("    ") once that ancestor's own
# subtree has no more siblings below it -- followed by this row's own "├── "/"└── " connector.
_CONNECTOR_RE = re.compile(r"^(?:│   |    )*(?:├── |└── )")


def _is_data_line(line):
    """True if line's final _SUFFIX_WIDTH characters parse as len(REPORT_HEADERS) right-aligned
    numeric (or "-") cells, in order -- format_aligned_rows()'s own exact suffix shape for a real
    tree row's first physical line. A hard-wrapped label's later physical lines carry no such
    suffix at all (pure continuation text, see wrap_leading_labels()), so this is also exactly how
    those are told apart from a genuine row."""
    if len(line) <= _SUFFIX_WIDTH:
        return False
    suffix = line[-_SUFFIX_WIDTH:]
    pos = 0
    for _name, width, _fmt in REPORT_HEADERS:
        chunk = suffix[pos:pos + 2 + width]
        pos += 2 + width
        cell = chunk[2:].strip()
        if cell != "-":
            try:
                float(cell)
            except ValueError:
                return False
    return True


def flat_rows_from_calltree_text(text):
    """Parses the first indented call-tree block found in text (bounded by the header line naming
    every REPORT_HEADERS column, and the first blank line after it -- render_report()'s own
    convention of a blank line between a tree block and whatever notes/footer follow it) into a
    flat list of {"label", "parent"} dicts -- "parent" another dict in the same list, or None for
    a root, exactly caller_chains_for_label()'s expected input shape. Returns [] if no such block
    is found at all (e.g. an empty or unrelated text file).

    The default (not --show-all-internals) rendering is exactly as parseable as any other: pruned
    noise only ever removes a node's own subtree, and wrapper-splicing/MPI-collapsing only affect
    what's within/below a node, never a real ancestor above it -- nothing about this parser
    requires or assumes any particular --show-* flag was used to produce the text.
    """
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "calls" in line and "total-avg(s)" in line:
            header_idx = i
            break
    if header_idx is None:
        return []

    logical_rows = []
    for line in lines[header_idx + 1:]:
        if not line:
            break
        if _is_data_line(line):
            logical_rows.append(line[:-_SUFFIX_WIDTH].rstrip())
        elif line.endswith(_MARKER_SUFFIX):
            continue  # "... (N more node(s) hidden ...)" -- not a real row, not a continuation
        elif logical_rows:
            logical_rows[-1] += line

    flat = []
    stack = []  # [(depth, node), ...], shallowest first
    for label_part in logical_rows:
        m = _CONNECTOR_RE.match(label_part)
        if m:
            depth = len(m.group(0)) // 4
            label = label_part[m.end():]
        else:
            depth = 0
            label = label_part

        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None

        node = {"label": label, "parent": parent}
        flat.append(node)
        stack.append((depth, node))

    return flat
