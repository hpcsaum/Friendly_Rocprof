"""Stage 2 (ancestry tree) for rocprof-sys text-table rows.

Scope: reconstructing each row's place in the call tree (parent links, thread-root flags) from
the DEPTH column stage 1 parsed out. Purely structural -- it has no opinion on what any row
means or what a later stage does with the ancestry it builds; every noise-classification and
tree-rendering stage downstream depends on the "parent"/"is_thread_root" fields this module adds.

Functions: attach_ancestry().
"""


def attach_ancestry(rows):
    """Reconstruct each row's parent in the call tree from DEPTH + file order
    (rows already arrive in call-tree pre-order) via a depth-stack walk: a
    row's parent is the most recent prior row at depth-1. Mutates rows in
    place, adding "parent" (a reference to the parent row dict, or None at a
    true root) and "is_thread_root" (True when this row's thread_id differs
    from its parent's -- i.e. this row is where a new OS thread's own subtree
    begins in this file's listing, right after whatever call spawned it).
    """
    stack = []
    for row in rows:
        while stack and stack[-1]["depth"] >= row["depth"]:
            stack.pop()
        parent = stack[-1] if stack else None
        row["parent"] = parent
        row["is_thread_root"] = parent is not None and parent["thread_id"] != row["thread_id"]
        stack.append(row)
    return rows
