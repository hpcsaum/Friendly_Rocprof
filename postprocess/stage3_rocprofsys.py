"""Shared noise-classification engine for rocprof-sys call-tree/flat-scan tools.

Scope: turning a flat list of rows (each carrying "label" and a "parent" back-reference, the
shape stage2_rocprofsys.attach_ancestry() produces) into a per-row set of noise TAGS, plus a
small set of generic, tag-driven tree-surgery primitives. Nothing in this module reads a
particular tool's own filtering rules or writes report output -- it has no opinion on which tags
a given tool treats as noise or what "noise" should become (drop it, hide its children, merge it
into its parent, route it to a different table) -- each calling tool supplies its own {tag: action}
map for that.

Philosophy: classification and treatment are separate. A tag is a fact about a row ("this row's
label matches wrapper_noise"); an action is a decision about what a tool does with that fact
("splice it out and fold its self-time into its parent" vs. "drop it outright") -- the same tag
can get a different action in a different tool, without re-deriving the classification.

Tag definitions (scope: which rows a pattern is allowed to look at) come from a JSON file --
usually default_noise_patterns.json, see stage6_noise_config.load_default_patterns() -- not from
Python constants, so new noise patterns don't require a code change. tag_rows() itself falls back
to stage6_noise_config.tag_defs() (the process-wide resolved patterns, bundled defaults optionally
customized by a user's --extra-noise-config/$FRIENDLY_ROCPROF_NOISE_CONFIG file) whenever a caller
doesn't pass its own tag_defs explicitly. Each tag definition may combine:
  - "prefixes"/"substrings"/"suffixes": label-matching rules (all case-insensitive, checked
    against the row's own label only).
  - "filename_substrings": matched case-insensitively against the basename of the file a whole
    batch of rows came from (see tag_rows()'s `filename` argument) -- a fact about where the data
    came from rather than what a row's own label says, but mechanically the same kind of match as
    the three above, just against a different string. True for one row in a file means true for
    every row in that file.
  - "ancestor_for_thread_roots": true -- a thread-root row (row["is_thread_root"]) that doesn't
    match by its own label also carries this tag if ANY ancestor's label does.
  - "first_real_descendant_skip_tag": <other tag name> -- an untethered root (row["parent"] is
    None) that doesn't already carry this tag inherits it from the first descendant reached by
    walking down through its first child at each level, skipping any child that itself carries
    the named skip tag (e.g. skip wrapper frames to find the first real content).
  - "sibling_group_source_tag": <other tag name> -- a DERIVED tag with no patterns of its own.
    Computed by comparing, across every group of rows sharing one parent, whose subtree contains a
    match for the source tag: a sibling's whole subtree is flagged (via row["structural_drop_tags"],
    not row["tags"] -- see tag_rows()) only when at least one sibling in the group is completely
    clean and the flagged sibling's OWN label doesn't itself match the source tag (that simpler
    case is already covered by the source tag's own self-match).

Functions: tag_rows(), remove_tagged_subtrees(), splice_by_tag(), make_collapses_children(),
make_is_pruned().
"""

import os

from stage6_noise_config import tag_defs as _stage6_tag_defs


def _label_matches(label, tag_def):
    lname = label.lower()
    if lname.startswith(tuple(tag_def.get("prefixes", ()))):
        return True
    if any(s in lname for s in tag_def.get("substrings", ())):
        return True
    if lname.endswith(tuple(tag_def.get("suffixes", ()))):
        return True
    return False


def _build_children_map(rows):
    children_map = {}
    for row in rows:
        parent = row["parent"]
        if parent is not None:
            children_map.setdefault(id(parent), []).append(row)
    return children_map


def tag_rows(rows, tag_defs=None, filename=None):
    """Mutates every row in place: row["tags"] becomes a set of every tag whose pattern matched
    this row's own label, or the file it came from (see `filename` below), or (for a thread-root
    row) any ancestor's label, or (for an untethered root, parent=None) the first real descendant's
    tags. row["self_tags"] is the raw self-match set alone (label + filename hints only, before
    ancestor/first-real-descendant enrichment) -- kept separate so a caller can tell "this row's
    own identity matches" apart from "this row inherited the tag from somewhere else," which
    matters when the same tag needs a different action depending on which one fired.
    row["structural_drop_tags"] becomes a set of tags for which this row's WHOLE SUBTREE should be
    removed because of a sibling comparison -- kept separate from "tags" since it's a removal
    decision about a row, not a fact about what the row's own label looks like. IMPORTANT: this is
    only ever set on the top of a contaminated subtree, never propagated down to its descendants --
    a caller iterating rows as a flat list (not a recursive tree walk) MUST route them through
    remove_tagged_subtrees() first to actually drop the whole subtree; checking
    row["structural_drop_tags"] directly, one row at a time, only catches the top row itself.

    tag_defs is a dict as returned by stage6_noise_config.load_default_patterns() (or an
    equivalent hand-built dict for tests) -- pattern-bearing tags and sibling-group-derived tags
    may be mixed freely. Omit it (or pass None) to use this process's current
    stage6_noise_config.tag_defs() instead -- the bundled defaults, optionally customized by
    whichever --extra-noise-config a tool's main() configured for this run.

    filename, when given, is the single source file every row in `rows` was parsed from -- matched
    against each tag's own "filename_substrings" once for the whole batch, since a file-level fact
    is equally true for every row in it.
    """
    if tag_defs is None:
        tag_defs = _stage6_tag_defs()
    children_map = _build_children_map(rows)
    top_level = [row for row in rows if row["parent"] is None]

    patterned_tags = {name: td for name, td in tag_defs.items() if "sibling_group_source_tag" not in td}
    derived_tags = {name: td for name, td in tag_defs.items() if "sibling_group_source_tag" in td}

    file_matches = set()
    if filename is not None:
        fname = os.path.basename(filename).lower()
        file_matches = {
            name for name, td in patterned_tags.items()
            if any(hint in fname for hint in td.get("filename_substrings", ()))
        }

    self_match = {}
    for row in rows:
        self_match[id(row)] = file_matches | {
            name for name, td in patterned_tags.items() if _label_matches(row["label"], td)
        }

    subtree_memo = {}

    def subtree_has_match(row):
        key = id(row)
        if key in subtree_memo:
            return subtree_memo[key]
        subtree_memo[key] = set(self_match[key])  # cycles should never happen; break them defensively
        result = set(self_match[key])
        for child in children_map.get(key, []):
            result |= subtree_has_match(child)
        subtree_memo[key] = result
        return result

    for row in rows:
        subtree_has_match(row)

    ancestor_memo = {}

    def ancestor_match(row):
        key = id(row)
        if key in ancestor_memo:
            return ancestor_memo[key]
        parent = row["parent"]
        result = set() if parent is None else self_match[id(parent)] | ancestor_match(parent)
        ancestor_memo[key] = result
        return result

    for row in rows:
        ancestor_match(row)

    for row in rows:
        key = id(row)
        tags = set(self_match[key])
        if row.get("is_thread_root"):
            for name, td in patterned_tags.items():
                if td.get("ancestor_for_thread_roots") and name in ancestor_memo[key]:
                    tags.add(name)
        row["self_tags"] = set(self_match[key])
        row["tags"] = tags
        row["structural_drop_tags"] = set()

    for siblings in [top_level] + list(children_map.values()):
        if len(siblings) < 2:
            continue
        for derived_name, derived_td in derived_tags.items():
            source_tag = derived_td["sibling_group_source_tag"]
            matches = [source_tag in subtree_memo[id(s)] for s in siblings]
            if not any(matches) or all(matches):
                continue
            for sibling, matched in zip(siblings, matches):
                if matched and source_tag not in self_match[id(sibling)]:
                    sibling["structural_drop_tags"].add(derived_name)

    for row in top_level:
        for name, td in patterned_tags.items():
            skip_tag = td.get("first_real_descendant_skip_tag")
            if not skip_tag or name in row["tags"]:
                continue
            node = row
            seen = set()
            while True:
                kids = children_map.get(id(node))
                if not kids:
                    break
                child = kids[0]
                if id(child) in seen:
                    break  # defensive: a real cycle should never happen here
                seen.add(id(child))
                if skip_tag in self_match[id(child)]:
                    node = child
                    continue
                if name in child["tags"]:
                    row["tags"].add(name)
                break


def remove_tagged_subtrees(rows, tags):
    """Returns a new list with every row whose "tags" or "structural_drop_tags" intersects
    `tags`, plus its whole subtree, removed -- the shared mechanical primitive behind both the
    "prune" and "structural_drop" actions (identical removal, different reason a row qualifies)."""
    tags = set(tags)
    children_map = _build_children_map(rows)
    removed = set()

    def mark(row):
        removed.add(id(row))
        for child in children_map.get(id(row), []):
            mark(child)

    for row in rows:
        if id(row) in removed:
            continue
        if (row.get("tags", set()) | row.get("structural_drop_tags", set())) & tags:
            mark(row)

    return [row for row in rows if id(row) not in removed]


def splice_by_tag(rows, tag, fold=True):
    """Reparents every row whose "parent" chain passes through a `tag`-matched row up to the
    nearest ancestor that doesn't match, then drops the matched rows entirely -- a row whose whole
    ancestor chain matched ends up with parent=None, becoming a new root. When fold=True, each
    removed row's own self_sum/sum is added into the ancestor its children got reparented to,
    instead of being discarded; fold=False keeps today's discard-it behavior."""
    def walk_skip(parent):
        while parent is not None and tag in parent.get("tags", set()):
            parent = parent["parent"]
        return parent

    fold_totals = {}
    for row in rows:
        new_parent = walk_skip(row["parent"])
        if tag in row.get("tags", set()):
            if fold and new_parent is not None:
                totals = fold_totals.setdefault(id(new_parent), {"self_sum": 0.0, "sum": 0.0})
                totals["self_sum"] += row.get("self_sum", 0.0)
                totals["sum"] += row.get("sum", 0.0)
        else:
            row["parent"] = new_parent

    if fold:
        for row in rows:
            totals = fold_totals.get(id(row))
            if totals:
                row["self_sum"] = row.get("self_sum", 0.0) + totals["self_sum"]
                row["sum"] = row.get("sum", 0.0) + totals["sum"]

    return [row for row in rows if tag not in row.get("tags", set())]


def make_collapses_children(tags):
    """A tree_render.build_children_map()-compatible collapses_children(row) callable: True for
    a row carrying any of `tags` -- the row itself still renders, its children don't."""
    tags = set(tags)
    return lambda row: bool(tags & row.get("tags", set()))


def make_is_pruned(tags):
    """A tree_render.render_forest()-compatible is_pruned(node) callable: True for a row carrying
    any of `tags` in either "tags" or "structural_drop_tags" -- the row and its whole subtree are
    hidden."""
    tags = set(tags)
    return lambda row: bool(tags & (row.get("tags", set()) | row.get("structural_drop_tags", set())))
