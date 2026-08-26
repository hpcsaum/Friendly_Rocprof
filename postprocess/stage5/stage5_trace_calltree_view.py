"""Stage 5 tree builder for the trace-CSV calltree tool (extract_trace_calltree.py).

Scope: this tool's own --show-*-driven prune/collapse predicates and its wrapper-noise
postprocess step; everything else (per-rank aggregate fetching/caching in
stage4_rocprofsys_trace_aggregate.py, cross-rank merge in stage4_rocprofsys_trace_tree.py,
rendering in stage5_tree_render.py) is generic. Simpler than stage5_calltree_view.py's own
build_calltree_view(): there's no GPU-kernel-fallback section here at all -- the corr_id join
already placed every kernel dispatch onto its exact launch site inside
stage4_rocprofsys_trace_aggregate.build_rank_aggregate(), with no ambiguity left needing a
fallback.

When a --time-range is active (stage6_time_range_config.active_ranges()), this also ORs
stage4_rocprofsys_common.make_zero_time_pruned() into the tag-based is_pruned predicate, so a
subtree with no overlap anywhere within it is cut from the tree while any ancestor chain to a
surviving descendant stays visible -- the range itself was already applied to every row's
self_sum/sum by the time merge_ranks() returns, so this is purely about which already-clipped
nodes get hidden from the rendered tree, not a second filtering pass over the data.

Functions: strip_wrapper_noise(), build_calltree_view().
"""

from stage3_rocprofsys_trace import (
    make_collapses_children,
    make_is_pruned,
    remove_tagged_subtrees,
    splice_by_tag,
)
from stage4_rocprofsys_common import flatten_tree, make_node_values, make_zero_time_pruned
from stage4_rocprofsys_trace_tree import merge_ranks
from stage5_tree_render import render_calltree_text
import stage6_time_range_config


def _or_predicates(a, b):
    """is_pruned(node) = a(node) or b(node) -- a plain named function rather than a lambda so the
    combined predicate stays easy to read at its one call site below."""
    return lambda node: a(node) or b(node)


def strip_wrapper_noise(rows):
    """The postprocess step passed to merge_ranks() unless show_rocprofsys_internals: drops any
    whole wrapper_branch_noise-contaminated sibling subtree, then splices out wrapper_noise-tagged
    rows themselves (reparenting their children, discarding their own self-time -- fold=False
    preserves that discard-self-time behavior), then splices out other-tagged rows the same way
    but folding their self-time into the new parent instead of discarding it -- the same three
    steps stage5_calltree_view.strip_wrapper_noise() already applies for the sample pipeline,
    reusing the identical tag names and primitives."""
    rows = remove_tagged_subtrees(rows, {"wrapper_branch_noise"})
    rows = splice_by_tag(rows, "wrapper_noise", fold=False)
    return splice_by_tag(rows, "other", fold=True)


def build_calltree_view(rank_inputs, cache_dir=None, max_depth=None, show_gpu_api=False,
                         show_rocprofsys_internals=False, show_mpi_internals=False,
                         show_compiler_runtime=False):
    """Builds this tool's own prune/collapse predicates from its four --show-* flags and its own
    wrapper-noise postprocess step (strip_wrapper_noise(), applied per rank before the cross-rank
    merge, active unless show_rocprofsys_internals), merges into one aggregated tree
    (stage4_rocprofsys_trace_tree.merge_ranks()), then delegates to the unchanged
    stage5_tree_render.render_calltree_text() for the actual tree text. Returns a dict: rank_keys,
    tree_text -- no fallback_text key at all (see module docstring)."""
    postprocess = None if show_rocprofsys_internals else strip_wrapper_noise
    merged_roots = merge_ranks(rank_inputs, cache_dir=cache_dir, postprocess=postprocess)

    prune_tags = (
        ({"gpu_api"} if not show_gpu_api else set())
        | ({"compiler_runtime_noise"} if not show_compiler_runtime else set())
    )
    is_pruned = make_is_pruned(prune_tags)
    if stage6_time_range_config.active_ranges():
        is_pruned = _or_predicates(is_pruned, make_zero_time_pruned(merged_roots))
    collapses_children = make_collapses_children({"mpi_territory"} if not show_mpi_internals else set())

    rank_keys = sorted({rank_key for rank_key, _csv_paths in rank_inputs})
    node_values = make_node_values(rank_keys)

    flat = flatten_tree(merged_roots)
    tree_text = render_calltree_text(merged_roots, flat, max_depth, is_pruned, node_values, collapses_children)

    return {"rank_keys": rank_keys, "tree_text": tree_text}
