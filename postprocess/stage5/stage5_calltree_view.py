"""Stage 5 tree builder for the sampling-based calltree tool (extract_calltree.py).

Scope: this tool's own --show-*-driven prune/collapse predicates and its wrapper-noise
postprocess step; everything else (rank loading, GPU-kernel pairing/attachment in
stage4_rocprofsys_sample_tree.py; rendering in stage5_tree_render.py) is generic, shared with
extract_wallclock_calltree.py's own companion module.

Functions: strip_wrapper_noise(), build_calltree_view().
"""

from stage3_rocprofsys_sample import make_collapses_children, make_is_pruned, remove_tagged_subtrees, splice_by_tag
from stage4_rocprofsys_sample_tree import (
    attach_gpu_kernels,
    flatten_tree,
    load_rank_trees,
    make_node_values,
    merge_rank_trees,
    pair_gpu_per_rank,
)
from stage5_tree_render import render_calltree_text, render_gpu_kernel_fallback


def strip_wrapper_noise(rows):
    """The postprocess step passed to load_rank_trees() unless show_rocprofsys_internals:
    drops any whole wrapper_branch_noise-contaminated sibling subtree, then splices out
    wrapper_noise-tagged rows themselves (reparenting their children, discarding their own
    self-time -- fold=False preserves that discard-self-time behavior), then splices out
    other-tagged rows the same way but folding their self-time into the new parent instead of
    discarding it (other's own default treatment -- a user's --extra-noise-config-defined noise
    is real work that happened somewhere, just not worth its own row)."""
    rows = remove_tagged_subtrees(rows, {"wrapper_branch_noise"})
    rows = splice_by_tag(rows, "wrapper_noise", fold=False)
    return splice_by_tag(rows, "other", fold=True)


def build_calltree_view(run_dir, cpu_dir, gpu_dir, max_depth=None, show_gpu_api=False,
                         show_rocprofsys_internals=False, show_mpi_internals=False,
                         show_compiler_runtime=False):
    """Builds this tool's own prune/collapse predicates from its four --show-* flags and its own
    wrapper-noise postprocess step (strip_wrapper_noise(), passed to the shared load_rank_trees(),
    active unless show_rocprofsys_internals), merges into one aggregated tree
    (stage4_rocprofsys_sample_tree.merge_rank_trees), then delegates to
    stage4_rocprofsys_sample_tree's shared load_rank_trees()/pair_gpu_per_rank()/
    attach_gpu_kernels() and stage5_tree_render's render_calltree_text()/
    render_gpu_kernel_fallback() for everything that's identical to
    extract_wallclock_calltree.py's own view. Returns a dict: rank_keys, gpu_paired (bool -- the
    caller's header line needs this), tree_text, fallback_text (empty string if nothing
    unattached).
    """
    ranks = load_rank_trees(
        cpu_dir, "sampling_wall_clock-*.txt", "wall_clock-*.txt",
        postprocess=None if show_rocprofsys_internals else strip_wrapper_noise,
    )
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {cpu_dir!r} "
            "(expected files like sampling_wall_clock-<pid>.txt) -- nothing to render"
        )

    prune_tags = (
        ({"gpu_api"} if not show_gpu_api else set())
        | ({"compiler_runtime_noise"} if not show_compiler_runtime else set())
    )
    is_pruned = make_is_pruned(prune_tags)
    collapses_children = make_collapses_children({"mpi_territory"} if not show_mpi_internals else set())

    rank_keys = [rank_key for rank_key, _rows, _roots in ranks]
    node_values = make_node_values(rank_keys)

    merged_roots = merge_rank_trees(ranks)
    flat = flatten_tree(merged_roots)

    gpu_per_rank = pair_gpu_per_rank(gpu_dir, run_dir, rank_keys)
    unattached, gpu_kernel_by_rank = attach_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, is_pruned)
    fallback_text = render_gpu_kernel_fallback(unattached, gpu_kernel_by_rank, node_values)
    tree_text = render_calltree_text(merged_roots, flat, max_depth, is_pruned, node_values, collapses_children)

    return {
        "rank_keys": rank_keys,
        "gpu_paired": gpu_per_rank is not None,
        "tree_text": tree_text,
        "fallback_text": fallback_text,
    }
