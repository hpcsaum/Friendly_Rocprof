"""Stage 5 tree builder for the traced/wall_clock-based calltree tool (extract_calltree_traced.py).

Scope: this tool's own single --show-gpu-api-driven prune predicate; everything else (rank
loading, GPU-kernel pairing/attachment, rendering) is generic, shared with extract_calltree.py's
own companion module -- see stage5_tree_render.py.

Functions: build_calltree_view().
"""

from stage3_rocprofsys import make_is_pruned
from stage4_rocprofsys_tree import flatten_tree, make_node_values, merge_rank_trees
from stage5_tree_render import (
    attach_and_render_gpu_kernels,
    load_rank_trees,
    pair_gpu_per_rank,
    render_calltree_text,
)


def build_calltree_view(run_dir, cpu_dir, gpu_dir, max_depth=None, show_gpu_api=False):
    """Same role as extract_calltree.py's own build_calltree_view(), narrower parameter set
    matching this tool's single --show-gpu-api flag: calls the shared load_rank_trees() preferring
    wall_clock-<pid>.txt over sampling_wall_clock-<pid>.txt, with no postprocess step (this tool
    has nothing extra to do after tagging), and calls render_calltree_text() with no
    collapses_children (default no-op). Returns the same dict shape as the sampling tool's
    version: rank_keys, gpu_paired, tree_text, fallback_text.
    """
    ranks = load_rank_trees(cpu_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt")
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {cpu_dir!r} "
            "(expected files like wall_clock-<pid>.txt) -- nothing to render"
        )

    is_pruned = make_is_pruned(set() if show_gpu_api else {"gpu_api"})
    rank_keys = [rank_key for rank_key, _rows, _roots in ranks]
    node_values = make_node_values(rank_keys)

    merged_roots = merge_rank_trees(ranks)
    flat = flatten_tree(merged_roots)

    gpu_per_rank = pair_gpu_per_rank(gpu_dir, run_dir, rank_keys)
    fallback_text = attach_and_render_gpu_kernels(flat, gpu_per_rank, gpu_dir, rank_keys, is_pruned, node_values)
    tree_text = render_calltree_text(merged_roots, flat, max_depth, is_pruned, node_values)

    return {
        "rank_keys": rank_keys,
        "gpu_paired": gpu_per_rank is not None,
        "tree_text": tree_text,
        "fallback_text": fallback_text,
    }
