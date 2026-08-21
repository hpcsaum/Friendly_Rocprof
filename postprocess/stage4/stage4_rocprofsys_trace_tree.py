"""Stage 4 (cross-rank tree merge) for rocprof-sys's Perfetto trace-CSV calltree view.

Scope: given N per-rank canonical aggregates (each from
stage4_rocprofsys_trace_aggregate.get_rank_aggregate() -- fresh or cached, indistinguishable here),
merges them into one call tree across ranks. Genuinely thin: no kernel-attachment step of its own
-- corr_id joining already happened per-rank inside build_rank_aggregate(), so by the time this
runs, kernel-dispatch nodes are already ordinary children. No file discovery of its own either --
rank_inputs is caller-supplied, explicit (stage4_rocprofsys_trace_ranks.discover_ranks() is the
actual multi-rank file-discovery convention every current caller uses; this module stays
unopinionated about how rank_inputs was built). Feeds the unchanged stage5_tree_render.py
rendering functions directly, same as the sample pipeline's calltree tools already do.

Functions: merge_ranks().
"""

from stage4_rocprofsys_common import merge_rank_trees
from stage4_rocprofsys_trace_aggregate import get_rank_aggregate


def merge_ranks(rank_inputs, cache_dir=None, postprocess=None):
    """rank_inputs is a list of (rank_key, csv_paths) tuples, explicit and caller-supplied. Fetches
    each rank's aggregate via get_rank_aggregate(), recomputes each rank's own roots (parent is
    None), and calls merge_rank_trees() a second time, across ranks -- the same function, unchanged,
    now merging already-deduped per-rank rows instead of raw ones. Returns the merged root nodes,
    ready for stage5_tree_render.py's rendering functions.

    postprocess, if given, is called as postprocess(rows) on each rank's own aggregate right after
    get_rank_aggregate() returns it and before that rank's roots are recomputed -- the same contract
    stage4_rocprofsys_sample_tree.load_rank_trees()'s own postprocess parameter already has. This is
    where a tool's own tag-driven splice/prune runs, per rank, before the cross-rank merge --
    get_rank_aggregate()'s on-disk cache itself is never touched by this: every call either decodes
    a fresh set of row dicts from the cache file or builds a fresh set from the raw CSV, so editing
    the returned rows here never feeds back into what gets cached or into any other caller's own
    copy."""
    ranks = []
    for rank_key, csv_paths in rank_inputs:
        rows = get_rank_aggregate(csv_paths, rank_key, cache_dir=cache_dir)
        if postprocess is not None:
            rows = postprocess(rows)
        roots = [row for row in rows if row["parent"] is None]
        ranks.append((rank_key, rows, roots))
    return merge_rank_trees(ranks)
