"""Stage 4 (build + cache the canonical per-rank aggregate) for rocprof-sys's Perfetto trace-CSV
pipeline.

Scope: the ONE expensive path for one rank -- parse, synthesize count/self_sum/sum, attach
kernel-dispatch rows to their exact launch site via corr_id, further reanchor a kernel sharing a
generic launch site with every other kernel in the program onto the real CPU call instance its own
embedded owner name and the trace's timestamps identify, tag (stage3), and collapse repeated
same-position calls within the rank (build_rank_aggregate()) -- plus an on-disk cache around it
(get_rank_aggregate()) so a 10GB+ trace-CSV is parsed at most once per rank, ever. The result is a
flat, parent-linked row list in the same canonical shape stage4_rocprofsys_common.merge_rank_trees()
already produces for one rank (label/parent/tags/structural_drop_tags/count/self_sum/sum) --
tool-independent: every row and every tag stays, nothing is dropped or renamed for any particular
tool's convenience. GPU-dispatch-detail columns (grid_size, workgroup_size, ...) deliberately do
NOT survive the merge -- once N raw launches at one call site collapse into one count, there's no
single meaningful value to keep, and no current stage5 view consumes them anyway.

Functions: build_rank_aggregate(), get_rank_aggregate().

Deliberately excludes kernel-owner-name decoding itself (kernel_owner_label(),
stage4_rocprofsys_common.py) -- that's a pure string transform with no trace-format dependency,
shared with the sample pipeline.
"""

import bisect
import json
import os

from stage1_rocprofsys_trace import LABEL_KEY, attach_ancestry, parse_trace_csv
from stage3_rocprofsys_trace import tag_for_category, tag_rows
from stage4_rocprofsys_common import flatten_tree, kernel_owner_label, merge_rank_trees


def _synthesize_self_sum(rows):
    """Sets count=1, sum=dur, self_sum=dur minus the sum of *structural* children's dur (children
    as Perfetto/stage1 already resolved them via parent_slice_id) on every row. Must run before
    any corr_id-based reparenting (_attach_kernels_by_corr_id()) -- a kernel-dispatch row about to
    be reparented onto its launch site isn't nested in wall-clock time the way a true structural
    child is (the GPU dispatch typically executes concurrently with, not strictly inside, the host
    launch call), so computing self_sum afterward would wrongly subtract a concurrently-running
    kernel's duration from its launcher's own self time."""
    children_by_parent_id = {}
    for row in rows:
        parent = row["parent"]
        if parent is not None:
            children_by_parent_id.setdefault(id(parent), []).append(row)

    for row in rows:
        row["count"] = 1
        row["sum"] = row["dur"]
        children_dur = sum(child["dur"] for child in children_by_parent_id.get(id(row), []))
        row["self_sum"] = row["dur"] - children_dur


def _attach_kernels_by_corr_id(rows):
    """Reparents every untethered (parent is None) gpu_kernel-category row with a corr_id onto the
    gpu_api-category row sharing that same corr_id -- an exact structural join, simpler than the
    sample pipeline's attach_gpu_kernels()/attach_kernel_summaries() (which grafts a *synthetic*
    static_children subtree precisely because rocprofv3's kernel_stats.csv carries no real tree
    position at all; corr_id gives trace data an exact one, so the dispatch row just becomes a
    normal child -- no proportional weight-splitting needed). Zero matches leaves the row as its
    own root, no guessing, matching stage1_rocprofsys_trace.attach_ancestry()'s own "don't guess a
    parent" precedent. More than one match warns once (count-based) and leaves the row unattached
    -- guessing would defeat the entire point of an exact join.

    Classifies rows by tag_for_category() directly rather than by row["tags"] -- this runs before
    tag_rows() (see build_rank_aggregate()'s ordering note) so every kernel-dispatch row is
    reparented before tag_rows() ever sees it as an untethered "root," which would otherwise get
    compared as a sibling of the real CPU thread root(s) by tag_rows()'s own sibling-group
    derivation and wrongly flag that thread root's entire subtree for removal."""
    launch_sites_by_corr_id = {}
    for row in rows:
        if row.get("corr_id") is not None and tag_for_category(row.get("category")) == "gpu_api":
            launch_sites_by_corr_id.setdefault(row["corr_id"], []).append(row)

    ambiguous_count = 0
    for row in rows:
        if row["parent"] is not None or tag_for_category(row.get("category")) != "gpu_kernel":
            continue
        corr_id = row.get("corr_id")
        if corr_id is None:
            continue
        candidates = launch_sites_by_corr_id.get(corr_id, [])
        if len(candidates) == 1:
            row["parent"] = candidates[0]
        elif len(candidates) > 1:
            ambiguous_count += 1

    if ambiguous_count:
        print(
            f"warning: {ambiguous_count} kernel-dispatch row(s) share a corr_id with more than "
            "one gpu_api row -- leaving unattached rather than guess which launch site is correct",
        )


def _flatten_rank_merge(merged_roots, rank_key):
    """Reshapes merge_rank_trees()'s single-rank output (per_rank keyed on rank_key) back into a
    flat, parent-linked row list with top-level count/self_sum/sum -- the "new glue function" plan
    3.1 flagged as the one real gap in "stage4 is reusable as-is": merge_rank_trees()'s own input
    and output shapes differ, so its output can't feed a second, cross-rank merge_rank_trees() call
    (or serve as a cache entry) without this reshape."""
    flat = flatten_tree(merged_roots)
    by_id = {}
    rows = []
    for node in flat:
        totals = node["per_rank"].get(rank_key, {"count": 0, "self_sum": 0.0, "sum": 0.0})
        row = {
            "label": node["label"],
            "tags": node["tags"],
            "structural_drop_tags": node["structural_drop_tags"],
            "count": totals["count"],
            "self_sum": totals["self_sum"],
            "sum": totals["sum"],
        }
        by_id[id(node)] = row
        parent_node = node["parent"]
        row["parent"] = by_id.get(id(parent_node)) if parent_node is not None else None
        rows.append(row)
    return rows


def _reanchor_kernels_by_owner_and_time(rows):
    """Reparents each gpu_kernel row onto the exact raw CPU call instance -- not merely a matching
    label, a specific instance -- that was actually running on the same thread immediately before
    the kernel dispatched, when its name embeds a recoverable owner name (kernel_owner_label()).
    A further correction on top of the exact corr_id join above, for the case OMPT-based
    instrumentation funnels every kernel launch in the whole program through the same generic
    entry points (e.g. ompt_target -> hipModuleLaunchKernel for every omp target region), leaving
    corr_id's own placement structurally uninformative even though it's exact.

    Operates on RAW, per-instance rows (parse_trace_csv() + attach_ancestry()'s output, each still
    carrying its own "ts"/"tid"), not the intra-rank-merged shape -- must run at this stage: only
    here does each instance still have the timestamp this technique depends on. A single CPU
    thread's own recorded call frames are strictly serial (one frame active at a time on that
    thread), so "the same-owner-labeled instance on the same thread that started most recently
    before this dispatch" identifies the exact real caller, not merely a plausible one -- exact,
    not an estimate, unlike splitting time across multiple label-matched candidates would be.
    Scoping to the launching thread's own tid (rather than searching every row in the rank) is
    what makes this correct without requiring the whole rank's CPU side to be single-threaded; it
    only relies on that one thread's own frames never overlapping in time, which is true for a
    normal call stack (not for, e.g., coroutines/fibers/signal-handler reentrancy on one thread --
    out of scope for this project's C/C++/Fortran + MPI/OpenMP target profile).

    A row with no "$ck_"/"__omp_offloading_..." owner name, no corr_id-resolved parent (so no
    known launching thread), or no owner-labeled candidate starting before it on that thread, is
    left exactly where _attach_kernels_by_corr_id() already put it -- no guessing."""
    candidates_by_owner_and_tid = {}
    for row in rows:
        candidates_by_owner_and_tid.setdefault((row["name"], row.get("tid")), []).append(row)
    # Sorted once per (name, tid) group, not per kernel row below -- an owner subroutine called
    # thousands of times (e.g. inside a timestep loop) would otherwise make this quadratic, since
    # many kernel dispatches share the same owner+tid group.
    starts_by_owner_and_tid = {}
    for key, candidates in candidates_by_owner_and_tid.items():
        candidates.sort(key=lambda r: r["ts"])
        starts_by_owner_and_tid[key] = [c["ts"] for c in candidates]

    for row in rows:
        if tag_for_category(row.get("category")) != "gpu_kernel":
            continue
        owner = kernel_owner_label(row["name"])
        if owner == row["name"]:
            continue
        launch_site = row["parent"]
        if launch_site is None:
            continue
        launch_tid = launch_site.get("tid")

        key = (owner, launch_tid)
        starts = starts_by_owner_and_tid.get(key)
        if not starts:
            continue
        idx = bisect.bisect_right(starts, row["ts"]) - 1
        if idx < 0:
            continue
        row["parent"] = candidates_by_owner_and_tid[key][idx]


def build_rank_aggregate(csv_paths, rank_key):
    """The one expensive path for one rank: parse + ancestry (unchanged stage1), self time
    synthesis, the corr_id kernel join, tag_rows() (stage3), and the intra-rank merge_rank_trees()
    collapse of repeated same-position calls -- confirmed directly against its source: nothing
    about its loop structure requires more than one rank, so feeding it just this rank's own
    (rank_key, rows, roots) performs exactly the same by-id(parent)+label merge it already does
    across ranks. Returns the flat, parent-linked, tool-independent row list get_rank_aggregate()
    caches.

    tag_rows() runs AFTER both the corr_id join and the owner+time reanchor, not before -- a
    kernel-dispatch row is untethered (parent is None) until the join reparents it, and tag_rows()'s
    own sibling-group derivation (for wrapper_branch_noise) compares every untethered row at the
    top level as if they were siblings sharing one real parent. Running it before either
    reparenting step would compare a real CPU thread root against an unrelated kernel-dispatch row
    (still untethered, or not yet at its final position) as "siblings," and wrongly flag the CPU
    thread's entire subtree as contaminated -- tag_rows() must see each row's FINAL structural
    position. Both reparenting steps only need tag_for_category() (a direct, stateless lookup --
    see _attach_kernels_by_corr_id()), not the full tag_rows() pass, so this ordering costs nothing.

    _reanchor_kernels_by_owner_and_time() runs right after the corr_id join, on the still-raw,
    per-instance rows -- before the intra-rank merge, not after (see its own docstring): only at
    this stage does each instance still carry the "ts"/"tid" the technique depends on; merging
    first would collapse exactly the timestamps it needs."""
    rows = parse_trace_csv(csv_paths)
    attach_ancestry(rows)

    _synthesize_self_sum(rows)
    _attach_kernels_by_corr_id(rows)
    _reanchor_kernels_by_owner_and_time(rows)
    tag_rows(rows)

    roots = [r for r in rows if r["parent"] is None]  # recomputed AFTER reparenting above
    merged_roots = merge_rank_trees([(rank_key, rows, roots)], label_key=LABEL_KEY)
    flat_rows = _flatten_rank_merge(merged_roots, rank_key)
    return flat_rows


def _cache_path(csv_paths, rank_key, cache_dir):
    if cache_dir is None:
        first_path = csv_paths if isinstance(csv_paths, str) else csv_paths[0]
        cache_dir = os.path.dirname(first_path)
    return os.path.join(cache_dir, f"{rank_key}.agg.json")


def _encode_rows(rows):
    index_by_id = {id(row): i for i, row in enumerate(rows)}
    encoded = []
    for row in rows:
        parent = row["parent"]
        encoded.append({
            "label": row["label"],
            "tags": sorted(row["tags"]),
            "structural_drop_tags": sorted(row["structural_drop_tags"]),
            "count": row["count"],
            "self_sum": row["self_sum"],
            "sum": row["sum"],
            "parent_index": index_by_id[id(parent)] if parent is not None else None,
        })
    return encoded


def _decode_rows(encoded):
    rows = [
        {
            "label": item["label"],
            "tags": set(item["tags"]),
            "structural_drop_tags": set(item["structural_drop_tags"]),
            "count": item["count"],
            "self_sum": item["self_sum"],
            "sum": item["sum"],
            "parent": None,
        }
        for item in encoded
    ]
    for row, item in zip(rows, encoded):
        if item["parent_index"] is not None:
            row["parent"] = rows[item["parent_index"]]
    return rows


def get_rank_aggregate(csv_paths, rank_key, cache_dir=None):
    """The cache-aware wrapper every other trace-pipeline module calls instead of
    build_rank_aggregate() directly. cache_dir defaults to the directory of the first csv_paths
    entry (cached alongside the raw CSV). Cache file: <cache_dir>/<rank_key>.agg.json -- keyed on
    rank_key, not source filename(s), since one rank's input can be a single file or a
    category-partitioned set and rank_key is the one identity stable across either shape. A valid
    cache (exists, mtime >= every csv_paths entry's mtime) is decoded and returned; missing, stale,
    or unreadable (corrupt JSON, unexpected schema) falls back to a fresh build_rank_aggregate()
    call, matching stage6_run_metadata.load_json_file()'s "best-effort, never raises" convention --
    a cache-read problem should never be the reason this function fails. The freshly-built result
    is written back to the cache (silently skipped on a write failure -- the cache is a pure
    performance optimization, never required for correctness)."""
    paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
    cache_file = _cache_path(csv_paths, rank_key, cache_dir)

    if os.path.exists(cache_file):
        try:
            source_mtime = max(os.path.getmtime(p) for p in paths)
            if os.path.getmtime(cache_file) >= source_mtime:
                with open(cache_file) as f:
                    return _decode_rows(json.load(f))
        except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    rows = build_rank_aggregate(csv_paths, rank_key)
    try:
        with open(cache_file, "w") as f:
            json.dump(_encode_rows(rows), f)
    except OSError:
        pass
    return rows
