"""Stage 4 (merge ranks) for rocprof-sys's flat (non-tree) CPU view.

Scope: turning per-rank timemory text tables into merged, by-label aggregated entries -- either
one global total per label (aggregate()) or one total per label per rank (aggregate_per_rank(),
what a load-imbalance table needs). Has no opinion on ranking, selection, or report formatting;
feeds stage5_cpu_hotspots_table.py and stage5_load_imbalance_table.py (via
stage5_table_render.select_entries()/render_table()) and stage5_pop_metrics_table.py. This is
explicitly NOT the tree-shaped stage 4 (see stage4_rocprofsys_tree.py, which keeps every call-tree
node distinct) or rocprofv3's own GPU-kernel aggregation (see stage4_rocprofv3.py).

Functions: scan_ranks(), aggregate(), aggregate_per_rank().
"""

import glob
import os

from stage1_rocprofsys import PID_SUFFIX_RE, parse_table_file
from stage2_rocprofsys import attach_ancestry
from stage3_rocprofsys import remove_tagged_subtrees, tag_rows

NON_TIMING_FILES = {"available.txt", "instrumented.txt", "excluded.txt", "overlapping.txt"}
# rocprof-sys's default config (ROCPROFSYS_FLAT_PROFILE=0, sampling on) writes THREE
# per-rank text tables, not one: wall_clock-<N>.txt (exact instrumented call-tree),
# sampling_wall_clock-<N>.txt and sampling_cpu_clock-<N>.txt (the same statistically-
# sampled call tree, timed two different ways). The CPU-clock variant is an alternate
# measurement of the same intervals sampling_wall_clock already captures, not an
# independent contribution -- keeping both would double-count every sampled function's
# self-time. Excluded by filename prefix (the numeric rank suffix varies) before
# parse_table_file() is even called, same treatment NON_TIMING_FILES gets.
EXCLUDED_METRIC_FILE_PREFIXES = ("sampling_cpu_clock-",)


def scan_ranks(output_dir):
    """Scan output_dir for timemory text tables and group them by RANK, not by
    file -- rocprof-sys's default config (sampling on) writes multiple per-rank
    metric-type files (wall_clock-<N>.txt, sampling_wall_clock-<N>.txt,
    sampling_cpu_clock-<N>.txt), and treating each file as its own rank (the
    bug this replaces) inflates every rank-based number by however many
    metric-type files exist per rank.

    Rank grouping uses the same numeric filename suffix guess_num_ranks() relies
    on (PID_SUFFIX_RE) -- a file with no recognizable suffix becomes its own
    single-file rank, preserving today's behavior for non-conforming inputs.
    sampling_cpu_clock-<N>.txt is excluded entirely (see
    EXCLUDED_METRIC_FILE_PREFIXES). Within a rank, wall_clock's row for a label
    wins over the sampling bucket's row for that same label -- a handful of
    functions are both explicitly instrumented and caught by sampling, and the
    exact instrumented value is preferred over the statistical one.

    Returns a list of {"rank_key": str, "rows": [...merged, each tagged with
    "gpu"/"mpi": bool...], "files": [source paths], "root_sum": float}, one
    entry per distinct rank, in the same order the sorted glob produces.
    "root_sum" is that rank's largest RAW row SUM across all its included
    files, from before same-label rows were merged together -- deliberately
    NOT recomputed from the merged rows, because a rank's wall_clock table has
    one raw row per call-tree node (e.g. one "start_thread" row per worker
    thread), and merging those by label first (as aggregate()'s per-label
    totals need) can make a leaf label's summed SUM exceed the true root
    scope's own SUM -- the same "outermost scope has the single largest raw
    SUM" assumption aggregate() always relied on, just computed correctly
    per rank now instead of per file. root_sum is computed from the FULL,
    unfiltered row list (see below), so it still reflects the whole run's
    true wall-clock regardless of what gets dropped next.

    Rows tagged wrapper_noise, compiler_runtime_noise, wrapper_branch_noise, other, or
    mpi_territory-via-ancestor-only (a thread-root row whose own label
    isn't itself an MPI call -- see stage3_rocprofsys.tag_rows()'s
    self_tags/tags distinction) are dropped entirely here, before either
    cpu/gpu classification or merging -- every consumer of this function
    (aggregate(), aggregate_per_rank(), and extract_pop_metrics.py's own
    direct use of scan_ranks()) inherits the exclusion for free.
    attach_ancestry() still runs on the full, unfiltered rows first, so a
    dropped row's parent-chain links stay intact for tag_rows()'s own
    ancestry/descendant/sibling checks -- it's just never itself added to the
    merged output.
    """
    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True))
    per_rank = {}
    order = []

    for path in candidates:
        base = os.path.basename(path)
        if base in NON_TIMING_FILES or base.startswith(EXCLUDED_METRIC_FILE_PREFIXES):
            continue
        rows = parse_table_file(path)
        if rows is None:
            continue
        attach_ancestry(rows)
        tag_rows(rows, filename=path)

        m = PID_SUFFIX_RE.search(base)
        rank_key = m.group(1) if m else path
        if rank_key not in per_rank:
            per_rank[rank_key] = {"files": [], "wall_clock": {}, "sampling": {}, "root_sum": 0.0}
            order.append(rank_key)
        bucket = per_rank[rank_key]
        bucket["files"].append(path)
        if rows:
            bucket["root_sum"] = max(bucket["root_sum"], max(row["sum"] for row in rows))

        # Keyed by (label, gpu), not just label -- two rows can share a generic
        # label (e.g. "start_thread") while tag_rows()'s ancestor scope tells them
        # apart by ancestry; merging them by label alone would silently recombine
        # what ancestry just split apart.
        target = bucket["wall_clock"] if base.startswith("wall_clock-") else bucket["sampling"]
        # remove_tagged_subtrees() cascades wrapper_branch_noise down to every row in a
        # contaminated sibling's subtree -- tag_rows() itself only marks the top of it.
        for row in remove_tagged_subtrees(rows, {"wrapper_branch_noise"}):
            if (
                "wrapper_noise" in row["tags"]
                or ("mpi_territory" in row["tags"] and "mpi_territory" not in row["self_tags"])
                or "compiler_runtime_noise" in row["tags"]
                or "other" in row["tags"]
            ):
                continue
            gpu = "gpu_api" in row["tags"]
            mpi = "mpi_territory" in row["tags"]  # always self-matched here -- ancestor-only already dropped above
            row = {
                "label": row["label"],
                "count": row["count"],
                "sum": row["sum"],
                "self_sum": row["self_sum"],
                "gpu": gpu,
                "mpi": mpi,
            }
            key = (row["label"], gpu)
            existing = target.get(key)
            if existing is None:
                target[key] = row
            else:
                existing["count"] += row["count"]
                existing["sum"] += row["sum"]
                existing["self_sum"] += row["self_sum"]

    ranks = []
    for rank_key in order:
        bucket = per_rank[rank_key]
        merged = dict(bucket["wall_clock"])
        for key, row in bucket["sampling"].items():
            if key not in merged:
                merged[key] = row
        ranks.append({
            "rank_key": rank_key,
            "rows": list(merged.values()),
            "files": bucket["files"],
            "root_sum": bucket["root_sum"],
        })

    return ranks


def aggregate(output_dir):
    """Scan output_dir for timemory text tables and aggregate rows by clean function name.

    Returns (cpu_entries, gpu_entries, scanned_files, total_runtime) where each
    entries list is [{"label", "count", "sum", "self_sum", "pct_self", "pct_total"}],
    unsorted, and total_runtime is the denominator used for each entry's "% of
    total runtime": the sum, across all ranks (see scan_ranks()), of that
    rank's own "root_sum" (its largest RAW row SUM, before same-label rows are
    merged -- a rank's largest raw SUM is -- barring unusual instrumentation --
    its outermost/root scope, since inclusive time only grows going up the
    call stack; this works whether the underlying file is a hierarchical or a
    flattened profile, without needing to guess the root function's name).
    Computed per rank, not per file, so a rank with multiple metric-type files
    doesn't inflate this sum.

    "sum" is inclusive time (this function plus everything it calls); "self_sum"
    is its own time only, summed across every call-tree node with this label --
    the metric select_entries() ranks by default, since it's the one that
    actually tells apart a real hotspot from a function that just calls the
    next thing (which is why a flat profile, where % SELF is always 100, can't
    support this distinction -- see scripts/profile_CPU_hotspots.sh). Entries'
    own "pct_total" here is self_sum-based (select_entries() recomputes it
    against whichever metric it's actually ranking by, so this is just a
    sensible default for callers that use aggregate()'s output directly).
    """
    ranks = scan_ranks(output_dir)
    scanned_files = [f for r in ranks for f in r["files"]]
    total_runtime = sum(r["root_sum"] for r in ranks)

    # Keyed by (label, gpu), not just label -- see scan_ranks()'s same note:
    # the same generic label (e.g. "start_thread") can be classified
    # differently by ancestry across occurrences, and merging by label alone
    # would recombine what that classification just told apart.
    totals = {}  # (label, gpu) -> {"count": int, "sum": float, "self_sum": float}
    for r in ranks:
        for row in r["rows"]:
            key = (row["label"], row["gpu"])
            entry = totals.setdefault(key, {"count": 0, "sum": 0.0, "self_sum": 0.0})
            entry["count"] += row["count"]
            entry["sum"] += row["sum"]
            entry["self_sum"] += row["self_sum"]

    cpu_entries = []
    gpu_entries = []
    for (label, gpu), entry in totals.items():
        pct_total = (entry["self_sum"] / total_runtime * 100.0) if total_runtime > 0 else None
        pct_self = (entry["self_sum"] / entry["sum"] * 100.0) if entry["sum"] > 0 else None
        item = {
            "label": label,
            "count": entry["count"],
            "sum": entry["sum"],
            "self_sum": entry["self_sum"],
            "pct_self": pct_self,
            "pct_total": pct_total,
        }
        (gpu_entries if gpu else cpu_entries).append(item)

    return cpu_entries, gpu_entries, scanned_files, total_runtime


def aggregate_per_rank(output_dir, unfiltered=False):
    """Like aggregate(), but keeps each RANK's (see scan_ranks()) CPU-only
    per-label totals separate instead of merging them into one global total --
    this is what a load-imbalance-across-ranks computation needs as its input.
    A rank with multiple metric-type files (wall_clock + sampling_wall_clock)
    still contributes exactly one entry here, not one per file.

    Uses self-time by default (not inclusive) for the same reason aggregate()
    ranks by it by default -- keeps the load-imbalance table's ranking
    consistent with the main hotspots table in the same report; unfiltered=True
    switches to inclusive time, matching aggregate()'s own --unfiltered view.

    Returns (per_rank_totals, rank_keys) where per_rank_totals is a
    list of {label: value} dicts, one per rank, in the same order as
    rank_keys.
    """
    metric = "sum" if unfiltered else "self_sum"
    ranks = scan_ranks(output_dir)
    per_rank_totals = []

    for r in ranks:
        file_totals = {}
        for row in r["rows"]:
            if row["gpu"]:
                continue
            file_totals[row["label"]] = file_totals.get(row["label"], 0.0) + row[metric]
        per_rank_totals.append(file_totals)

    return per_rank_totals, [r["rank_key"] for r in ranks]
