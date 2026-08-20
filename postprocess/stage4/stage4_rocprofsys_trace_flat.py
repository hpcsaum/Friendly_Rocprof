"""Stage 4 (cross-rank flat/per-rank derivations) for rocprof-sys's Perfetto trace-CSV pipeline.

Scope: three pure, tool-independent reductions over N per-rank canonical aggregates (each from
stage4_rocprofsys_trace_aggregate.get_rank_aggregate()) -- no re-parsing, no re-tagging, no file
discovery of its own (see stage4_rocprofsys_trace_tree.py's docstring for why). A global flat-entry
list (aggregate()), a per-rank {label: value} breakdown (aggregate_per_rank()), and a per-rank
timing-summary list (gather_timing_summary_per_rank()) -- feeding, respectively, the by-label
hotspots tables' column specs, stage5_load_imbalance_table.compute_load_imbalance(), and
stage5_pop_metrics_table.compute_metrics_from_per_rank(), all three unchanged. Unlike the sample
pipeline's paired-rocprofv3 case, GPU visibility is always built into the same trace -- gpu_busy_time
is never None here, it just may be 0.0 when a rank has no gpu_kernel-tagged rows.

Functions: aggregate(), aggregate_per_rank(), gather_timing_summary_per_rank().
"""

from stage4_rocprofsys_trace_aggregate import get_rank_aggregate


def _rank_rows(rank_inputs, cache_dir):
    return [
        (rank_key, get_rank_aggregate(csv_paths, rank_key, cache_dir=cache_dir))
        for rank_key, csv_paths in rank_inputs
    ]


def aggregate(rank_inputs, cache_dir=None):
    """Global by-label totals across every rank -- returns (entries, total_runtime), entries being
    [{"label", "count", "sum", "self_sum", "pct_self", "pct_total"}], the same flat-entry shape
    stage4_rocprofsys_sample_flat.aggregate() produces. total_runtime is the sum, across ranks, of
    that rank's own largest root sum (its outermost scope's inclusive time) -- the same "outermost
    scope has the single largest sum" convention the sample pipeline's aggregate() already relies
    on, computed here from each rank's own root rows instead of a raw text-table row."""
    ranks = _rank_rows(rank_inputs, cache_dir)

    total_runtime = 0.0
    totals = {}
    for _rank_key, rows in ranks:
        roots = [row for row in rows if row["parent"] is None]
        if roots:
            total_runtime += max(row["sum"] for row in roots)
        for row in rows:
            entry = totals.setdefault(row["label"], {"count": 0, "sum": 0.0, "self_sum": 0.0})
            entry["count"] += row["count"]
            entry["sum"] += row["sum"]
            entry["self_sum"] += row["self_sum"]

    entries = []
    for label, entry in totals.items():
        pct_total = (entry["self_sum"] / total_runtime * 100.0) if total_runtime > 0 else None
        pct_self = (entry["self_sum"] / entry["sum"] * 100.0) if entry["sum"] > 0 else None
        entries.append({
            "label": label,
            "count": entry["count"],
            "sum": entry["sum"],
            "self_sum": entry["self_sum"],
            "pct_self": pct_self,
            "pct_total": pct_total,
        })
    return entries, total_runtime


def aggregate_per_rank(rank_inputs, cache_dir=None, unfiltered=False):
    """Per-rank {label: value} breakdown -- returns (per_rank_totals, rank_keys), one dict per
    rank, the per-rank label->value shape stage5_load_imbalance_table.compute_load_imbalance()
    consumes directly. self_sum by default (matching aggregate()'s own default ranking metric);
    unfiltered=True switches to inclusive sum, mirroring the sample pipeline's own
    aggregate_per_rank()'s --unfiltered switch."""
    metric = "sum" if unfiltered else "self_sum"
    ranks = _rank_rows(rank_inputs, cache_dir)

    per_rank_totals = []
    for _rank_key, rows in ranks:
        totals = {}
        for row in rows:
            totals[row["label"]] = totals.get(row["label"], 0.0) + row[metric]
        per_rank_totals.append(totals)

    return per_rank_totals, [rank_key for rank_key, _rows in ranks]


def gather_timing_summary_per_rank(rank_inputs, cache_dir=None):
    """Per-rank {"rank_key", "total_time", "comm_time", "useful_compute", "cpu_only_time",
    "gpu_busy_time"} -- the per-rank timing-summary shape
    stage5_pop_metrics_table.compute_metrics_from_per_rank() needs. total_time is each rank's own
    largest root sum; comm_time is the self_sum sum of every mpi_territory-tagged row (self-time,
    not inclusive, so nested MPI-internal helper calls aren't double-counted with their parent);
    gpu_busy_time is the self_sum sum of every gpu_kernel-tagged row -- always a real number here
    (never None the way the sample pipeline's paired-rocprofv3 case can be), since GPU visibility
    is always part of the same trace."""
    ranks = _rank_rows(rank_inputs, cache_dir)

    per_rank = []
    for rank_key, rows in ranks:
        roots = [row for row in rows if row["parent"] is None]
        total_time = max((row["sum"] for row in roots), default=0.0)
        comm_time = sum(row["self_sum"] for row in rows if "mpi_territory" in row["tags"])
        gpu_busy_time = sum(row["self_sum"] for row in rows if "gpu_kernel" in row["tags"])
        cpu_pure = max(0.0, total_time - comm_time - gpu_busy_time)
        useful_compute = cpu_pure + gpu_busy_time
        per_rank.append({
            "rank_key": rank_key,
            "total_time": total_time,
            "comm_time": comm_time,
            "useful_compute": useful_compute,
            "cpu_only_time": cpu_pure,
            "gpu_busy_time": gpu_busy_time,
        })

    return per_rank
