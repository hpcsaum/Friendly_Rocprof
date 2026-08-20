"""Stage 4 (merge ranks) for rocprofv3's GPU kernel view.

Scope: turning per-rank *_kernel_stats.csv files into merged, by-kernel-name aggregated entries --
either one global total per kernel (aggregate()) or one total per kernel per rank
(aggregate_per_rank(), what a load-imbalance table needs). No tag/noise-filtering machinery is
needed here, unlike stage4_rocprofsys_sample_flat.py's CPU side -- rocprofv3's kernel_stats.csv is already
clean per-kernel data, not a raw call tree that needs noise classification. Feeds
stage5_gpu_hotspots_table.py, stage5_load_imbalance_table.py, stage5_fused_hotspots_table.py, and
stage5_pop_metrics_table.py.

Functions: aggregate(), aggregate_per_rank().
"""

import glob
import os

from stage1_rocprofv3 import parse_kernel_stats_csv


def aggregate(output_dir):
    """Recursively find *_kernel_stats.csv under output_dir and aggregate by
    kernel name. Returns (entries, scanned_files, total_ns) where entries is
    [{"label", "count", "total_ns"}] (unsorted) and total_ns is the sum of
    every row's TotalDurationNs across every scanned file -- the denominator
    for each entry's "% of total" (rocprofv3 already aggregates duplicate
    kernel names *within* one file, so no per-file dedup is needed here).
    """
    scanned_files = []
    totals = {}  # label -> {"count": int, "total_ns": float}
    total_ns = 0.0

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*_kernel_stats.csv"), recursive=True))
    for path in candidates:
        rows = parse_kernel_stats_csv(path)
        if rows is None:
            continue
        scanned_files.append(path)
        for row in rows:
            entry = totals.setdefault(row["label"], {"count": 0, "total_ns": 0.0})
            entry["count"] += row["count"]
            entry["total_ns"] += row["total_ns"]
            total_ns += row["total_ns"]

    entries = []
    for label, entry in totals.items():
        pct_total = (entry["total_ns"] / total_ns * 100.0) if total_ns > 0 else None
        avg_us = (entry["total_ns"] / entry["count"] / 1000.0) if entry["count"] > 0 else None
        entries.append({
            "label": label,
            "count": entry["count"],
            "sum": entry["total_ns"] / 1e9,  # seconds, for consistent naming with the CPU side
            "avg_us": avg_us,
            "pct_total": pct_total,
        })

    return entries, scanned_files, total_ns


def aggregate_per_rank(output_dir):
    """Like aggregate(), but keeps each scanned file's per-kernel sums
    separate instead of merging them into one global total -- one scanned
    file is treated as one rank's contribution (same file-per-process
    assumption guess_num_ranks() already relies on). rocprofv3 already
    aggregates duplicate kernel names within one file internally, so each
    file's own rows are already a clean per-rank subtotal, same as
    aggregate() itself relies on.

    Returns (per_file_totals, scanned_files) where per_file_totals is a
    list of {kernel_name: total_seconds} dicts, one per scanned file, in
    the same order as scanned_files.
    """
    scanned_files = []
    per_file_totals = []

    candidates = sorted(glob.glob(os.path.join(output_dir, "**", "*_kernel_stats.csv"), recursive=True))
    for path in candidates:
        rows = parse_kernel_stats_csv(path)
        if rows is None:
            continue
        scanned_files.append(path)
        file_totals = {}
        for row in rows:
            file_totals[row["label"]] = file_totals.get(row["label"], 0.0) + row["total_ns"] / 1e9
        per_file_totals.append(file_totals)

    return per_file_totals, scanned_files
