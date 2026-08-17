"""Stage 5 fused (CPU+GPU) hotspots table.

Scope: merging stage4_rocprofsys_flat's CPU entries and stage4_rocprofv3's GPU entries into one
combined-pool ranked view, correcting for the CPU-side GPU-sync-wait time double-count, plus the
fused table's own column spec. Ranking/filtering/rendering themselves are generic (see
stage5_table_render.py).

Functions: build_combined_view().
"""

import stage4_rocprofsys_flat
import stage4_rocprofv3

# The two HIP calls that mean "block the CPU until the GPU catches up" -- see
# build_combined_view()'s docstring for why only these two are subtracted out of the
# combined pool.
SYNC_WAIT_LABELS = {"hipStreamSynchronize", "hipDeviceSynchronize"}

FUSED_HOTSPOTS_COLUMNS = [
    {"header": "#", "width": 3, "value": lambda e, i: str(i)},
    {"header": "self(s)", "width": 12, "value": lambda e, i: f"{e['self_sum']:.6f}"},
    {"header": "%total", "width": 7,
     "value": lambda e, i: f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"},
    {"header": "total(s)", "width": 12, "value": lambda e, i: f"{e['sum']:.6f}"},
    {"header": "dom", "width": 3, "value": lambda e, i: e["domain"]},
    {"header": "calls", "width": 10, "value": lambda e, i: str(e["count"])},
    {"header": "name", "width": None, "value": lambda e, i: e["label"]},
]


def build_combined_view(rocprof_sys_dir, rocprofv3_dir):
    """Returns (fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info).

    fused_entries: cpu_entries + gpu_entries merged into one list, each tagged
    "domain" ("CPU"/"GPU") with "pct_total" recomputed against info's
    combined_total_sec (NOT copied from either source dict, which are each
    relative to their own run's own total).

    info: {cpu_scanned, gpu_scanned, cpu_total_raw, gpu_api_overhead_sec,
    cpu_pure_total_sec, gpu_total_sec, combined_total_sec} -- every number that
    feeds the fused %total, so the report can show its own arithmetic.

    The double-counting fix: rocprof-sys's CPU total is inclusive of time
    blocked inside hipStreamSynchronize/hipDeviceSynchronize -- the same
    physical interval rocprofv3's kernel TotalDurationNs already counts from
    the device side. Subtracting that out of the CPU total before adding the
    GPU total avoids counting that overlap twice.

    gpu_api_overhead_sec sums ONLY these two exact labels' self_sum --
    deliberately not every GPU-API-classified entry in cpu_gpu_api_entries
    (that bucket still holds everything, unabridged, for table 4). The wider
    bucket includes things like hsakmt_ioctl and
    rocr::core::BusyWaitSignal::WaitAcquire, whose self-time is summed across
    however many concurrent threads call them -- on a real multi-threaded run
    that sum can legitimately exceed a single rank's own wall-clock span
    (several threads can be simultaneously blocked on the GPU at once), which
    would clamp cpu_pure_total_sec to 0 if the wider bucket were used here.
    hipStreamSynchronize/hipDeviceSynchronize are the two calls that
    actually mean "block the CPU until the GPU catches up" -- a good enough
    beginner-tool approximation of "time spent on the GPU" without that
    multi-thread-sum inflation. Self-time (not inclusive sum) still matters
    here too: if either label appears as more than one raw row (e.g. called
    from multiple threads), self_sum sums correctly across them since every
    node's self-time is disjoint from every other's.
    """
    cpu_entries, cpu_gpu_api_entries, cpu_scanned, cpu_total_raw = stage4_rocprofsys_flat.aggregate(rocprof_sys_dir)
    gpu_entries, gpu_scanned, gpu_total_ns = stage4_rocprofv3.aggregate(rocprofv3_dir)

    gpu_api_overhead_sec = sum(e["self_sum"] for e in cpu_gpu_api_entries if e["label"] in SYNC_WAIT_LABELS)
    cpu_pure_total_sec = max(0.0, cpu_total_raw - gpu_api_overhead_sec)
    gpu_total_sec = gpu_total_ns / 1e9
    combined_total_sec = cpu_pure_total_sec + gpu_total_sec

    # self_sum drives the fused ranking by default (see stage5_table_render.select_entries()'s
    # rank_by) -- for GPU kernel entries there's no self-vs-inclusive distinction
    # (a kernel is already a leaf event), so self_sum == sum there. pct_total here
    # is self-based, same convention as stage4_rocprofsys_flat.aggregate()'s own output --
    # select_entries() recomputes it against whichever metric it actually ranks by.
    fused_entries = []
    for e in cpu_entries:
        pct = (e["self_sum"] / combined_total_sec * 100.0) if combined_total_sec > 0 else None
        fused_entries.append({
            "label": e["label"], "domain": "CPU", "count": e["count"],
            "sum": e["sum"], "self_sum": e["self_sum"], "pct_total": pct,
        })
    for e in gpu_entries:
        pct = (e["sum"] / combined_total_sec * 100.0) if combined_total_sec > 0 else None
        fused_entries.append({
            "label": e["label"], "domain": "GPU", "count": e["count"],
            "sum": e["sum"], "self_sum": e["sum"], "pct_total": pct,
        })

    info = {
        "cpu_scanned": cpu_scanned,
        "gpu_scanned": gpu_scanned,
        "cpu_total_raw": cpu_total_raw,
        "gpu_api_overhead_sec": gpu_api_overhead_sec,
        "cpu_pure_total_sec": cpu_pure_total_sec,
        "gpu_total_sec": gpu_total_sec,
        "combined_total_sec": combined_total_sec,
    }
    return fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info
