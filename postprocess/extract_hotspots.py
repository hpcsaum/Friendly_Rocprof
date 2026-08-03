#!/usr/bin/env python3
"""Merge a rocprof-sys CPU-hotspots run and a rocprofv3 GPU-kernel-hotspots run
into one coherent report: a single fused CPU+GPU ranking, plus the three
underlying per-domain tables it was built from.

Takes both tools' output directories directly -- it does NOT check that they
came from the same executable/test case/run. That's the caller's
responsibility (garbage in, garbage out).

Import note: this only works when run directly (`python3
extract_hotspots.py ...`), since it relies on Python putting this
script's own directory at the front of sys.path so `extract_CPU_hotspots` and
`extract_GPU_hotspots` import as plain siblings with no path hacking.
"""

import argparse
import os
from datetime import datetime

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool

HELP_BLURB = """\
Reads the output of a profile_hotspots.sh run (or a matching pair of
rocprof-sys / rocprofv3 output directories) and writes ONE combined report:
a single ranking that mixes CPU functions and GPU kernels together, telling
you what's worth looking at first regardless of which side it's on.

Also includes the three tables it was built from (CPU-only, GPU-only, and
the CPU-side GPU-API/wait-time bucket it corrects for), so you can see
exactly what fed into the combined ranking.

Takes the two directories as-is and does NOT check they came from the same
program/run -- that's on you. Numbers are percentages, good enough to find
your top bottleneck, not a precise benchmark.

Under the hood, this parses output written by AMD's rocprof-sys and
rocprofv3 -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ and
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
for details.
"""


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
    blocked inside hipStreamSynchronize/hipDeviceSynchronize/a synchronous
    hipMemcpy, etc. (cpu_gpu_api_entries) -- the same physical interval
    rocprofv3's kernel TotalDurationNs already counts from the device side.
    Subtracting that bucket out of the CPU total before adding the GPU total
    avoids counting that overlap twice. This also strips the (comparatively
    tiny) non-blocking launch overhead in the same bucket -- accepted, since
    there's no reliable way to tell blocking from non-blocking HIP/HSA calls
    by name alone, and it's negligible next to the sync-wait time being fixed.
    """
    cpu_entries, cpu_gpu_api_entries, cpu_scanned, cpu_total_raw = cpu_tool.aggregate(rocprof_sys_dir)
    gpu_entries, gpu_scanned, gpu_total_ns = gpu_tool.aggregate(rocprofv3_dir)

    gpu_api_overhead_sec = sum(e["sum"] for e in cpu_gpu_api_entries)
    cpu_pure_total_sec = max(0.0, cpu_total_raw - gpu_api_overhead_sec)
    gpu_total_sec = gpu_total_ns / 1e9
    combined_total_sec = cpu_pure_total_sec + gpu_total_sec

    # self_sum drives the fused ranking by default (see cpu_tool.select_entries's
    # rank_by) -- for GPU kernel entries there's no self-vs-inclusive distinction
    # (a kernel is already a leaf event), so self_sum == sum there. pct_total here
    # is self-based, same convention as cpu_tool.aggregate()'s own output --
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


def format_table_fused(entries):
    if not entries:
        return "  (none found)\n"
    lines = []
    lines.append(f"  {'#':>3}  {'self(s)':>12}  {'%total':>7}  {'total(s)':>12}  {'dom':>3}  {'calls':>10}  name")
    for i, e in enumerate(entries, 1):
        pct_str = f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"
        lines.append(
            f"  {i:>3}  {e['self_sum']:>12.6f}  {pct_str:>7}  {e['sum']:>12.6f}  "
            f"{e['domain']:>3}  {e['count']:>10}  {e['label']}"
        )
    return "\n".join(lines) + "\n"


def write_report(rocprof_sys_dir, rocprofv3_dir, dest_path, top=None, threshold=None, show_all=False,
                  unfiltered=False):
    fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = build_combined_view(rocprof_sys_dir, rocprofv3_dir)
    rank_by = "inclusive" if unfiltered else "self"

    if not info["cpu_scanned"]:
        raise SystemExit(
            f"error: no rocprof-sys timemory data found under {rocprof_sys_dir!r} -- nothing to combine"
        )
    if not info["gpu_scanned"]:
        raise SystemExit(
            f"error: no rocprofv3 kernel_stats.csv found under {rocprofv3_dir!r} -- nothing to combine"
        )

    cpu_run_info = cpu_tool.gather_run_info(rocprof_sys_dir, info["cpu_scanned"])
    gpu_run_info = gpu_tool.gather_run_info(rocprofv3_dir, info["gpu_scanned"])

    fused_selected, fused_desc = cpu_tool.select_entries(
        fused_entries, info["combined_total_sec"], top, threshold, show_all, rank_by
    )
    cpu_selected, cpu_desc = cpu_tool.select_entries(
        cpu_entries, info["cpu_total_raw"], top, threshold, show_all, rank_by
    )
    gpu_selected, gpu_desc = gpu_tool.select_entries(gpu_entries, info["gpu_total_sec"], top, threshold, show_all)

    parts = []
    parts.append("rocprof combined (CPU + GPU) hotspots report\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"CPU run directory (rocprof-sys): {os.path.abspath(rocprof_sys_dir)}\n")
    parts.append(f"  executable: {cpu_run_info['executable'] or ''}\n")
    parts.append(f"  run date/time: {cpu_run_info['run_datetime'] or ''}\n")
    parts.append(f"  total runtime: {cpu_run_info['total_runtime'] or ''}\n")
    parts.append(f"  MPI ranks: {cpu_run_info['num_ranks'] if cpu_run_info['num_ranks'] is not None else ''}\n")
    parts.append(f"GPU run directory (rocprofv3): {os.path.abspath(rocprofv3_dir)}\n")
    parts.append(f"  executable: {gpu_run_info['executable'] or ''}\n")
    parts.append(f"  run date/time: {gpu_run_info['run_datetime'] or ''}\n")
    parts.append(f"  total runtime: {gpu_run_info['total_runtime'] or ''}\n")
    parts.append(f"  MPI ranks: {gpu_run_info['num_ranks'] if gpu_run_info['num_ranks'] is not None else ''}\n")
    parts.append(
        "Note: the two directories above are not checked against each other "
        "(same executable/test case/run) -- that's the caller's responsibility.\n"
    )
    parts.append("\n")
    if unfiltered:
        parts.append(
            "Ranked by inclusive (total) time -- a function that just calls other "
            "functions can still rank high. Drop --unfiltered for the self-time view.\n"
        )
    else:
        parts.append(
            "Ranked by self time (each function/kernel's own work, not counting time "
            "spent in what it calls) -- pass-through CPU functions fall out of the "
            "ranking on their own; GPU kernels are unaffected (already leaf events). "
            "Pass --unfiltered for the old inclusive/cumulative-time view.\n"
        )
    parts.append("\n")
    parts.append("Combined-pool arithmetic (see the footer note below for why):\n")
    parts.append(f"  CPU run raw total:              {info['cpu_total_raw']:.6f} sec\n")
    parts.append(f"  - GPU API / launch overhead:    {info['gpu_api_overhead_sec']:.6f} sec\n")
    parts.append(f"  = CPU pure-compute total:        {info['cpu_pure_total_sec']:.6f} sec\n")
    parts.append(f"  + GPU kernel total:              {info['gpu_total_sec']:.6f} sec\n")
    parts.append(f"  = combined pool:                 {info['combined_total_sec']:.6f} sec\n")
    parts.append("\n")

    parts.append(f"=== 1. Combined hotspots (fused CPU+GPU ranking) -- showing {fused_desc} ===\n")
    parts.append("%total here is each entry's share of the combined pool above (double-counting-corrected).\n")
    parts.append(format_table_fused(fused_selected))
    parts.append(
        "Note: this fused ranking excludes the CPU run's GPU API/launch-overhead bucket "
        "(e.g. hipStreamSynchronize) to avoid counting GPU execution time twice -- once as "
        "CPU-side wait time, once as GPU-side kernel time. See table 4 below for that bucket.\n"
    )
    parts.append("\n")

    parts.append(f"=== 2. CPU compute hotspots (rocprof-sys run) -- showing {cpu_desc} ===\n")
    parts.append("%total here is each entry's share of the CPU run's OWN total (not the combined pool).\n")
    parts.append(cpu_tool.format_table(cpu_selected))
    parts.append("\n")

    parts.append(f"=== 3. GPU kernel hotspots (rocprofv3 run) -- showing {gpu_desc} ===\n")
    parts.append("%total here is each entry's share of the GPU run's OWN total (not the combined pool).\n")
    parts.append(gpu_tool.format_table(gpu_selected))
    parts.append("\n")

    gpu_api_selected, gpu_api_desc = cpu_tool.select_entries(cpu_gpu_api_entries, info["cpu_total_raw"], top, threshold, show_all)
    parts.append(f"=== 4. GPU API / launch overhead (rocprof-sys run) -- showing {gpu_api_desc} ===\n")
    parts.append(
        "%total here is each entry's share of the CPU run's OWN total. This is exactly the "
        "bucket subtracted out of table 1's combined pool.\n"
    )
    parts.append(cpu_tool.format_table(gpu_api_selected))
    parts.append("\n")

    cpu_per_rank, cpu_imbalance_scanned = cpu_tool.aggregate_per_rank(rocprof_sys_dir, unfiltered=unfiltered)
    if len(cpu_imbalance_scanned) < 2:
        parts.append(
            "=== 5. CPU load imbalance across ranks (rocprof-sys run) -- skipped: only "
            f"{len(cpu_imbalance_scanned)} rank/file found, need at least 2 to compare ===\n"
        )
    else:
        cpu_imbalance_selected, cpu_imbalance_desc = cpu_tool.compute_load_imbalance(cpu_per_rank, top, threshold, show_all)
        parts.append(
            f"=== 5. CPU load imbalance across {len(cpu_imbalance_scanned)} ranks (rocprof-sys run) "
            f"-- showing {cpu_imbalance_desc} ===\n"
        )
        parts.append(
            ("Each function's own inclusive" if unfiltered else "Each function's own self")
            + " time on each rank, compared across ranks -- a rank "
            "that never called a function counts as 0.0 for that rank, not omitted.\n"
        )
        parts.append(cpu_tool.format_table_load_imbalance(cpu_imbalance_selected))
    parts.append("\n")

    gpu_per_rank, gpu_imbalance_scanned = gpu_tool.aggregate_per_rank(rocprofv3_dir)
    if len(gpu_imbalance_scanned) < 2:
        parts.append(
            "=== 6. GPU kernel load imbalance across ranks (rocprofv3 run) -- skipped: only "
            f"{len(gpu_imbalance_scanned)} rank/file found, need at least 2 to compare ===\n"
        )
    else:
        gpu_imbalance_selected, gpu_imbalance_desc = gpu_tool.compute_load_imbalance(gpu_per_rank, top, threshold, show_all)
        parts.append(
            f"=== 6. GPU kernel load imbalance across {len(gpu_imbalance_scanned)} ranks (rocprofv3 run) "
            f"-- showing {gpu_imbalance_desc} ===\n"
        )
        parts.append(
            "Each kernel's own total time on each rank, compared across ranks -- a rank "
            "that never launched a kernel counts as 0.0 for that rank, not omitted.\n"
        )
        parts.append(gpu_tool.format_table_load_imbalance(gpu_imbalance_selected))

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rocprof_sys_dir", help="rocprof-sys output directory (CPU side)")
    parser.add_argument("rocprofv3_dir", help="rocprofv3 output directory (GPU side)")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the combined report (default: <rocprof_sys_dir>/hotspots.txt)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-n", "--top", dest="top", type=int, default=None,
                            help="number of hotspots to list per table (default: 20)")
    selection.add_argument("--threshold", dest="threshold", type=float, default=None,
                            help="only list entries at or above this %% of their table's total")
    selection.add_argument("--all", dest="show_all", action="store_true",
                            help="list every entry, no truncation")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="rank CPU-side entries by inclusive (total) time instead of self "
                              "time -- the old behavior, where a function that just calls other "
                              "functions can still rank high")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.rocprof_sys_dir):
        raise SystemExit(f"error: no such directory: {args.rocprof_sys_dir!r}")
    if not os.path.isdir(args.rocprofv3_dir):
        raise SystemExit(f"error: no such directory: {args.rocprofv3_dir!r}")

    dest = args.dest or os.path.join(args.rocprof_sys_dir, "hotspots.txt")
    write_report(args.rocprof_sys_dir, args.rocprofv3_dir, dest, top=args.top, threshold=args.threshold,
                 show_all=args.show_all, unfiltered=args.unfiltered)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
