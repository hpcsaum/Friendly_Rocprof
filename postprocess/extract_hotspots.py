#!/usr/bin/env python3
"""Merge a rocprof-sys CPU-hotspots run and a rocprofv3 GPU-kernel-hotspots run
into one coherent report: a single fused CPU+GPU ranking, plus the three
underlying per-domain tables it was built from.

Takes both tools' output directories directly -- it does NOT check that they
came from the same executable/test case/run. That's the caller's
responsibility (garbage in, garbage out).

Import note: this only works when run directly (`python3
extract_hotspots.py ...`), since it relies on Python putting this
script's own directory at the front of sys.path so its sibling stage/tool
modules import as plain siblings with no path hacking.
"""

import argparse
import os
from datetime import datetime

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
import stage4_rocprofsys_flat
import stage4_rocprofv3
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_fused_hotspots_table import FUSED_HOTSPOTS_COLUMNS, build_combined_view
from stage5_gpu_hotspots_table import GPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, load_imbalance_columns
from stage5_table_render import render_table, select_entries
from stage6_report_builder import render_report, write_report_file

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


def _prepare_pct_total(field, total):
    def _prepare(entries):
        for e in entries:
            e["pct_total"] = (e[field] / total * 100.0) if total > 0 else None
    return _prepare


def write_report(rocprof_sys_dir, rocprofv3_dir, dest_path, top=None, threshold=None, show_all=False,
                  unfiltered=False):
    fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = build_combined_view(rocprof_sys_dir, rocprofv3_dir)
    rank_by = "inclusive" if unfiltered else "self"
    key_field = "sum" if rank_by == "inclusive" else "self_sum"

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

    fused_selected, fused_desc = select_entries(
        fused_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
        prepare=_prepare_pct_total(key_field, info["combined_total_sec"]),
    )
    cpu_selected, cpu_desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
        prepare=_prepare_pct_total(key_field, info["cpu_total_raw"]),
    )
    gpu_selected, gpu_desc = select_entries(
        gpu_entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
    )

    gpu_api_selected, gpu_api_desc = select_entries(
        cpu_gpu_api_entries, rank_field="self_sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime",
        prepare=_prepare_pct_total("self_sum", info["cpu_total_raw"]),
    )

    cpu_per_rank, cpu_imbalance_scanned = stage4_rocprofsys_flat.aggregate_per_rank(rocprof_sys_dir, unfiltered=unfiltered)
    if len(cpu_imbalance_scanned) < 2:
        cpu_imbalance_title, cpu_imbalance_body = None, (
            "=== 5. CPU load imbalance across ranks (rocprof-sys run) -- skipped: only "
            f"{len(cpu_imbalance_scanned)} rank/file found, need at least 2 to compare ===\n"
        )
    else:
        cpu_imbalance_selected, cpu_imbalance_desc = compute_load_imbalance(cpu_per_rank, top, threshold, show_all)
        cpu_imbalance_title = (
            f"=== 5. CPU load imbalance across {len(cpu_imbalance_scanned)} ranks (rocprof-sys run) "
            f"-- showing {cpu_imbalance_desc} ===\n"
        )
        cpu_imbalance_body = (
            ("Each function's own inclusive" if unfiltered else "Each function's own self")
            + " time on each rank, compared across ranks -- a rank "
            "that never called a function counts as 0.0 for that rank, not omitted.\n"
        ) + render_table(load_imbalance_columns(), cpu_imbalance_selected)

    gpu_per_rank, gpu_imbalance_scanned = stage4_rocprofv3.aggregate_per_rank(rocprofv3_dir)
    if len(gpu_imbalance_scanned) < 2:
        gpu_imbalance_title, gpu_imbalance_body = None, (
            "=== 6. GPU kernel load imbalance across ranks (rocprofv3 run) -- skipped: only "
            f"{len(gpu_imbalance_scanned)} rank/file found, need at least 2 to compare ===\n"
        )
    else:
        gpu_imbalance_selected, gpu_imbalance_desc = compute_load_imbalance(gpu_per_rank, top, threshold, show_all)
        gpu_imbalance_title = (
            f"=== 6. GPU kernel load imbalance across {len(gpu_imbalance_scanned)} ranks (rocprofv3 run) "
            f"-- showing {gpu_imbalance_desc} ===\n"
        )
        gpu_imbalance_body = (
            "Each kernel's own total time on each rank, compared across ranks -- a rank "
            "that never launched a kernel counts as 0.0 for that rank, not omitted.\n"
        ) + render_table(load_imbalance_columns(item_label="kernel"), gpu_imbalance_selected)

    header = (
        "rocprof combined (CPU + GPU) hotspots report\n"
        f"generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"CPU run directory (rocprof-sys): {os.path.abspath(rocprof_sys_dir)}\n"
        f"  executable: {cpu_run_info['executable'] or ''}\n"
        f"  run date/time: {cpu_run_info['run_datetime'] or ''}\n"
        f"  total runtime: {cpu_run_info['total_runtime'] or ''}\n"
        f"  MPI ranks: {cpu_run_info['num_ranks'] if cpu_run_info['num_ranks'] is not None else ''}\n"
        f"GPU run directory (rocprofv3): {os.path.abspath(rocprofv3_dir)}\n"
        f"  executable: {gpu_run_info['executable'] or ''}\n"
        f"  run date/time: {gpu_run_info['run_datetime'] or ''}\n"
        f"  total runtime: {gpu_run_info['total_runtime'] or ''}\n"
        f"  MPI ranks: {gpu_run_info['num_ranks'] if gpu_run_info['num_ranks'] is not None else ''}\n"
        "Note: the two directories above are not checked against each other "
        "(same executable/test case/run) -- that's the caller's responsibility.\n"
        "\n"
        + (
            "Ranked by inclusive (total) time -- a function that just calls other "
            "functions can still rank high. Drop --unfiltered for the self-time view.\n"
            if unfiltered else
            "Ranked by self time (each function/kernel's own work, not counting time "
            "spent in what it calls) -- pass-through CPU functions fall out of the "
            "ranking on their own; GPU kernels are unaffected (already leaf events). "
            "Pass --unfiltered for the old inclusive/cumulative-time view.\n"
        )
        + "\n"
        "Combined-pool arithmetic (see the footer note below for why):\n"
        f"  CPU run raw total:              {info['cpu_total_raw']:.6f} sec\n"
        f"  - GPU sync-wait time:           {info['gpu_api_overhead_sec']:.6f} sec\n"
        f"  = CPU pure-compute total:        {info['cpu_pure_total_sec']:.6f} sec\n"
        f"  + GPU kernel total:              {info['gpu_total_sec']:.6f} sec\n"
        f"  = combined pool:                 {info['combined_total_sec']:.6f} sec\n"
        "\n"
    )
    sections = [
        (f"=== 1. Combined hotspots (fused CPU+GPU ranking) -- showing {fused_desc} ===\n",
         "%total here is each entry's share of the combined pool above (double-counting-corrected).\n"
         + render_table(FUSED_HOTSPOTS_COLUMNS, fused_selected)
         + "Note: this fused ranking excludes CPU-side time spent blocked in "
         "hipStreamSynchronize/hipDeviceSynchronize, to avoid counting GPU execution time "
         "twice -- once as CPU-side wait time, once as GPU-side kernel time. See table 4 "
         "below for that (and every other GPU-API call rocprof-sys saw).\n"),
        (f"=== 2. CPU compute hotspots (rocprof-sys run) -- showing {cpu_desc} ===\n",
         "%total here is each entry's share of the CPU run's OWN total (not the combined pool).\n"
         + render_table(CPU_HOTSPOTS_COLUMNS, cpu_selected)),
        (f"=== 3. GPU kernel hotspots (rocprofv3 run) -- showing {gpu_desc} ===\n",
         "%total here is each entry's share of the GPU run's OWN total (not the combined pool).\n"
         + render_table(GPU_HOTSPOTS_COLUMNS, gpu_selected)),
        (f"=== 4. GPU API / launch overhead (rocprof-sys run) -- showing {gpu_api_desc} ===\n",
         "%total here is each entry's share of the CPU run's OWN total. Every ROCm-library "
         "call rocprof-sys saw is listed here, for digging in -- but only "
         "hipStreamSynchronize/hipDeviceSynchronize (the two calls that actually mean \"block "
         "the CPU until the GPU catches up\") are subtracted out of table 1's combined pool; "
         "the rest (e.g. hsakmt_ioctl, rocr::* runtime-internal busy-wait/event threads) is "
         "shown here but deliberately left out of that subtraction -- see the header note.\n"
         + render_table(CPU_HOTSPOTS_COLUMNS, gpu_api_selected)),
        (cpu_imbalance_title, cpu_imbalance_body),
        (gpu_imbalance_title, gpu_imbalance_body),
    ]

    return write_report_file(dest_path, render_report(header, sections))


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
