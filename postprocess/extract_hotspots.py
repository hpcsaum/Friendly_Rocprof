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
import sys

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
import stage4_rocprofsys_flat
import stage4_rocprofv3
from stage1_run_dirs import resolve_two_dirs
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_fused_hotspots_table import FUSED_HOTSPOTS_COLUMNS, build_combined_view
from stage5_gpu_hotspots_table import GPU_HOTSPOTS_COLUMNS
from stage5_load_imbalance_table import compute_load_imbalance, imbalance_note, load_imbalance_columns
from stage5_table_render import pct_total_note, ranking_note, render_table, select_entries
import stage6_noise_config
from stage6_report_builder import command_header, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Combines a rocprof-sys CPU run and a rocprofv3 GPU run into one fused CPU+GPU hotspots\n"
    "ranking.\n"
)

HELP_BLURB = """\
Reads the output of a profile_hotspots.sh run (a single directory containing both a
rocprof-sys/ and a rocprofv3/ subdir), or a matching pair of separately-run
rocprof-sys / rocprofv3 output directories, and writes ONE combined report:
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
                  unfiltered=False, command_line=""):
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
    # One shared identity for the run being profiled, not two independent blocks -- there's only
    # one program running, regardless of how many separate tools captured it. CPU's own metadata
    # is preferred; GPU's is used only for whichever field CPU's own metadata didn't have.
    run_info = {
        key: cpu_run_info[key] if cpu_run_info[key] is not None else gpu_run_info[key]
        for key in ("executable", "run_datetime", "total_runtime", "num_ranks")
    }

    fused_unit = "of the combined pool above (double-counting-corrected)"
    cpu_unit = "of the CPU run's own total (not the combined pool)"
    gpu_unit = "of the GPU run's own total (not the combined pool)"

    fused_selected, fused_desc = select_entries(
        fused_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=fused_unit,
        prepare=_prepare_pct_total(key_field, info["combined_total_sec"]),
    )
    cpu_selected, cpu_desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=cpu_unit,
        prepare=_prepare_pct_total(key_field, info["cpu_total_raw"]),
    )
    gpu_selected, gpu_desc = select_entries(
        gpu_entries, rank_field="sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=gpu_unit,
    )

    gpu_api_selected, gpu_api_desc = select_entries(
        cpu_gpu_api_entries, rank_field="self_sum", threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=cpu_unit,
        prepare=_prepare_pct_total("self_sum", info["cpu_total_raw"]),
    )

    cpu_per_rank, cpu_imbalance_scanned = stage4_rocprofsys_flat.aggregate_per_rank(rocprof_sys_dir, unfiltered=unfiltered)
    if len(cpu_imbalance_scanned) < 2:
        cpu_imbalance_title = (
            f"CPU load imbalance across ranks (rocprof-sys run) -- skipped: only "
            f"{len(cpu_imbalance_scanned)} rank/file found, need at least 2 to compare\n"
        )
        cpu_imbalance_body = ""
    else:
        cpu_imbalance_selected, cpu_imbalance_desc = compute_load_imbalance(cpu_per_rank, top, threshold, show_all)
        cpu_imbalance_title = (
            f"CPU load imbalance across {len(cpu_imbalance_scanned)} ranks (rocprof-sys run) "
            f"-- showing {cpu_imbalance_desc}\n"
        )
        cpu_imbalance_body = (
            render_table(load_imbalance_columns(), cpu_imbalance_selected) + "\n"
            + imbalance_note("function", "inclusive" if unfiltered else "self")
        )

    gpu_per_rank, gpu_imbalance_scanned = stage4_rocprofv3.aggregate_per_rank(rocprofv3_dir)
    if len(gpu_imbalance_scanned) < 2:
        gpu_imbalance_title = (
            f"GPU kernel load imbalance across ranks (rocprofv3 run) -- skipped: only "
            f"{len(gpu_imbalance_scanned)} rank/file found, need at least 2 to compare\n"
        )
        gpu_imbalance_body = ""
    else:
        gpu_imbalance_selected, gpu_imbalance_desc = compute_load_imbalance(gpu_per_rank, top, threshold, show_all)
        gpu_imbalance_title = (
            f"GPU kernel load imbalance across {len(gpu_imbalance_scanned)} ranks (rocprofv3 run) "
            f"-- showing {gpu_imbalance_desc}\n"
        )
        gpu_imbalance_body = (
            render_table(load_imbalance_columns(item_label="kernel"), gpu_imbalance_selected) + "\n"
            + imbalance_note("kernel", "total")
        )

    header = standard_header("extract_hotspots.py", SHORT_DESCRIPTION, [{
        "directories": [("CPU run directory", rocprof_sys_dir), ("GPU run directory", rocprofv3_dir)],
        "executable": run_info["executable"], "run_datetime": run_info["run_datetime"],
        "runtime": run_info["total_runtime"], "num_ranks": run_info["num_ranks"],
    }])
    sections = [
        (f"Combined hotspots (fused CPU+GPU ranking) -- showing {fused_desc}\n",
         render_table(FUSED_HOTSPOTS_COLUMNS, fused_selected) + "\n"
         + pct_total_note("entry", fused_unit)
         + ranking_note(unfiltered, extra_clause=" GPU kernels are unaffected (already leaf events).")
         + "  - This fused ranking excludes CPU-side time spent blocked in "
           "hipStreamSynchronize/hipDeviceSynchronize, to avoid counting GPU execution time "
           "twice -- once as CPU-side wait time, once as GPU-side kernel time. See the GPU API "
           "table below for that (and every other GPU-API call rocprof-sys saw).\n"
         + "  - Combined-pool arithmetic:\n"
           f"    CPU run raw total:              {info['cpu_total_raw']:.6f} sec\n"
           f"    - GPU sync-wait time:           {info['gpu_api_overhead_sec']:.6f} sec\n"
           f"    = CPU pure-compute total:        {info['cpu_pure_total_sec']:.6f} sec\n"
           f"    + GPU kernel total:              {info['gpu_total_sec']:.6f} sec\n"
           f"    = combined pool:                 {info['combined_total_sec']:.6f} sec\n"),
        (f"CPU compute hotspots (rocprof-sys run) -- showing {cpu_desc}\n",
         render_table(CPU_HOTSPOTS_COLUMNS, cpu_selected) + "\n"
         + pct_total_note("entry", cpu_unit)
         + ranking_note(unfiltered)),
        (f"GPU kernel hotspots (rocprofv3 run) -- showing {gpu_desc}\n",
         render_table(GPU_HOTSPOTS_COLUMNS, gpu_selected) + "\n"
         + pct_total_note("entry", gpu_unit)),
        (f"GPU API / launch overhead (rocprof-sys run) -- showing {gpu_api_desc}\n",
         render_table(CPU_HOTSPOTS_COLUMNS, gpu_api_selected) + "\n"
         + pct_total_note("entry", cpu_unit)
         + "  - Every ROCm-library call rocprof-sys saw is listed here, for digging in -- but "
           "only hipStreamSynchronize/hipDeviceSynchronize (the two calls that actually mean "
           "\"block the CPU until the GPU catches up\") are subtracted out of the combined "
           "hotspots table's pool above; the rest (e.g. hsakmt_ioctl, rocr::* runtime-internal "
           "busy-wait/event threads) is shown here but deliberately left out of that "
           "subtraction -- see that table's combined-pool arithmetic.\n"),
        (cpu_imbalance_title, cpu_imbalance_body),
        (gpu_imbalance_title, gpu_imbalance_body),
    ]

    return write_report_file(dest_path, render_report(header, sections, command_line))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rocprof_sys_dir", help="rocprof-sys output directory (CPU side), or a "
                                                 "combined run directory containing both a "
                                                 "rocprof-sys/ and a rocprofv3/ subdir")
    parser.add_argument("rocprofv3_dir", nargs="?", default=None,
                         help="rocprofv3 output directory (GPU side) -- omit when rocprof_sys_dir "
                              "already contains both subdirs")
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
    parser.add_argument("--extra-noise-config", dest="extra_noise_config", default=None,
                         help="path to a JSON file customizing noise-tag patterns (add/remove "
                              "substrings, disable a tag) -- see stage6_noise_config.py's "
                              "configure() for the file schema; falls back to "
                              "$FRIENDLY_ROCPROF_NOISE_CONFIG if not given")
    args = parser.parse_args(argv)

    cpu_dir, gpu_dir = resolve_two_dirs(args.rocprof_sys_dir, args.rocprofv3_dir)
    if not os.path.isdir(cpu_dir):
        raise SystemExit(f"error: no such directory: {cpu_dir!r}")
    if gpu_dir is None:
        raise SystemExit(
            f"error: no rocprofv3 directory given, and none found alongside {args.rocprof_sys_dir!r} "
            "-- pass it explicitly as a second argument"
        )
    if not os.path.isdir(gpu_dir):
        raise SystemExit(f"error: no such directory: {gpu_dir!r}")

    stage6_noise_config.configure(args.extra_noise_config or os.environ.get("FRIENDLY_ROCPROF_NOISE_CONFIG"))

    dest = args.dest or os.path.join(args.rocprof_sys_dir, "hotspots.txt")
    tokens = [os.path.abspath(args.rocprof_sys_dir)]
    if args.rocprofv3_dir:
        tokens.append(os.path.abspath(args.rocprofv3_dir))
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.top is not None:
        tokens += ["--top", str(args.top)]
    elif args.threshold is not None:
        tokens += ["--threshold", str(args.threshold)]
    elif args.show_all:
        tokens += ["--all"]
    if args.unfiltered:
        tokens += ["--unfiltered"]
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    command_line = command_header(sys.argv[0], tokens)

    write_report(cpu_dir, gpu_dir, dest, top=args.top, threshold=args.threshold,
                 show_all=args.show_all, unfiltered=args.unfiltered, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
