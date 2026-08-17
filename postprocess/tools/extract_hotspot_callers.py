#!/usr/bin/env python3
"""Top hotspots plus, for each one, its caller chain(s) back to a real program root -- CPU-only, or
fused CPU+GPU when a rocprofv3 directory is paired in.

The inverse view of the calltree tools: extract_calltree.py/extract_calltree_traced.py walk DOWN
from roots, rendering every descendant; this walks UP from a specific hot function (or, when GPU
kernel data is paired in, a specific kernel) to however many distinct root-to-it ancestor chains
exist in the data. Built almost entirely from existing stage1-6 primitives --
stage4_rocprofsys_flat.aggregate()/stage5_fused_hotspots_table.build_combined_view() for the ranked
hotspot selection (same as extract_CPU_hotspots.py/extract_hotspots.py) and
stage5_tree_render.load_rank_trees()/stage4_rocprofsys_tree.merge_rank_trees() for the same merged
call tree the calltree tools build.

Two new primitives this tool's two phases needed, both in stage4_rocprofsys_tree.py:
caller_chains_for_label() (the upward walk itself) and, for GPU kernels, a `parent` link on
synthetic kernel nodes plus attach_kernel_summaries()'s `collect_into` param -- once an attached
kernel is a real member of the same flat row pool, caller_chains_for_label() finds and walks it
exactly like any CPU function, with no kernel-specific tree-walking code needed at all.

Reuses extract_CPU_hotspots.gather_run_info()/extract_GPU_hotspots.gather_run_info() and
stage5_calltree_view.strip_wrapper_noise() directly, the same cross-tool-module reuse
extract_calltree.py itself already relies on -- an established pattern, not new here.

Functions: write_report(), main().
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
from stage1_run_dirs import resolve_two_dirs
from stage3_rocprofsys import make_is_pruned
from stage4_rocprofsys_flat import aggregate
from stage4_rocprofsys_tree import caller_chains_for_label, flatten_tree, make_node_values, merge_rank_trees
from stage5_calltree_view import strip_wrapper_noise
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_fused_hotspots_table import FUSED_HOTSPOTS_COLUMNS, build_combined_view
from stage5_table_render import pct_total_note, ranking_note, render_table, select_entries
from stage5_tree_render import (
    REPORT_HEADERS,
    aggregation_note,
    attach_and_render_gpu_kernels,
    format_aligned_rows,
    load_rank_trees,
    pair_gpu_per_rank,
)
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Top hotspots (CPU, or fused CPU+GPU), each followed by its own caller chain(s) back to the\n"
    "program's entry point.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys output directory and writes a report showing your top hotspots, then, for
each one, exactly how the program reached it -- the chain of callers from the program's entry
point down to that function (or, when a GPU run is paired in below, down through the CPU call
site that launched a hot kernel), for every distinct place it was reached from.

Where extract_CPU_hotspots.py/extract_hotspots.py tell you WHICH functions or kernels are hot,
this tells you WHERE in the program they're reached from -- useful when a hot function's name
alone doesn't say enough (a generic helper, a templated routine, something called from more than
one place) and you need to see its actual call path(s) to know what to do about it.

Pass a second, rocprofv3 directory to rank CPU functions and GPU kernels together (same fused
pool as extract_hotspots.py) and get a kernel's own caller chain traced back through whichever
CPU call site launched it -- omit it for a CPU-only report. Ranking, selection, and
noise-filtering otherwise behave like extract_CPU_hotspots.py's own CPU table -- see its --help
for details on --unfiltered/--top/--threshold/--all and --extra-noise-config.

Under the hood, this parses output written by AMD's rocprof-sys (and, optionally, rocprofv3) --
see https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def _render_chain(chain, node_values, max_depth):
    """One caller chain (root-first, see caller_chains_for_label()) as (label_text, values)
    rows, indented with the calltree tools' own connector style. max_depth counts UPWARD from
    the target function itself, not downward from the root -- the nearest max_depth callers are
    what matters when chasing a specific already-known hotspot, unlike the calltree tools' own
    --max-depth (root-relative), so a truncated chain hides its farthest-from-target ancestors,
    replacing them with one marker row (values=None), same convention as render_node()'s own
    "N more node(s) hidden" marker, just pointed the other direction."""
    truncated = max_depth is not None and len(chain) > max_depth + 1
    visible = chain[-(max_depth + 1):] if truncated else chain
    rows = []
    if truncated:
        hidden = len(chain) - len(visible)
        rows.append((
            f"... ({hidden} more ancestor(s) hidden above this point, raise --max-depth to see them)",
            None,
        ))
    prefix = ""
    for i, node in enumerate(visible):
        is_real_root_line = not truncated and i == 0
        text = node["label"] if is_real_root_line else f"{prefix}└── {node['label']}"
        rows.append((text, node_values(node)))
        prefix += "    "
    return rows


def _prepare_pct_total(field, total):
    """The same tiny "recompute pct_total against this specific denominator" closure
    extract_hotspots.py's own write_report() builds locally -- one CPU-only denominator
    (total_runtime) or one fused-pool denominator (info["combined_total_sec"]) here."""
    def _prepare(entries):
        for e in entries:
            e["pct_total"] = (e[field] / total * 100.0) if total > 0 else None
    return _prepare


def write_report(output_dir, gpu_dir, dest_path, top=None, threshold=None, show_all=False,
                  unfiltered=False, max_depth=None, command_line=""):
    key_field = "sum" if unfiltered else "self_sum"

    if gpu_dir is not None:
        fused_entries, _cpu_entries, _cpu_gpu_api_entries, _gpu_entries, info = build_combined_view(output_dir, gpu_dir)
        if not info["cpu_scanned"]:
            raise SystemExit(f"error: no rocprof-sys timemory data found under {output_dir!r} -- nothing to report")
        if not info["gpu_scanned"]:
            raise SystemExit(f"error: no rocprofv3 kernel_stats.csv found under {gpu_dir!r} -- nothing to combine")

        scanned_files = info["cpu_scanned"]
        threshold_unit = "of the combined CPU+GPU pool (double-counting-corrected)"
        selected, desc = select_entries(
            fused_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
            show_all=show_all, threshold_unit=threshold_unit,
            prepare=_prepare_pct_total(key_field, info["combined_total_sec"]),
        )
        columns, entry_noun = FUSED_HOTSPOTS_COLUMNS, "entry"
        table_title = f"Top hotspots (fused CPU+GPU) -- showing {desc}\n"
        table_note = (
            "  - This fused ranking excludes CPU-side time spent blocked in "
            "hipStreamSynchronize/hipDeviceSynchronize, to avoid counting GPU execution time twice "
            "-- see extract_hotspots.py's own combined-pool arithmetic for the full breakdown.\n"
        )
        cpu_run_info = cpu_tool.gather_run_info(output_dir, scanned_files)
        gpu_run_info = gpu_tool.gather_run_info(gpu_dir, info["gpu_scanned"])
        run_info = {
            key: cpu_run_info[key] if cpu_run_info[key] is not None else gpu_run_info[key]
            for key in ("executable", "run_datetime", "total_runtime", "num_ranks")
        }
        directories = [("CPU run directory", output_dir), ("GPU run directory", gpu_dir)]
    else:
        cpu_entries, _gpu_entries, scanned_files, total_runtime = aggregate(output_dir)
        if not scanned_files:
            raise SystemExit(f"error: no rocprof-sys timemory text table found in {output_dir!r} -- nothing to report")

        threshold_unit = "of total runtime"
        selected, desc = select_entries(
            cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
            show_all=show_all, threshold_unit=threshold_unit, prepare=_prepare_pct_total(key_field, total_runtime),
        )
        columns, entry_noun = CPU_HOTSPOTS_COLUMNS, "function"
        table_title = f"Top CPU hotspots -- showing {desc}\n"
        table_note = ""
        run_info = cpu_tool.gather_run_info(output_dir, scanned_files)
        directories = [("source directory", output_dir)]

    # wall_clock preferred over sampling_wall_clock (the opposite of extract_calltree.py's own
    # default): the hotspots selected above are ranked from aggregate()'s wall_clock-wins-per-label
    # merge, so a hotspot that's exactly instrumented needs wall_clock's own tree, not sampling's --
    # a function hot enough to matter is sometimes too short/rare for a fixed-interval sampler to
    # ever catch as its own distinct frame, even though it's a real, exactly-instrumented node here.
    ranks = load_rank_trees(
        output_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt", postprocess=strip_wrapper_noise,
    )
    if not ranks:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found under {output_dir!r} "
            "(expected files like wall_clock-<pid>.txt or sampling_wall_clock-<pid>.txt) -- "
            "nothing to trace caller chains from"
        )
    rank_keys = [rank_key for rank_key, _rows, _roots in ranks]
    node_values = make_node_values(rank_keys)
    merged_roots = merge_rank_trees(ranks)
    flat = flatten_tree(merged_roots)

    fallback_text = ""
    if gpu_dir is not None:
        # gpu_api hidden by default from anchor resolution, matching every other tool's default --
        # this tool has no --show-gpu-api flag of its own (yet); the chains walked below aren't
        # filtered by this at all, only which node a kernel attaches under is.
        is_pruned = make_is_pruned({"gpu_api"})
        gpu_per_rank = pair_gpu_per_rank(gpu_dir, output_dir, rank_keys)
        fallback_text = attach_and_render_gpu_kernels(
            flat, gpu_per_rank, gpu_dir, rank_keys, is_pruned, node_values, collect_into=flat,
        )

    header = standard_header("extract_hotspot_callers.py", SHORT_DESCRIPTION, [{
        "directories": directories, "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": run_info["num_ranks"], "scanned_files": scanned_files,
    }])

    sections = [
        (table_title,
         render_table(columns, selected) + "\n"
         + ranking_note(unfiltered, extra_clause=" GPU kernels are unaffected (already leaf events)." if gpu_dir else "")
         + pct_total_note(entry_noun, threshold_unit)
         + table_note),
        (None, "Caller-chain columns below share the calltree tools' own aggregation convention:\n"
               + aggregation_note()),
    ]
    if fallback_text:
        sections.append((None, fallback_text))

    for entry in selected:
        label = entry["label"]
        chains = caller_chains_for_label(flat, label)
        if not chains:
            body = (
                "  (no caller chain found -- every call-tree node for this entry was removed as "
                "noise, it was never sampled/instrumented as its own distinct call-tree frame, or "
                "-- for a GPU kernel -- no owning subroutine or launch call site could be found)\n"
            )
        else:
            rows_out = []
            for i, chain in enumerate(chains):
                if len(chains) > 1:
                    rows_out.append((f"-- call site {i + 1} of {len(chains)} --", None))
                rows_out.extend(_render_chain(chain, node_values, max_depth))
            body = format_aligned_rows(rows_out, REPORT_HEADERS)
        sections.append((f"Caller chain(s) for '{label}'\n", body))

    footer = help_redirect(
        "ranking/selection flags and how caller chains are found",
        script_name="extract_hotspot_callers.py",
    ) + command_line

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys output directory to read, or a combined "
                                            "run directory containing both a rocprof-sys/ and a "
                                            "rocprofv3/ subdir")
    parser.add_argument("gpu_dir", nargs="?", default=None,
                         help="rocprofv3 output directory (GPU side) -- omit when output_dir "
                              "already contains both subdirs, or when there's no GPU data to pair; "
                              "when given (or auto-resolved), CPU functions and GPU kernels are "
                              "ranked together and a hot kernel gets its own caller chain traced "
                              "back through its launching CPU call site")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspot-callers report "
                              "(default: <output_dir>/hotspot_callers.txt)")
    stage6_cli_common.add_selection_args(parser, "hotspots", "of total runtime", singular_noun="hotspot",
                                          top_help_suffix=" caller chains for (default: 10)", verb="show")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="rank CPU-side entries by inclusive (total) time instead of self "
                              "time -- a function that just calls other functions can still rank "
                              "high this way")
    stage6_cli_common.add_max_depth_arg(
        parser, help_text="truncate each caller chain to its N nearest callers, counting upward "
                           "from the hotspot itself (default: unlimited, show the whole chain "
                           "back to the program's entry point)",
    )
    stage6_noise_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    cpu_dir, gpu_dir = resolve_two_dirs(args.output_dir, args.gpu_dir)
    stage6_cli_common.require_directories([cpu_dir, gpu_dir])
    stage6_noise_config.configure_from_args(args)

    if args.top is None and args.threshold is None and not args.show_all:
        args.top = 10

    dest = stage6_cli_common.resolve_dest(args.dest, args.output_dir, "hotspot_callers.txt")
    tokens = [os.path.abspath(args.output_dir)]
    if args.gpu_dir:
        tokens.append(os.path.abspath(args.gpu_dir))
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
    if args.max_depth is not None:
        tokens += ["--max-depth", str(args.max_depth)]
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    command_line = command_header(sys.argv[0], tokens)

    write_report(cpu_dir, gpu_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 unfiltered=args.unfiltered, max_depth=args.max_depth, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
