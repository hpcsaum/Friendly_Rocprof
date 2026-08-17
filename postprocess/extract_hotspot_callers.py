#!/usr/bin/env python3
"""Top CPU hotspots plus, for each one, its caller chain(s) back to a real program root.

The inverse view of the calltree tools: extract_calltree.py/extract_calltree_traced.py walk DOWN
from roots, rendering every descendant; this walks UP from a specific hot function to however many
distinct root-to-it ancestor chains exist in the data (a function called from more than one call
site gets more than one chain). Built almost entirely from existing stage1-6 primitives --
stage4_rocprofsys_flat.aggregate() for the ranked hotspot selection (same as
extract_CPU_hotspots.py) and stage5_tree_render.load_rank_trees()/stage4_rocprofsys_tree.
merge_rank_trees() for the same merged call tree the calltree tools build; the one new primitive
this tool needed is stage4_rocprofsys_tree.caller_chains_for_label(), the actual upward walk.
Reuses extract_CPU_hotspots.gather_run_info() and stage5_calltree_view.strip_wrapper_noise()
directly, the same cross-tool-module reuse extract_calltree.py itself already relies on for
gather_run_info() -- an established pattern in this codebase, not new here.

Functions: write_report(), main().
"""

import argparse
import os
import sys

import extract_CPU_hotspots as cpu_tool
from stage4_rocprofsys_flat import aggregate
from stage4_rocprofsys_tree import caller_chains_for_label, flatten_tree, make_node_values, merge_rank_trees
from stage5_calltree_view import strip_wrapper_noise
from stage5_cpu_hotspots_table import CPU_HOTSPOTS_COLUMNS
from stage5_table_render import pct_total_note, ranking_note, render_table, select_entries
from stage5_tree_render import REPORT_HEADERS, aggregation_note, format_aligned_rows, load_rank_trees
import stage6_cli_common
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Top CPU hotspots, each followed by its own caller chain(s) back to the program's entry point.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys output directory and writes a report showing your top CPU hotspot functions,
then, for each one, exactly how the program reached it -- the chain of callers from the program's
entry point down to that function, for every distinct place in the code it was called from.

Where extract_CPU_hotspots.py tells you WHICH functions are hot, this tells you WHERE in the
program they're called from -- useful when a hot function's name alone doesn't say enough (a
generic helper, a templated routine, something called from more than one place) and you need to
see its actual call path(s) to know what to do about it.

Ranking, selection, and noise-filtering behave exactly like extract_CPU_hotspots.py's own CPU
table -- see its --help for details on --unfiltered/--top/--threshold/--all and
--extra-noise-config.

Under the hood, this parses output written by AMD's rocprof-sys (ROCm Systems Profiler) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
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


def write_report(output_dir, dest_path, top=None, threshold=None, show_all=False, unfiltered=False,
                  max_depth=None, command_line=""):
    cpu_entries, _gpu_entries, scanned_files, total_runtime = aggregate(output_dir)
    if not scanned_files:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found in {output_dir!r} -- "
            "nothing to report"
        )

    key_field = "sum" if unfiltered else "self_sum"
    threshold_unit = "of total runtime"

    def _set_pct_total(entries):
        for e in entries:
            e["pct_total"] = (e[key_field] / total_runtime * 100.0) if total_runtime > 0 else None

    selected, desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit=threshold_unit, prepare=_set_pct_total,
    )

    run_info = cpu_tool.gather_run_info(output_dir, scanned_files)

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

    header = standard_header("extract_hotspot_callers.py", SHORT_DESCRIPTION, [{
        "directories": [("source directory", output_dir)], "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": run_info["num_ranks"], "scanned_files": scanned_files,
    }])

    sections = [
        (f"Top CPU hotspots -- showing {desc}\n",
         render_table(CPU_HOTSPOTS_COLUMNS, selected) + "\n"
         + ranking_note(unfiltered)
         + pct_total_note("function", threshold_unit)),
        (None, "Caller-chain columns below share the calltree tools' own aggregation convention:\n"
               + aggregation_note()),
    ]

    for entry in selected:
        label = entry["label"]
        chains = caller_chains_for_label(flat, label)
        if not chains:
            body = (
                "  (no caller chain found -- every call-tree node for this function was removed "
                "as noise, or it was never sampled/instrumented as its own distinct call-tree frame)\n"
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
    parser.add_argument("output_dir", help="rocprof-sys output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspot-callers report "
                              "(default: <output_dir>/hotspot_callers.txt)")
    stage6_cli_common.add_selection_args(parser, "hotspots", "of total runtime", singular_noun="hotspot",
                                          top_help_suffix=" caller chains for (default: 10)", verb="show")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="rank by inclusive (total) time instead of self time -- a function "
                              "that just calls other functions can still rank high this way")
    stage6_cli_common.add_max_depth_arg(
        parser, help_text="truncate each caller chain to its N nearest callers, counting upward "
                           "from the hotspot itself (default: unlimited, show the whole chain "
                           "back to the program's entry point)",
    )
    stage6_noise_config.add_cli_argument(parser)
    args = parser.parse_args(argv)

    stage6_cli_common.require_directory(args.output_dir)
    stage6_noise_config.configure_from_args(args)

    if args.top is None and args.threshold is None and not args.show_all:
        args.top = 10

    dest = stage6_cli_common.resolve_dest(args.dest, args.output_dir, "hotspot_callers.txt")
    tokens = [os.path.abspath(args.output_dir)]
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

    write_report(args.output_dir, dest, top=args.top, threshold=args.threshold, show_all=args.show_all,
                 unfiltered=args.unfiltered, max_depth=args.max_depth, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
