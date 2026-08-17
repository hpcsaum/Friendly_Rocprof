#!/usr/bin/env python3
"""Render an indented call tree from rocprof-sys timemory text output, optionally
with rocprofv3 GPU kernel data nested in at the CPU call site(s) that launched them.

This is the fast, exact-where-instrumented variant: it prefers wall_clock-<pid>.txt
(GOTCHA-instrumented -- exact timing for every intercepted call) per rank, falling
back to sampling_wall_clock-<pid>.txt only for a rank with no wall_clock file at all.
Its tree is therefore only as deep as rocprof-sys's own instrumentation boundaries --
a real intermediate frame that was never instrumented is invisible, so a child can
appear to hang directly off a much higher ancestor than it really does. See
`extract_calltree.py` (the sampling-based tool) for the true, deeper call structure,
at the cost of needing noise filtering and only statistically-approximate timing.

Unlike extract_CPU_hotspots.py's scan_ranks()/aggregate(), this does NOT merge
same-label rows within one rank's own file -- a calltree needs every individual
call-tree node kept distinct (attach_ancestry()'s parent links intact), not summed
by label. It DOES merge structurally ACROSS ranks (see stage4_rocprofsys_tree.merge_rank_trees())
into one aggregated tree -- a global view, not one call tree per rank -- with each
node's CALLS/SELF/TOTAL columns averaged (and self-time's load imbalance shown via
std_dev/min/max) across every rank, the same avg/std_dev/min/max convention
extract_CPU_hotspots.compute_load_imbalance() already uses elsewhere in this codebase.
Kernel placement is never per-dispatch-exact: this toolchain's text/JSON output has
no per-call timestamps to correlate against, only the binary Perfetto trace does,
and there's no stdlib-friendly way to parse it -- a future capability, not silently
dropped.

Functions: write_report(), main().
"""

import argparse
import os
import sys

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool
from stage1_run_dirs import resolve_two_dirs
from stage5_calltree_traced_view import build_calltree_view
from stage5_tree_render import aggregation_note, tree_view_note
import stage6_noise_config
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

SHORT_DESCRIPTION = (
    "Renders an aggregated call tree (GOTCHA-instrumented, exact timing) across every rank,\n"
    "optionally with GPU kernel data nested in.\n"
)

HELP_BLURB = """\
Reads a rocprof-sys output directory (optionally paired with rocprofv3, either
as a nested rocprofv3/ subdir or as a second, separately-run directory) and
writes an indented call tree -- actual function nesting, not a flat ranked
list -- aggregated into one global view across every rank (not one tree per
rank): each line's CALLS/TOTAL-AVG(s) are averaged, and SELF gets a full
avg/std_dev/min/max load-balance breakdown, the same convention this
codebase's other load-imbalance tables already use. Works on tool 1/3 output
(profile_CPU_hotspots.sh/profile_hotspots.sh) and tool 4's scan directory
(instrument_hotspots.sh trace) identically -- same underlying data format
either way.

This is the fast, exact-where-instrumented variant: it prefers
wall_clock-<pid>.txt (GOTCHA-instrumented) per rank, so timing is exact for
every intercepted call, but the tree is only as deep as rocprof-sys's own
instrumentation boundaries -- a real but never-instrumented intermediate
frame is invisible, so a child can appear to hang directly off a much higher
ancestor than it really does. See extract_calltree.py (the sampling-based
tool) for the true, deeper call structure instead, at the cost of needing
noise filtering and only statistically-approximate timing.

Filtering matches tool 3's "CPU compute hotspots" bucket: GPU-API/runtime
noise (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-prefixed calls, plus
kernel-descriptor sampling artifacts -- labels ending in ".kd") is hidden by
default, so what's left is your own code plus MPI calls. These ".kd" rows
happen when rocprof-sys's own sampling attributes a GPU kernel launch to its
compiled kernel-descriptor symbol directly in the CPU call tree, at
near-zero duration -- duplicating the same kernel's real device time already
shown separately, hence hidden the same way, for the same reason. Pass
--show-gpu-api to see the hidden chain too.

When a paired rocprofv3 directory is found, real GPU kernel data is nested
into the tree at the CPU subroutine that actually contains it, matched by
name when the compiler's own kernel naming allows it, or a structural
estimate (nearest launch-call ancestor, proportionally split by launch-call
count when there's more than one candidate site) otherwise. NEITHER is
per-dispatch-exact: this toolchain's text/JSON output has no per-call
timestamps to correlate against, only the binary Perfetto trace does, and
that has no stdlib-friendly Python parser (a future tool, not attempted
here).

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def write_report(cpu_dir, gpu_dir, dest_path, max_depth=None, show_gpu_api=False, command_line=""):
    view = build_calltree_view(cpu_dir, cpu_dir, gpu_dir, max_depth=max_depth, show_gpu_api=show_gpu_api)
    rank_keys = view["rank_keys"]

    # One shared identity for the run being profiled -- CPU's own metadata is preferred; GPU's
    # (when a GPU directory was actually given) is used only for whichever field CPU didn't have.
    # MPI ranks comes from the tree merge itself (len(rank_keys)), ground truth rather than a guess.
    run_info = cpu_tool.gather_run_info(cpu_dir, [])
    directories = [("CPU run directory", cpu_dir)]
    if gpu_dir is not None:
        directories.append(("GPU run directory", gpu_dir))
        gpu_run_info = gpu_tool.gather_run_info(gpu_dir, [])
        run_info = {
            key: run_info[key] if run_info[key] is not None else gpu_run_info[key]
            for key in ("executable", "run_datetime", "total_runtime")
        }

    header = standard_header("extract_calltree_traced.py", SHORT_DESCRIPTION, [{
        "directories": directories, "executable": run_info["executable"],
        "run_datetime": run_info["run_datetime"], "runtime": run_info["total_runtime"],
        "num_ranks": len(rank_keys),
    }])
    tree_notes = aggregation_note() + tree_view_note(rank_keys, max_depth, show_gpu_api)
    sections = [(None, view["tree_text"] + "\n" + tree_notes)]
    if view["fallback_text"]:
        sections.append((None, view["fallback_text"]))

    footer = help_redirect(
        "noise filtering, instrumentation depth, and kernel-placement caveats",
        script_name="extract_calltree_traced.py",
    ) + command_line

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="rocprof-sys output directory to read, or a combined "
                                            "run directory containing both a rocprof-sys/ and a "
                                            "rocprofv3/ subdir")
    parser.add_argument("gpu_dir", nargs="?", default=None,
                         help="rocprofv3 output directory (GPU side) -- omit when output_dir "
                              "already contains both subdirs, or when there's no GPU data to pair")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the call tree report (default: <output_dir>/calltree_traced.txt)")
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=None,
                         help="truncate the tree at this depth (default: unlimited, print the whole tree)")
    parser.add_argument("--show-gpu-api", dest="show_gpu_api", action="store_true",
                         help="also show GPU-API/runtime calls (hip/hsa/roctx/kfd/rocdecode/rocjpeg/rocr-"
                              "prefixed) instead of hiding them")
    parser.add_argument("--extra-noise-config", dest="extra_noise_config", default=None,
                         help="path to a JSON file customizing noise-tag patterns (add/remove "
                              "substrings, disable a tag) -- see stage6_noise_config.py's "
                              "configure() for the file schema; falls back to "
                              "$FRIENDLY_ROCPROF_NOISE_CONFIG if not given")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")
    cpu_dir, gpu_dir = resolve_two_dirs(args.output_dir, args.gpu_dir)
    if gpu_dir is not None and not os.path.isdir(gpu_dir):
        raise SystemExit(f"error: no such directory: {gpu_dir!r}")

    stage6_noise_config.configure(args.extra_noise_config or os.environ.get("FRIENDLY_ROCPROF_NOISE_CONFIG"))

    dest = args.dest or os.path.join(args.output_dir, "calltree_traced.txt")
    tokens = [os.path.abspath(args.output_dir)]
    if args.gpu_dir:
        tokens.append(os.path.abspath(args.gpu_dir))
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.max_depth is not None:
        tokens += ["--max-depth", str(args.max_depth)]
    if args.show_gpu_api:
        tokens.append("--show-gpu-api")
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    command_line = command_header(sys.argv[0], tokens)

    write_report(cpu_dir, gpu_dir, dest, max_depth=args.max_depth, show_gpu_api=args.show_gpu_api,
                 command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
