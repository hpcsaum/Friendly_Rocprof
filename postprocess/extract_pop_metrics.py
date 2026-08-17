#!/usr/bin/env python3
"""Compute POP-inspired parallel-efficiency metrics from one or more rocprof-sys
(+ optional rocprofv3) output directories.

See docs/pop_metrics_reference.md for the full metric hierarchy and which POP
metrics are/aren't derivable from this data. In short: Load Balance,
Communication Efficiency, and Parallel Efficiency come from a single run;
Computation Efficiency and Global Efficiency need a scaling study (2+ runs,
compared against the first as reference). Serialisation/Transfer Efficiency
(need Dimemas) and Instruction/IPC Scaling (need PAPI hardware counters) are
NOT computed here -- see that doc for why.

Functions: write_report(), main().
"""

import argparse
import os
import sys

from stage5_pop_metrics_table import compute_run_metrics, format_metrics_table, metrics_legend, run_label
import stage6_noise_config
from stage6_noise_config import load_default_patterns
from stage6_report_builder import command_header, help_redirect, render_report, standard_header, write_report_file

TAG_DEFS = load_default_patterns()

SHORT_DESCRIPTION = (
    "Computes POP-inspired parallel efficiency metrics (Load Balance, Communication/Parallel/\n"
    "Computation/Global Efficiency) from one or more runs.\n"
)

HELP_BLURB = f"""\
Reads one or more rocprof-sys (optionally paired with rocprofv3) output
directories from the same program and computes POP-inspired parallel
efficiency metrics: Load Balance, Communication Efficiency, and Parallel
Efficiency from a single run; Computation Efficiency and Global Efficiency
when 2+ runs from a scaling study (e.g. different -np counts) are given,
compared against the first directory as the reference.

This does NOT compute every POP metric -- Serialisation/Transfer Efficiency
need a Dimemas-style network simulation, and Instruction/IPC Scaling need
PAPI hardware counters; neither is available from rocprof-sys's own output.
See docs/pop_metrics_reference.md for the full picture.

Communication time is classified by function-name prefix (case-insensitive:
{', '.join(TAG_DEFS['mpi_territory']['prefixes'])}) or Fortran-shim suffix
({', '.join(TAG_DEFS['mpi_territory']['suffixes'])}). MPICH/Cray-MPICH coverage is solid; Open MPI
coverage is narrower, based on its own naming convention rather than a
captured Open MPI run. CPU<->GPU per-rank pairing (when a rocprofv3
directory is given) assumes matching sorted-filename order between the two
directories -- not cross-checked.

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def write_report(run_dirs, dest_path, scaling=None, command_line=""):
    all_metrics = [compute_run_metrics(d) for d in run_dirs]
    multi_run = len(all_metrics) > 1

    runs = []
    for i, m in enumerate(all_metrics):
        label = "reference run" if i == 0 else f"scaling run {i + 1} ({run_label(m['run_dir'])})"
        pool = "CPU+GPU combined" if m["gpu_dir"] else "CPU-only"
        runs.append({
            "directories": [(label, m["run_dir"])], "num_ranks": m["num_ranks"],
            "extra_lines": [f"  pool: {pool}\n"],
        })
    header = standard_header("extract_pop_metrics.py", SHORT_DESCRIPTION, runs)

    table_text, show_gpu_cols, show_gpu_eff = format_metrics_table(all_metrics, scaling)
    sections = [(None, table_text)]

    footer = (
        metrics_legend(show_gpu_cols, show_gpu_eff, multi_run, scaling)
        + "\n"
        + help_redirect("what's not computed and classification caveats", script_name="extract_pop_metrics.py")
        + command_line
    )

    return write_report_file(dest_path, render_report(header, sections, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir", help="reference run's output directory")
    parser.add_argument("scaled_dirs", nargs="*", help="additional runs from the same scaling study, compared against reference_dir")
    parser.add_argument("--scaling", choices=["strong", "weak"], default=None,
                         help="required when scaled_dirs is non-empty: 'strong' (fixed global problem "
                              "size) or 'weak' (fixed problem size per rank)")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the report (default: <reference_dir>/pop_metrics.txt)")
    parser.add_argument("--extra-noise-config", dest="extra_noise_config", default=None,
                         help="path to a JSON file customizing noise-tag patterns (add/remove "
                              "substrings, disable a tag) -- see stage6_noise_config.py's "
                              "configure() for the file schema; falls back to "
                              "$FRIENDLY_ROCPROF_NOISE_CONFIG if not given")
    args = parser.parse_args(argv)

    run_dirs = [args.reference_dir] + args.scaled_dirs
    for d in run_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"error: no such directory: {d!r}")

    if args.scaled_dirs and args.scaling is None:
        raise SystemExit("error: --scaling {strong,weak} is required when scaled_dirs are given")

    stage6_noise_config.configure(args.extra_noise_config or os.environ.get("FRIENDLY_ROCPROF_NOISE_CONFIG"))

    dest = args.dest or os.path.join(args.reference_dir, "pop_metrics.txt")
    tokens = [os.path.abspath(d) for d in run_dirs]
    if args.scaling:
        tokens += ["--scaling", args.scaling]
    if args.dest:
        tokens += ["-o", os.path.abspath(args.dest)]
    if args.extra_noise_config:
        tokens += ["--extra-noise-config", os.path.abspath(args.extra_noise_config)]
    command_line = command_header(sys.argv[0], tokens)

    write_report(run_dirs, dest, scaling=args.scaling, command_line=command_line)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
