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
"""

import argparse
import os
from datetime import datetime

from stage3_rocprofsys import load_default_patterns
from stage5_pop_metrics_table import compute_run_metrics, format_metrics_table, run_label

TAG_DEFS = load_default_patterns()

HELP_BLURB = """\
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

Under the hood, this parses output written by AMD's rocprof-sys (and,
optionally, rocprofv3) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def write_report(run_dirs, dest_path, scaling=None):
    all_metrics = [compute_run_metrics(d) for d in run_dirs]
    multi_run = len(all_metrics) > 1

    parts = []
    parts.append("POP-inspired parallel efficiency metrics report\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"reference run: {os.path.abspath(run_dirs[0])}\n")
    for m in all_metrics:
        parts.append(f"  - {run_label(m['run_dir'])}: {m['num_ranks']} rank(s)")
        parts.append(", CPU+GPU combined pool" if m["gpu_dir"] else ", CPU-only pool")
        parts.append("\n")
    parts.append("\n")

    table_text, show_gpu_cols, show_gpu_eff = format_metrics_table(all_metrics, scaling)
    parts.append(table_text)

    parts.append("\n")
    parts.append("Metric explanation:\n")
    parts.append("  - LB    = avg / max useful compute time across ranks\n")
    parts.append(
        "  - CommE = max useful compute time / max total elapsed time across ranks "
        "(direct formula, not Dimemas's Serialisation x Transfer split -- see docs/pop_metrics_reference.md)\n"
    )
    parts.append("  - PE    = LB x CommE\n")
    if show_gpu_cols:
        parts.append(
            "  - GPU-Util = max GPU busy time / max total elapsed time across ranks -- NOT an official POP "
            "metric; the GPU's raw share of wall-clock time, INCLUDING any idling caused by growing "
            "communication overhead -- unlike GPU-Off, this drops when CommE drops too, since a "
            "comm-starved GPU is genuinely less utilized, whatever the root cause\n"
        )
        parts.append(
            "  - GPU-Off = 1 - (max non-offloaded CPU compute time / max total elapsed time) across ranks -- "
            "NOT an official POP metric; how much of the critical-path rank's time is still CPU-only "
            "compute (serial, not-yet-ported, or not-worth-porting code) -- deliberately excludes "
            "communication time, already covered by CommE\n"
        )
        parts.append(
            "  - GPU-LB = avg / max GPU busy time across ranks -- NOT an official POP metric; load balance "
            "between GPUs specifically, separate from LB's whole CPU+GPU pool\n"
        )
    if multi_run:
        if scaling == "weak":
            parts.append(
                "  - CompE = avg per-rank useful compute time (reference) / avg per-rank useful compute time (this run) "
                "-- weak scaling's total is expected to grow with rank count even at perfect efficiency, "
                "so only the average is meaningful\n"
            )
        else:
            parts.append(
                "  - CompE = total useful compute time (reference) / total useful compute time (this run), summed "
                "across ranks -- strong scaling's ideal keeps this total constant as rank count grows\n"
            )
        parts.append("  - GE    = PE x CompE\n")
    else:
        parts.append(
            "  - CompE, GE need a scaling study (2+ directories, compared against the first as reference) "
            "-- pass additional directories to see them\n"
        )
    if show_gpu_eff:
        if scaling == "weak":
            parts.append(
                "  - GPU-Eff = avg per-rank GPU busy time (reference) / avg per-rank GPU busy time (this run) -- "
                "NOT an official POP metric; isolates whether it's specifically the GPU's own contribution "
                "that stopped scaling, as opposed to CompE's whole-pool view\n"
            )
        else:
            parts.append(
                "  - GPU-Eff = total GPU busy time (reference) / total GPU busy time (this run), summed across "
                "ranks -- NOT an official POP metric; a low value in strong scaling flags the per-rank "
                "problem size shrinking below what keeps the GPU saturated\n"
            )
    parts.append("\n")

    parts.append(
        "Not computed (see docs/pop_metrics_reference.md for why):\n"
        "  - Serialisation Efficiency / Transfer Efficiency: need a Dimemas-style ideal-network\n"
        "    simulation, not part of this toolchain.\n"
        "  - Instruction Scaling / IPC Scaling: need PAPI hardware counters, not present in\n"
        "    rocprof-sys's output unless ROCPROFSYS_PAPI_EVENTS was explicitly configured.\n"
    )
    parts.append("\n")
    parts.append(
        "Caveats:\n"
        "  - Communication time is classified by function-name prefix (case-insensitive: "
        f"{', '.join(TAG_DEFS['mpi_territory']['prefixes'])}) or Fortran-shim suffix "
        f"({', '.join(TAG_DEFS['mpi_territory']['suffixes'])}) --\n"
        "    MPICH/Cray-MPICH prefixes are confirmed from real captured data; the Open MPI\n"
        "    prefixes (ompi_/opal_/orte_) are a probable addition, not yet confirmed against a\n"
        "    real Open MPI run, and may need refinement.\n"
        "  - CPU<->GPU per-rank pairing (when a rocprofv3 dir is present) assumes matching\n"
        "    sorted-filename order between the two directories -- not cross-checked.\n"
    )

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir", help="reference run's output directory")
    parser.add_argument("scaled_dirs", nargs="*", help="additional runs from the same scaling study, compared against reference_dir")
    parser.add_argument("--scaling", choices=["strong", "weak"], default=None,
                         help="required when scaled_dirs is non-empty: 'strong' (fixed global problem "
                              "size) or 'weak' (fixed problem size per rank)")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the report (default: <reference_dir>/pop_metrics.txt)")
    args = parser.parse_args(argv)

    run_dirs = [args.reference_dir] + args.scaled_dirs
    for d in run_dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"error: no such directory: {d!r}")

    if args.scaled_dirs and args.scaling is None:
        raise SystemExit("error: --scaling {strong,weak} is required when scaled_dirs are given")

    dest = args.dest or os.path.join(args.reference_dir, "pop_metrics.txt")
    write_report(run_dirs, dest, scaling=args.scaling)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
