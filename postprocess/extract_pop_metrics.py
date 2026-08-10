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
import statistics
from datetime import datetime

import extract_CPU_hotspots as cpu_tool
import extract_GPU_hotspots as gpu_tool

# MPICH / Cray-MPICH function-name prefixes -- covers both the user-facing MPI_*
# calls and PMPI_* (the profiling interface most GOTCHA-based tools actually
# intercept), plus MPIR_/MPID_ internal helpers that do real work on behalf of
# an MPI_* call (e.g. MPIR_Typerep_icopy, seen packing/copying data during a
# Waitall) -- these are communication overhead, not application compute, even
# though they aren't named "MPI_" themselves. Self-time (see compute_run_metrics)
# makes summing this prefix set safe regardless of nesting depth: a rank's total
# self-time across every call-tree node always equals its root's inclusive time,
# so there is no double-counting to worry about from, say, an MPI_Waitall row and
# a nested MPIR_Typerep_icopy row both matching this prefix set.
# Known limitation: MPICH/Cray-MPICH only. An Open MPI run's internal helpers
# (ompi_/opal_/orte_ prefixes) won't be recognized and will be misclassified as
# application compute instead of communication -- not yet configurable, may be
# extended later if a non-MPICH MPI implementation needs to be supported.
MPI_PREFIXES = ("MPI_", "PMPI_", "MPIR_", "MPID_")

# The two HIP calls that mean "block the CPU until the GPU catches up" -- same
# definition and same self-time-only rationale as extract_hotspots.py's own
# SYNC_WAIT_LABELS (duplicated here, not imported, matching this codebase's
# existing "small constants are duplicated across standalone tools" convention --
# see extract_GPU_hotspots.py's compute_load_imbalance docstring).
SYNC_WAIT_LABELS = {"hipStreamSynchronize", "hipDeviceSynchronize"}

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


def resolve_run_dirs(run_dir):
    """A "run" is one directory that may contain a rocprof-sys/ subdir (CPU
    timing) and/or a rocprofv3/ subdir (GPU kernel timing) -- the layout
    profile_hotspots.sh and instrument_hotspots.sh's scan mode already produce.
    Falls back to treating run_dir itself as the CPU dir when there's no
    rocprof-sys/ subdir (profile_CPU_hotspots.sh's un-nested layout).

    Returns (cpu_dir, gpu_dir_or_None). Does not check either actually
    contains data -- see compute_run_metrics()'s own empty-input handling.
    """
    cpu_subdir = os.path.join(run_dir, "rocprof-sys")
    gpu_subdir = os.path.join(run_dir, "rocprofv3")
    cpu_dir = cpu_subdir if os.path.isdir(cpu_subdir) else run_dir
    gpu_dir = gpu_subdir if os.path.isdir(gpu_subdir) else None
    return cpu_dir, gpu_dir


def gpu_sync_wait_per_rank(cpu_dir):
    """Per-rank self-time sum of hipStreamSynchronize/hipDeviceSynchronize --
    the two calls that mean "block the CPU until the GPU catches up" (see
    SYNC_WAIT_LABELS). Can't use cpu_tool.aggregate_per_rank() for this: it
    only returns CPU-classified rows (row["gpu"] is False), and both labels
    here start with "hip" -- one of GPU_API_PREFIXES -- so scan_ranks()
    classifies them as GPU rows and aggregate_per_rank() silently drops them.
    Goes one level lower, straight to scan_ranks(), to see those rows at all.
    """
    ranks = cpu_tool.scan_ranks(cpu_dir)
    return [
        sum(row["self_sum"] for row in r["rows"] if row["gpu"] and row["label"] in SYNC_WAIT_LABELS)
        for r in ranks
    ]


def compute_run_metrics(run_dir):
    """Load Balance, Communication Efficiency, and Parallel Efficiency for one
    run, plus the per-rank "useful compute time" totals a scaling comparison
    needs. Returns a dict -- see the bottom of this function for every key.

    Per rank: total_time is that rank's root/whole-program inclusive wall time
    (max of its inclusive-time dict, same "largest value is the root" heuristic
    scan_ranks() already relies on internally). comm_time is the self-time sum
    of every label matching MPI_PREFIXES -- self-time, not inclusive, so nested
    MPI-internal helper calls (e.g. MPIR_Typerep_icopy under a PMPI_Waitall)
    are counted exactly once, not double-counted with their parent.

    When a paired rocprofv3 dir is present, useful_compute follows
    extract_hotspots.py's own combined-pool arithmetic (CPU total minus GPU
    sync-wait time, plus real GPU kernel time), applied PER RANK instead of
    pooled across the whole run -- necessary for Load Balance/Communication
    Efficiency, which need per-rank granularity. This assumes rank i's
    rocprof-sys file corresponds to rank i's rocprofv3 file (matching
    sorted-filename order) -- not cross-checked, same "caller's
    responsibility" spirit as extract_hotspots.py's own pairing. If the two
    directories don't report the same number of ranks, the GPU side is
    skipped entirely for this run (falls back to CPU-only) with a warning,
    rather than risk combining mismatched ranks.
    """
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    self_per_rank, rank_keys = cpu_tool.aggregate_per_rank(cpu_dir)
    incl_per_rank, _ = cpu_tool.aggregate_per_rank(cpu_dir, unfiltered=True)

    if not rank_keys:
        if gpu_dir is not None and gpu_tool.aggregate_per_rank(gpu_dir)[1]:
            raise SystemExit(
                f"error: no rocprof-sys CPU-side timing found under {cpu_dir!r}, only GPU "
                f"kernel data under {gpu_dir!r} -- POP metrics need CPU-side timing (point "
                "this tool at a profile_hotspots.sh / profile_CPU_hotspots.sh / "
                "instrument_hotspots.sh scan output directory instead)"
            )
        raise SystemExit(f"error: no rocprof-sys timing data found under {cpu_dir!r} -- nothing to compute")

    gpu_per_rank = None
    sync_wait_per_rank = None
    if gpu_dir is not None:
        gpu_totals, gpu_scanned = gpu_tool.aggregate_per_rank(gpu_dir)
        if gpu_scanned and len(gpu_totals) == len(rank_keys):
            gpu_per_rank = gpu_totals
            sync_wait_per_rank = gpu_sync_wait_per_rank(cpu_dir)
        elif gpu_scanned:
            print(
                f"warning: {run_dir!r}: rocprof-sys reports {len(rank_keys)} rank(s) but "
                f"rocprofv3 reports {len(gpu_totals)} -- skipping GPU data for this run "
                "rather than risk pairing mismatched ranks",
            )

    per_rank = []
    for i, rank_key in enumerate(rank_keys):
        self_totals = self_per_rank[i]
        incl_totals = incl_per_rank[i]
        total_time = max(incl_totals.values()) if incl_totals else 0.0
        comm_time = sum(v for label, v in self_totals.items() if label.startswith(MPI_PREFIXES))

        if gpu_per_rank is not None:
            gpu_api_overhead = sync_wait_per_rank[i]
            cpu_pure = max(0.0, total_time - comm_time - gpu_api_overhead)
            useful_compute = cpu_pure + sum(gpu_per_rank[i].values())
        else:
            useful_compute = max(0.0, total_time - comm_time)

        per_rank.append({
            "rank_key": rank_key,
            "total_time": total_time,
            "comm_time": comm_time,
            "useful_compute": useful_compute,
        })

    useful_values = [r["useful_compute"] for r in per_rank]
    total_values = [r["total_time"] for r in per_rank]
    max_useful = max(useful_values)
    max_total = max(total_values)

    load_balance = (statistics.mean(useful_values) / max_useful) if max_useful > 0 else None
    comm_efficiency = (max_useful / max_total) if max_total > 0 else None
    parallel_efficiency = (
        load_balance * comm_efficiency if load_balance is not None and comm_efficiency is not None else None
    )

    return {
        "run_dir": run_dir,
        "cpu_dir": cpu_dir,
        "gpu_dir": gpu_dir if gpu_per_rank is not None else None,
        "num_ranks": len(rank_keys),
        "per_rank": per_rank,
        "load_balance": load_balance,
        "communication_efficiency": comm_efficiency,
        "parallel_efficiency": parallel_efficiency,
        "total_useful_compute": sum(useful_values),
        "avg_useful_compute": statistics.mean(useful_values),
    }


def compute_scaling_metrics(ref_metrics, run_metrics, scaling):
    """Computation Efficiency and Global Efficiency for run_metrics, relative
    to ref_metrics. Strong scaling (fixed global problem size) compares TOTAL
    useful compute time summed across ranks -- the POP page's own formula --
    since ideal strong scaling keeps that total constant as rank count grows.
    Weak scaling (fixed problem size per rank) instead compares AVERAGE
    per-rank useful compute time, since under weak scaling the total is
    expected to grow with rank count even at perfect efficiency -- only the
    average stays meaningful as a ratio.
    """
    if scaling == "weak":
        ref_value, run_value = ref_metrics["avg_useful_compute"], run_metrics["avg_useful_compute"]
    else:
        ref_value, run_value = ref_metrics["total_useful_compute"], run_metrics["total_useful_compute"]

    computation_efficiency = (ref_value / run_value) if run_value > 0 else None
    pe = run_metrics["parallel_efficiency"]
    global_efficiency = (pe * computation_efficiency) if pe is not None and computation_efficiency is not None else None
    return {"computation_efficiency": computation_efficiency, "global_efficiency": global_efficiency}


def run_label(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "n/a"


def write_report(run_dirs, dest_path, scaling=None):
    all_metrics = [compute_run_metrics(d) for d in run_dirs]
    ref_metrics = all_metrics[0]
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

    row_labels = [
        run_label(m["run_dir"]) + (" (reference)" if multi_run and m is ref_metrics else "")
        for m in all_metrics
    ]
    label_width = max(len("run"), *(len(label) for label in row_labels))

    header_title = "=== Metrics" + (f" ({scaling} scaling, relative to {run_label(ref_metrics['run_dir'])}) ===\n" if multi_run else " ===\n")
    parts.append(header_title)
    if multi_run:
        parts.append(f"  {'run':<{label_width}}  {'ranks':>5}  {'LB':>7}  {'CommE':>7}  {'PE':>7}  {'CompE':>7}  {'GE':>7}\n")
    else:
        parts.append(f"  {'run':<{label_width}}  {'ranks':>5}  {'LB':>7}  {'CommE':>7}  {'PE':>7}\n")
    for label, m in zip(row_labels, all_metrics):
        row = (
            f"  {label:<{label_width}}  "
            f"{m['num_ranks']:>5}  {fmt(m['load_balance']):>7}  {fmt(m['communication_efficiency']):>7}  "
            f"{fmt(m['parallel_efficiency']):>7}"
        )
        if multi_run:
            if m is ref_metrics:
                comp_e, ge = 1.0, ref_metrics["parallel_efficiency"]
            else:
                scaling_metrics = compute_scaling_metrics(ref_metrics, m, scaling)
                comp_e, ge = scaling_metrics["computation_efficiency"], scaling_metrics["global_efficiency"]
            row += f"  {fmt(comp_e):>7}  {fmt(ge):>7}"
        parts.append(row + "\n")

    parts.append("\n")
    parts.append("Metric explanation:\n")
    parts.append("  - LB    = avg / max useful compute time across ranks\n")
    parts.append(
        "  - CommE = max useful compute time / max total elapsed time across ranks "
        "(direct formula, not Dimemas's Serialisation x Transfer split -- see docs/pop_metrics_reference.md)\n"
    )
    parts.append("  - PE    = LB x CommE\n")
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
        "  - Communication time is classified by function-name prefix "
        f"({', '.join(MPI_PREFIXES)}) -- MPICH/Cray-MPICH only; other MPI\n"
        "    implementations' internal helpers may be misclassified as compute.\n"
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
