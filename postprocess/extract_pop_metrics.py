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
from stage1_run_dirs import resolve_run_dirs

# MPICH / Cray-MPICH function-name prefixes (plus a "most probable" Open MPI
# addition), shared with extract_CPU_hotspots.py -- see its own definition
# for the full rationale. Matched case-insensitively
# (label.lower().startswith(...)) against an all-lowercase tuple, not the
# previous case-sensitive uppercase-only comparison this file used to have,
# which only happened to work because every MPI symbol observed in real data
# so far is uppercase-prefixed -- a differently-cased symbol would have
# silently been undercounted as compute instead of communication.
MPI_PREFIXES = cpu_tool.MPI_PREFIXES

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
    needs, plus four GPU-specific extensions (GPU Offload Efficiency, GPU
    Utilization, GPU Load Balance, and the totals GPU Efficiency needs) when a
    paired rocprofv3 dir is present -- see docs/pop_metrics_reference.md's
    "GPU-specific extensions" section for why these aren't official POP
    metrics. Returns a dict -- see the bottom of this function for every key.

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
        comm_time = sum(v for label, v in self_totals.items() if label.lower().startswith(MPI_PREFIXES))

        if gpu_per_rank is not None:
            gpu_api_overhead = sync_wait_per_rank[i]
            cpu_pure = max(0.0, total_time - comm_time - gpu_api_overhead)
            gpu_busy_time = sum(gpu_per_rank[i].values())
            useful_compute = cpu_pure + gpu_busy_time
        else:
            cpu_pure = None
            gpu_busy_time = None
            useful_compute = max(0.0, total_time - comm_time)

        per_rank.append({
            "rank_key": rank_key,
            "total_time": total_time,
            "comm_time": comm_time,
            "useful_compute": useful_compute,
            "cpu_only_time": cpu_pure,
            "gpu_busy_time": gpu_busy_time,
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

    # All-or-nothing per run (see resolve_run_dirs()/the mismatched-rank-count fallback above):
    # either every rank has GPU data or none do, so checking one field is enough.
    gpu_busy_values = [r["gpu_busy_time"] for r in per_rank]
    has_gpu_data = all(v is not None for v in gpu_busy_values)
    max_gpu_busy = max(gpu_busy_values) if has_gpu_data else None

    gpu_offload_efficiency = (
        1.0 - max(r["cpu_only_time"] for r in per_rank) / max_total
        if has_gpu_data and max_total > 0 else None
    )
    # Distinct from gpu_offload_efficiency above: that one isolates the CPU-only
    # COMPUTE split (excluding comm/GPU-wait, on purpose -- comm is CommE's job).
    # This one is the GPU's raw share of wall-clock time, including any idling
    # caused by growing communication overhead -- gpu_busy_time doesn't nest
    # inside total_time additively (async kernels can overlap CPU work), so this
    # is a genuinely separate quantity, not derivable from gpu_offload_efficiency.
    gpu_utilization = (max_gpu_busy / max_total) if has_gpu_data and max_total > 0 else None
    gpu_load_balance = (
        statistics.mean(gpu_busy_values) / max_gpu_busy if has_gpu_data and max_gpu_busy > 0 else None
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
        "gpu_offload_efficiency": gpu_offload_efficiency,
        "gpu_utilization": gpu_utilization,
        "gpu_load_balance": gpu_load_balance,
        "total_gpu_busy_time": sum(gpu_busy_values) if has_gpu_data else None,
        "avg_gpu_busy_time": statistics.mean(gpu_busy_values) if has_gpu_data else None,
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


def compute_gpu_efficiency(ref_metrics, run_metrics, scaling):
    """GPU Efficiency: same strong/weak ratio shape as compute_scaling_metrics(),
    but over GPU-only busy time instead of the whole CPU+GPU useful_compute pool
    -- isolates whether it's specifically the GPU's own contribution that
    stopped scaling (e.g. per-rank problem size shrinking below what keeps the
    GPU saturated under strong scaling), as opposed to Computation Efficiency's
    whole-pool view, which could equally reflect a CPU-side or communication
    effect. None whenever either run lacks paired GPU data -- not an official
    POP metric, see docs/pop_metrics_reference.md.
    """
    key = "avg_gpu_busy_time" if scaling == "weak" else "total_gpu_busy_time"
    ref_value, run_value = ref_metrics[key], run_metrics[key]
    if ref_value is None or run_value is None or run_value <= 0:
        return None
    return ref_value / run_value


def run_label(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "n/a"


def write_report(run_dirs, dest_path, scaling=None):
    all_metrics = [compute_run_metrics(d) for d in run_dirs]
    ref_metrics = all_metrics[0]
    multi_run = len(all_metrics) > 1
    # GPUOff/GPULB share one gate: both come from the same has_gpu_data check per run
    # (see compute_run_metrics()), so they always appear/disappear together.
    show_gpu_cols = any(m["gpu_offload_efficiency"] is not None for m in all_metrics)
    show_gpu_eff = multi_run and any(
        compute_gpu_efficiency(ref_metrics, m, scaling) is not None for m in all_metrics if m is not ref_metrics
    )

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

    gpu_col_width = 8  # widest label below ("GPU-Util") is 8 chars

    header = f"  {'run':<{label_width}}  {'ranks':>5}  {'LB':>7}  {'CommE':>7}  {'PE':>7}"
    if show_gpu_cols:
        header += f"  {'GPU-Util':>{gpu_col_width}}  {'GPU-Off':>{gpu_col_width}}  {'GPU-LB':>{gpu_col_width}}"
    if multi_run:
        header += f"  {'CompE':>7}  {'GE':>7}"
    if show_gpu_eff:
        header += f"  {'GPU-Eff':>{gpu_col_width}}"
    parts.append(header + "\n")

    for label, m in zip(row_labels, all_metrics):
        row = (
            f"  {label:<{label_width}}  "
            f"{m['num_ranks']:>5}  {fmt(m['load_balance']):>7}  {fmt(m['communication_efficiency']):>7}  "
            f"{fmt(m['parallel_efficiency']):>7}"
        )
        if show_gpu_cols:
            row += (
                f"  {fmt(m['gpu_utilization']):>{gpu_col_width}}  {fmt(m['gpu_offload_efficiency']):>{gpu_col_width}}  "
                f"{fmt(m['gpu_load_balance']):>{gpu_col_width}}"
            )
        if multi_run:
            if m is ref_metrics:
                comp_e, ge = 1.0, ref_metrics["parallel_efficiency"]
            else:
                scaling_metrics = compute_scaling_metrics(ref_metrics, m, scaling)
                comp_e, ge = scaling_metrics["computation_efficiency"], scaling_metrics["global_efficiency"]
            row += f"  {fmt(comp_e):>7}  {fmt(ge):>7}"
        if show_gpu_eff:
            # show_gpu_eff being True guarantees ref_metrics has GPU data (see its
            # definition above), so the reference row's own ratio is trivially 1.0.
            gpu_eff = 1.0 if m is ref_metrics else compute_gpu_efficiency(ref_metrics, m, scaling)
            row += f"  {fmt(gpu_eff):>{gpu_col_width}}"
        parts.append(row + "\n")

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
        f"{', '.join(MPI_PREFIXES)}) --\n"
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
