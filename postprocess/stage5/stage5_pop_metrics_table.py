"""Stage 5 POP-inspired parallel-efficiency metrics: computation and the metrics table.

Scope: Load Balance, Communication Efficiency, and Parallel Efficiency from a single run; Computation
Efficiency, Global Efficiency, and the GPU-specific extensions when a scaling study (2+ runs) or a
paired rocprofv3 directory is available. See docs/pop_metrics_reference.md for the full metric
hierarchy. Also owns the "=== Metrics ===" table's column spec and assembly -- which sub-metric
columns appear depends on the data (GPU columns only when any run has GPU data, CompE/GE only for a
scaling study), unlike the hotspots tables' fixed column sets.

Functions: gpu_sync_wait_per_rank(), mpi_comm_time_per_rank(), gather_timing_summary_per_rank(),
compute_metrics_from_per_rank(), compute_run_metrics(), compute_scaling_metrics(),
compute_gpu_efficiency(), run_label(), fmt(), pop_metrics_columns(), metrics_legend(),
format_metrics_table().
"""

import os
import statistics

import stage4_rocprofsys_sample_flat
import stage4_rocprofv3
from stage1_run_dirs import resolve_run_dirs
from stage5_table_render import render_table

# The two HIP calls that mean "block the CPU until the GPU catches up" -- same
# definition and same self-time-only rationale as stage5_fused_hotspots_table.py's own
# SYNC_WAIT_LABELS (duplicated here, not imported, matching this codebase's
# existing "small constants are duplicated across standalone tools" convention).
SYNC_WAIT_LABELS = {"hipStreamSynchronize", "hipDeviceSynchronize"}


def gpu_sync_wait_per_rank(cpu_dir):
    """Per-rank self-time sum of hipStreamSynchronize/hipDeviceSynchronize --
    the two calls that mean "block the CPU until the GPU catches up" (see
    SYNC_WAIT_LABELS). Can't use aggregate_per_rank() for this: it
    only returns CPU-classified rows (row["gpu"] is False), and both labels
    here match the shared gpu_api tag, so scan_ranks() classifies them as GPU
    rows and aggregate_per_rank() silently drops them. Goes one level lower,
    straight to scan_ranks(), to see those rows at all.
    """
    ranks = stage4_rocprofsys_sample_flat.scan_ranks(cpu_dir)
    return [
        sum(row["self_sum"] for row in r["rows"] if row["gpu"] and row["label"] in SYNC_WAIT_LABELS)
        for r in ranks
    ]


def mpi_comm_time_per_rank(cpu_dir):
    """Per-rank self-time sum of every row tagged mpi_territory -- reads the tag
    scan_ranks() already computed on the real row tree, rather than re-deriving
    MPI classification from a bare label string a second time. Can't use
    aggregate_per_rank() for this, same reason gpu_sync_wait_per_rank() can't:
    it only returns non-GPU rows with the tag already discarded."""
    ranks = stage4_rocprofsys_sample_flat.scan_ranks(cpu_dir)
    return [sum(row["self_sum"] for row in r["rows"] if row["mpi"]) for r in ranks]


def gather_timing_summary_per_rank(run_dir):
    """Gathers the raw per-rank timing data compute_metrics_from_per_rank() needs -- the "get the
    data" half of what used to be one function here (compute_run_metrics()), kept in this file
    (rather than moved to a stage4 module) matching stage5_fused_hotspots_table.build_combined_view()'s
    own already-established shape: a stage5 backend that gathers from stage4 modules and does
    per-run correction arithmetic, not something stage4 itself owns.

    Per rank: total_time is that rank's root/whole-program inclusive wall time
    (max of its inclusive-time dict, same "largest value is the root" heuristic
    scan_ranks() already relies on internally). comm_time is the self-time sum
    of every row tagged mpi_territory (see mpi_comm_time_per_rank()) -- self-time,
    not inclusive, so nested MPI-internal helper calls (e.g. MPIR_Typerep_icopy
    under a PMPI_Waitall) are counted exactly once, not double-counted with
    their parent.

    When a paired rocprofv3 dir is present, useful_compute follows
    the combined-pool arithmetic stage5_fused_hotspots_table.py's own
    build_combined_view() uses (CPU total minus GPU sync-wait time, plus real
    GPU kernel time), applied PER RANK instead of pooled across the whole run --
    necessary for Load Balance/Communication Efficiency, which need per-rank
    granularity. This assumes rank i's rocprof-sys file corresponds to rank i's
    rocprofv3 file (matching sorted-filename order) -- not cross-checked, same
    "caller's responsibility" spirit as the fused table's own pairing. If the
    two directories don't report the same number of ranks, the GPU side is
    skipped entirely for this run (falls back to CPU-only) with a warning,
    rather than risk combining mismatched ranks.

    Returns (per_rank, cpu_dir, gpu_dir, rank_keys): per_rank is a list of
    {"rank_key", "total_time", "comm_time", "useful_compute", "cpu_only_time", "gpu_busy_time"}
    dicts, one per rank, in the "per-rank timing summary" shape compute_metrics_from_per_rank()
    consumes; gpu_dir comes back None when GPU data wasn't actually paired in (even if a gpu_dir
    argument existed, on a rank-count mismatch).
    """
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    incl_per_rank, rank_keys = stage4_rocprofsys_sample_flat.aggregate_per_rank(cpu_dir, unfiltered=True)
    comm_time_per_rank = mpi_comm_time_per_rank(cpu_dir)

    if not rank_keys:
        if gpu_dir is not None and stage4_rocprofv3.aggregate_per_rank(gpu_dir)[1]:
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
        gpu_totals, gpu_scanned = stage4_rocprofv3.aggregate_per_rank(gpu_dir)
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
        incl_totals = incl_per_rank[i]
        total_time = max(incl_totals.values()) if incl_totals else 0.0
        comm_time = comm_time_per_rank[i]

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

    return per_rank, cpu_dir, (gpu_dir if gpu_per_rank is not None else None), rank_keys


def compute_metrics_from_per_rank(per_rank):
    """Load Balance, Communication Efficiency, and Parallel Efficiency from a plain per-rank
    timing summary (see gather_timing_summary_per_rank()'s return shape) -- the "compute what the
    table needs" half of what used to be one function here. Source-agnostic once that list
    exists: no stage4 imports, no opinion on where total_time/comm_time/useful_compute/
    gpu_busy_time actually came from, so a new data source's own gathering step can feed this
    function directly. Also the per-run totals a scaling comparison needs
    (total/avg_useful_compute) and the four GPU-specific extensions (GPU Offload Efficiency, GPU
    Utilization, GPU Load Balance, and the totals GPU Efficiency needs) when any rank carries GPU
    data -- see docs/pop_metrics_reference.md's "GPU-specific extensions" section for why these
    aren't official POP metrics. Returns a dict -- see the bottom of this function for every key.
    """
    useful_values = [r["useful_compute"] for r in per_rank]
    total_values = [r["total_time"] for r in per_rank]
    max_useful = max(useful_values)
    max_total = max(total_values)

    load_balance = (statistics.mean(useful_values) / max_useful) if max_useful > 0 else None
    comm_efficiency = (max_useful / max_total) if max_total > 0 else None
    parallel_efficiency = (
        load_balance * comm_efficiency if load_balance is not None and comm_efficiency is not None else None
    )

    # All-or-nothing per run (see gather_timing_summary_per_rank()'s mismatched-rank-count
    # fallback): either every rank has GPU data or none do, so checking one field is enough.
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


def compute_run_metrics(run_dir):
    """One run's full metrics dict: gather_timing_summary_per_rank() then
    compute_metrics_from_per_rank(), merged with this run's own identifying metadata. Kept as one
    call for existing callers (e.g. extract_pop_metrics.py) that just want "give me a directory,
    get me the metrics" -- a new data source reuses compute_metrics_from_per_rank() directly
    instead, feeding it from its own gathering step rather than this one."""
    per_rank, cpu_dir, gpu_dir, rank_keys = gather_timing_summary_per_rank(run_dir)
    metrics = compute_metrics_from_per_rank(per_rank)
    return {
        "run_dir": run_dir,
        "cpu_dir": cpu_dir,
        "gpu_dir": gpu_dir,
        "num_ranks": len(rank_keys),
        "per_rank": per_rank,
        **metrics,
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


def pop_metrics_columns(all_metrics, ref_metrics, scaling, multi_run):
    """Decides which sub-metric columns appear (LB/CommE/PE always; GPU-Util/GPU-Off/GPU-LB when
    any run has GPU data; CompE/GE when multi_run; GPU-Eff when multi_run and any non-reference
    run has GPU efficiency data) and returns (columns, show_gpu_cols, show_gpu_eff) -- the two
    booleans are returned alongside the column spec since format_metrics_table()'s caller also
    needs them, for the explanation prose below the table. Each column's value callable closes
    over ref_metrics/scaling to compute per-row derived values (e.g. calling
    compute_scaling_metrics() for CompE/GE) -- this "which sub-metrics show" decision is
    data-dependent, unlike the hotspots tables' fixed column sets.
    """
    row_labels = [
        run_label(m["run_dir"]) + (" (reference)" if multi_run and m is ref_metrics else "")
        for m in all_metrics
    ]
    label_width = max(len("run"), *(len(label) for label in row_labels))
    gpu_col_width = 8  # widest label below ("GPU-Util") is 8 chars

    # GPUOff/GPULB share one gate: both come from the same has_gpu_data check per run
    # (see compute_run_metrics()), so they always appear/disappear together.
    show_gpu_cols = any(m["gpu_offload_efficiency"] is not None for m in all_metrics)
    show_gpu_eff = multi_run and any(
        compute_gpu_efficiency(ref_metrics, m, scaling) is not None for m in all_metrics if m is not ref_metrics
    )

    columns = [
        {"header": "run", "width": label_width, "align": "left", "value": lambda e, i: row_labels[i - 1]},
        {"header": "ranks", "width": 5, "value": lambda e, i: str(e["num_ranks"])},
        {"header": "LB", "width": 7, "value": lambda e, i: fmt(e["load_balance"])},
        {"header": "CommE", "width": 7, "value": lambda e, i: fmt(e["communication_efficiency"])},
        {"header": "PE", "width": 7, "value": lambda e, i: fmt(e["parallel_efficiency"])},
    ]
    if show_gpu_cols:
        columns += [
            {"header": "GPU-Util", "width": gpu_col_width, "value": lambda e, i: fmt(e["gpu_utilization"])},
            {"header": "GPU-Off", "width": gpu_col_width, "value": lambda e, i: fmt(e["gpu_offload_efficiency"])},
            {"header": "GPU-LB", "width": gpu_col_width, "value": lambda e, i: fmt(e["gpu_load_balance"])},
        ]
    if multi_run:
        def _comp_e(e, i):
            if e is ref_metrics:
                return fmt(1.0)
            return fmt(compute_scaling_metrics(ref_metrics, e, scaling)["computation_efficiency"])

        def _ge(e, i):
            if e is ref_metrics:
                return fmt(ref_metrics["parallel_efficiency"])
            return fmt(compute_scaling_metrics(ref_metrics, e, scaling)["global_efficiency"])

        columns += [
            {"header": "CompE", "width": 7, "value": _comp_e},
            {"header": "GE", "width": 7, "value": _ge},
        ]
    if show_gpu_eff:
        def _gpu_eff(e, i):
            # show_gpu_eff being True guarantees ref_metrics has GPU data (see its
            # definition above), so the reference row's own ratio is trivially 1.0.
            return fmt(1.0 if e is ref_metrics else compute_gpu_efficiency(ref_metrics, e, scaling))

        columns.append({"header": "GPU-Eff", "width": gpu_col_width, "value": _gpu_eff})

    return columns, show_gpu_cols, show_gpu_eff


def metrics_legend(show_gpu_cols, show_gpu_eff, multi_run, scaling):
    """Bulleted explanation of every sub-metric column format_metrics_table() might show,
    mirroring exactly the same show_gpu_cols/show_gpu_eff/multi_run/scaling booleans
    pop_metrics_columns() already used to decide which columns exist -- the column-inclusion
    decision and its own explanation can't drift apart, since both come from the same inputs."""
    legend = "Metric explanation:\n"
    legend += "  - LB    = avg / max useful compute time across ranks\n"
    legend += (
        "  - CommE = max useful compute time / max total elapsed time across ranks "
        "(direct formula, not Dimemas's Serialisation x Transfer split -- see docs/pop_metrics_reference.md)\n"
    )
    legend += "  - PE    = LB x CommE\n"
    if show_gpu_cols:
        legend += (
            "  - GPU-Util = max GPU busy time / max total elapsed time across ranks -- NOT an official POP "
            "metric; the GPU's raw share of wall-clock time, INCLUDING any idling caused by growing "
            "communication overhead -- unlike GPU-Off, this drops when CommE drops too, since a "
            "comm-starved GPU is genuinely less utilized, whatever the root cause\n"
        )
        legend += (
            "  - GPU-Off = 1 - (max non-offloaded CPU compute time / max total elapsed time) across ranks -- "
            "NOT an official POP metric; how much of the critical-path rank's time is still CPU-only "
            "compute (serial, not-yet-ported, or not-worth-porting code) -- deliberately excludes "
            "communication time, already covered by CommE\n"
        )
        legend += (
            "  - GPU-LB = avg / max GPU busy time across ranks -- NOT an official POP metric; load balance "
            "between GPUs specifically, separate from LB's whole CPU+GPU pool\n"
        )
    if multi_run:
        if scaling == "weak":
            legend += (
                "  - CompE = avg per-rank useful compute time (reference) / avg per-rank useful compute time (this run) "
                "-- weak scaling's total is expected to grow with rank count even at perfect efficiency, "
                "so only the average is meaningful\n"
            )
        else:
            legend += (
                "  - CompE = total useful compute time (reference) / total useful compute time (this run), summed "
                "across ranks -- strong scaling's ideal keeps this total constant as rank count grows\n"
            )
        legend += "  - GE    = PE x CompE\n"
    else:
        legend += (
            "  - CompE, GE need a scaling study (2+ directories, compared against the first as reference) "
            "-- pass additional directories to see them\n"
        )
    if show_gpu_eff:
        if scaling == "weak":
            legend += (
                "  - GPU-Eff = avg per-rank GPU busy time (reference) / avg per-rank GPU busy time (this run) -- "
                "NOT an official POP metric; isolates whether it's specifically the GPU's own contribution "
                "that stopped scaling, as opposed to CompE's whole-pool view\n"
            )
        else:
            legend += (
                "  - GPU-Eff = total GPU busy time (reference) / total GPU busy time (this run), summed across "
                "ranks -- NOT an official POP metric; a low value in strong scaling flags the per-rank "
                "problem size shrinking below what keeps the GPU saturated\n"
            )
    return legend


def format_metrics_table(all_metrics, scaling=None):
    """Returns (table_text, show_gpu_cols, show_gpu_eff) for the '=== Metrics ===' block -- a single
    titled section still gets the "=== Title ===" style without a number, since there's nothing to
    number it against. Decided once here since format_metrics_table() is structurally incapable of
    a 2nd section, so it never goes through render_report()'s own numbering logic at all. The two
    booleans are returned because metrics_legend() above also branches on them."""
    ref_metrics = all_metrics[0]
    multi_run = len(all_metrics) > 1
    columns, show_gpu_cols, show_gpu_eff = pop_metrics_columns(all_metrics, ref_metrics, scaling, multi_run)

    header_title = "=== Metrics" + (
        f" ({scaling} scaling, relative to {run_label(ref_metrics['run_dir'])}) ===\n" if multi_run else " ===\n"
    )
    table_text = header_title + render_table(columns, all_metrics)
    return table_text, show_gpu_cols, show_gpu_eff
