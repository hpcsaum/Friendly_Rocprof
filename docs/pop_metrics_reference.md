# POP metrics: what's computable from Friendly_Rocprof output

Reference for a future POP-metrics post-processing tool. Maps the standard POP
(Performance Optimization and Productivity Centre of Excellence,
[pop-coe.eu/node/69](https://pop-coe.eu/node/69)) parallel-efficiency metric hierarchy onto the
raw fields actually present in Friendly_Rocprof's existing tool output, so a computation tool
isn't designed around data that doesn't exist.

## Metric hierarchy

```
Global Efficiency (GE) = Parallel Efficiency x Computation Efficiency
Parallel Efficiency (PE) = Load Balance x Communication Efficiency
Communication Efficiency (CommE) = Serialisation Efficiency x Transfer Efficiency
```

## Tool naming used below

- **Tool 1-3** (sampling): `profile_CPU_hotspots.sh`, `profile_GPU_hotspots.sh`,
  `profile_hotspots.sh` — statistical call-stack sampling (`rocprof-sys`) + exact GPU kernel
  dispatch capture (`rocprofv3`). No binary rewrite.
- **Tool 4** (instrumented, MPI-aware): `instrument_hotspots.sh trace` — binary-rewrite
  instrumentation of hotspot functions, MPI calls captured automatically via
  `ROCPROFSYS_MPI_GOTCHA_ENABLED`, producing exact per-call timing plus a full Perfetto trace.
- **Tool 5** (not yet integrated with this analysis): `profile_hotspot_kernels.sh` —
  `rocprof-compute` GPU hardware-counter deep dive per kernel.

## Metric table

| Metric | Formula | Raw data required | Tool 4 (instrumented + MPI) | Tools 1-3 (sampling) | Notes |
|---|---|---|---|---|---|
| **Load Balance (LB)** | avg(useful compute time) / max(useful compute time), across ranks | Per-rank total wall time minus per-rank MPI/comm time | **Available, exact.** `wall_clock-N.json/.txt` gives per-rank inclusive/exclusive time per function, including every `MPI_*`/`PMPI_*` call via GOTCHA interception — subtract MPI inclusive time from root ("main") inclusive time per rank. | **Available, approximate.** `sampling_wall_clock-N`/`sampling_percent-N` give statistically sampled per-rank time-in-MPI vs time-in-compute. | Short/async calls (`MPI_Isend`/`Irecv`) can be under- or mis-sampled at low sampling rates, skewing the compute/comm split. Reuses the same per-rank/per-function aggregation already built for `extract_CPU_hotspots.py`'s load-imbalance table. |
| **Communication Efficiency (CommE)**, direct/aggregate form | max(useful compute time) / max(total elapsed time), across ranks | Same per-rank useful-time and total-elapsed-time data as LB | **Available, exact.** | **Available, approximate.** | This aggregate form does **not** need Dimemas — only the SerE/TE *split* below does. Worth computing even if the split is out of reach. |
| **Serialisation Efficiency (SerE)** | max computation time (ideal zero-latency network) / total runtime (ideal network) | Dimemas (or equivalent) replay of the trace on a simulated ideal network | **Not available.** | **Not available.** | No Dimemas-equivalent network simulator exists in this toolchain. Not approximated — documented as a hard gap. |
| **Transfer Efficiency (TE)** | total runtime (ideal network) / total runtime (real network) | Same Dimemas simulation output as SerE | **Not available.** | **Not available.** | Same gap as SerE — both require simulated-vs-real runtime, which needs Dimemas. |
| **Parallel Efficiency (PE)** | Load Balance x Communication Efficiency | LB and (aggregate) CommE above | **Available, exact**, as LB x CommE(aggregate). | **Available, approximate.** | Derived metric — inherits the precision of its two inputs. |
| **Computation Efficiency (CompE)** | total useful computation time (reference run) / total useful computation time (scaled run) | Per-rank useful compute time, summed per run, at 2+ process/thread counts | **Available, exact**, but needs 2+ tool-4 output directories (a scaling study), not one. | **Available, approximate**, same multi-run requirement. | Not computable from a single trace regardless of tool — always needs a reference config and a scaled config. |
| **Instruction Scaling** | total instructions (reference) / total instructions (scaled) | Per-rank hardware instruction counts (e.g. PAPI `PAPI_TOT_INS`), at 2+ configs | **Not available today.** | **Not available today.** | Confirmed: `ROCPROFSYS_PAPI_ARRAY_ENABLED`/`ROCPROFSYS_PAPI_VECTOR_1_ENABLED` flags are set in `metadata-N.json`, but `ROCPROFSYS_PAPI_EVENTS` is empty and no `papi_*` output file exists anywhere in the example trees. Requires re-running with PAPI events explicitly configured. |
| **IPC Scaling** | IPC(scaled) / IPC(reference), IPC = instructions / cycles | Per-rank instructions **and** cycles (PAPI), at 2+ configs | **Not available today.** | **Not available today.** | Same PAPI gap as Instruction Scaling — needs both instruction and cycle counters, neither present. |
| **Global Efficiency (GE)** | Parallel Efficiency x Computation Efficiency | PE and CompE above | **Available, exact**, only as part of a 2+ run scaling study. | **Available, approximate**, same requirement. | Inherits CompE's multi-run requirement — not computable from a single output directory. |

## Data gaps confirmed in the example traces

- **PAPI / hardware counters**: config flags are present (`ROCPROFSYS_PAPI_*_ENABLED=true`) but
  no events are configured and no counter output files exist. Instruction Scaling and IPC
  Scaling need a future run with `ROCPROFSYS_PAPI_EVENTS` explicitly set.
- **Dimemas-style ideal-network simulation**: not part of this toolchain. Serialisation
  Efficiency and Transfer Efficiency are out of reach unless such a simulator is added
  separately — not approximated here by design.
- **`hotspots.txt` header fields**: `executable:` and `total runtime:` were blank in the example
  runs — an already-known ROCm 7.0.2 limitation (the `--output-config` metadata file doesn't
  exist yet; see `README.md`). A metrics tool needs its own way to get total elapsed runtime
  per rank (e.g. the root/`main` entry's inclusive time in `wall_clock-N.json`, not the
  `hotspots.txt` header).
- **Missing per-rank trace file**: in the tool 4 example
  (`instrument_hotspots-trace-output-2026-08-10_12.55.53/2026-08-10_12.59/`), rank 2's
  `perfetto-trace-2.proto` is absent even though its `metadata-2.json`/`functions-2.json` exist.
  A real computation tool must tolerate partial per-rank data rather than assume every rank
  produced every file.

- **MPI-call classification**: `postprocess/extract_pop_metrics.py`'s `MPI_PREFIXES = ("MPI_",
  "PMPI_", "MPIR_", "MPID_")` constant is the first (and, for now, only) place in this codebase
  that classifies a function name as communication rather than compute. It's MPICH/Cray-MPICH
  specific by design decision — an Open MPI run's internal helpers (`ompi_`/`opal_`/`orte_`
  prefixes) aren't recognized and would be misclassified as application compute. Not
  CLI-configurable today; may be extended if a non-MPICH MPI implementation needs support.

## Implemented: `postprocess/extract_pop_metrics.py`

This reference doc's Load Balance / Communication Efficiency / Parallel Efficiency /
Computation Efficiency / Global Efficiency rows are now computed by
`postprocess/extract_pop_metrics.py` — see [README.md](../README.md#computing-pop-metrics---extract_pop_metricspy)
for usage. The two building blocks below, described as future work when this doc was first
written, are exactly what that tool builds on:

- `postprocess/extract_CPU_hotspots.py`'s per-function avg/min/max/stddev-across-ranks logic
  (already used for the CPU load-imbalance table) is the same aggregation Load Balance needs at
  the whole-program level — specifically its self-time `aggregate_per_rank()` output, which
  partitions each rank's total time additively across every call-tree node with no
  double-counting, regardless of nesting.
- `extract_hotspots.py`'s combined CPU+GPU pool arithmetic (`(CPU total - GPU API/overhead) +
  GPU kernel total`) is the right definition of "useful compute time" for a heterogeneous
  (CPU+GPU) code — `extract_pop_metrics.py` builds LB/CommE on top of that pool (applied per
  rank, not pooled across the whole run) whenever a paired `rocprofv3/` directory is found,
  instead of raw CPU wall-clock time alone.

## GPU-specific extensions (not part of the official POP catalog)

A separate question came up: POP also defines hybrid (MPI+OpenMP) metrics — see
[pop-coe.eu's hybrid metrics page](https://pop-coe.eu/further-information/learning-material/pop-standard-hybrid-metrics-for-parallel-performance-analysis)
— that decompose Parallel Efficiency into a process (MPI) level and a thread (OpenMP) level, the
OpenMP-level metrics derived as ratios of the hybrid view to the MPI-only view (e.g. `OpenMP
Parallel Efficiency = Hybrid Parallel Efficiency / MPI Parallel Efficiency`). Could the same
substitution ("time on GPU" for "time in an OpenMP parallel region") produce a GPU-level metric
set the same way?

**Mostly no, for the thread-balance-shaped metrics.** POP's OpenMP-level formulas (Thread Load
Balance, Serial Region Efficiency) rely on per-thread timing across a handful of *persistent,
comparable workers* — each OpenMP thread does a slice of the same region for that region's whole
duration, and `rocprof-sys` genuinely captures that per-thread. GPU kernels don't have that
shape: `rocprofv3` only gives per-dispatch aggregate duration, not per-GPU-thread timing (that's
a different instrument — `rocprof-compute`'s hardware occupancy/wavefront counters, tool 5).
Computing "GPU thread balance" the OpenMP way would measure the wrong thing.

**Four questions turned out to be well-posed anyway**, using data the tool already gathers. None
of these are official POP metrics — they're named and formatted to avoid confusion with POP's own
catalog (in particular, POP's real *Serialisation Efficiency* is the Dimemas-based metric a few
sections up, listed there as "not computed").

| Metric | Formula | Level | Rationale |
|---|---|---|---|
| **GPU Offload Efficiency** (`GPU-Off`) | `1 - (max non-offloaded CPU compute time / max total elapsed time)` across ranks | Per run | Amdahl's-law-style: how much of the critical-path rank's time is still CPU-only *compute* — genuinely serial code, code not yet ported, or code not worth porting. Deliberately excludes communication time (already `CommE`'s job). |
| **GPU Utilization** (`GPU-Util`) | `max(GPU busy time) / max(total elapsed time)` across ranks | Per run | The GPU's raw share of wall-clock time — including any idling caused by growing communication overhead. `gpu_busy_time` doesn't nest additively inside `total_time` (async kernels can overlap CPU work), so this is a genuinely different quantity from `GPU-Off`, not its complement. |
| **GPU Load Balance** (`GPU-LB`) | `avg(GPU busy time) / max(GPU busy time)` across ranks | Per run | The overall `LB` already mixes GPU kernel time into the combined CPU+GPU pool per rank, so imbalance *between GPUs specifically* (as opposed to between whole ranks) is otherwise invisible on its own. |
| **GPU Efficiency** (`GPU-Eff`) | Strong: `total GPU busy time (reference) / total GPU busy time (scaled)`, summed across ranks. Weak: same ratio using the *average* per-rank GPU busy time. | Scaling (2+ runs) | GPU kernel time that's compute- or memory-bound should scale roughly linearly with the work assigned to that GPU. In **strong scaling**, the classic failure mode is the per-rank problem size shrinking below what the GPU needs to stay saturated — kernel launch overhead stops being amortized, occupancy drops, the GPU becomes latency- rather than throughput-bound. Structurally identical to `Computation Efficiency` but restricted to just the GPU-kernel-time bucket instead of the whole CPU+GPU pool, isolating whether it's specifically the GPU's own contribution that stopped scaling (as opposed to a CPU-side or communication effect `CompE`'s whole-pool view can't tell apart). |

**Why both `GPU-Off` and `GPU-Util` exist, not just one:** the first implementation only had
`GPU-Off`, normalized against total elapsed time. Validated against a real 2-rank vs. 4-rank
strong-scaling run, `GPU-Off` went *up* (0.48 → 0.84) between the two — counterintuitive, since
per-rank GPU busy time stayed roughly flat while communication overhead nearly tripled. The cause:
normalizing the CPU-only-compute bucket against *total* time (which includes communication)
means growing communication dilutes the ratio and makes offload look artificially better, even
though nothing about offload changed — `CommE` dropping (0.81 → 0.35) was doing all the work.
`GPU-Util`, which measures the GPU's raw share of wall-clock time directly, moved the way intuition
expects (0.30 → 0.19: the GPU sat idle a larger fraction of the time as communication grew to
dominate). Keeping both is more informative than picking one: `GPU-Off` isolates the CPU-compute
offload question cleanly (comm-independent, by construction); `GPU-Util` answers "is the GPU
actually busy," which legitimately degrades when communication starves it, regardless of root
cause.

All four are computed in `postprocess/extract_pop_metrics.py` whenever the underlying data
supports them (a paired `rocprofv3/` directory for `GPU-Off`/`GPU-Util`/`GPU-LB`; a scaling study
where both the reference and compared run have paired GPU data for `GPU-Eff`) — no new
instrumentation or CLI flags needed.

## Future data sources

- **Tool 5** (`profile_hotspot_kernels.sh`, `rocprof-compute`) exposes GPU hardware counters
  (occupancy, VALU/SALU active cycles, cache/bandwidth) per kernel. This could feed a GPU-side
  analog of Instruction/IPC Scaling later, but it's a distinct data source from the classic
  CPU/PAPI-based version above and isn't wired into any of tools 1-4's output today.
