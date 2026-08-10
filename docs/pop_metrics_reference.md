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

## Future data sources

- **Tool 5** (`profile_hotspot_kernels.sh`, `rocprof-compute`) exposes GPU hardware counters
  (occupancy, VALU/SALU active cycles, cache/bandwidth) per kernel. This could feed a GPU-side
  analog of Instruction/IPC Scaling later, but it's a distinct data source from the classic
  CPU/PAPI-based version above and isn't wired into any of tools 1-4's output today.
