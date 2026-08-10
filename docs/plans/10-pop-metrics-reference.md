# POP metrics availability reference for rocprof-sys outputs

## Context

The long-term goal is a post-processing tool (in `Friendly_Rocprof`) that computes POP
(Performance Optimization and Productivity Centre of Excellence, https://pop-coe.eu/node/69)
parallel-efficiency metrics from `rocprof-sys` output. Before writing any computation code, the
user wants a grounded reference: for each POP metric, what raw data it needs, and whether that
data actually exists in the outputs Friendly_Rocprof's tools currently produce — both the
MPI-aware instrumented trace (tool 4, `instrument_hotspots.sh`) and the cheaper sampling-based
tools (tools 1-3, `profile_CPU_hotspots.sh` / `profile_GPU_hotspots.sh` / `profile_hotspots.sh`).
This avoids designing a metric computation that silently can't be fed real data.

Research already done this session:
- Fetched the POP metric hierarchy and formulas from pop-coe.eu/node/69 (Global Efficiency →
  Parallel Efficiency × Computation Efficiency; Parallel Efficiency → Load Balance ×
  Communication Efficiency; Communication Efficiency → Serialisation Efficiency × Transfer
  Efficiency, the latter two requiring Dimemas ideal-network simulation).
- An Explore agent inspected real example output trees under
  `~/Documents/Porting_Adventure/Heat_Convection_Solver/`:
  `instrument_hotspots-scan-2026-08-10_12.55.53/`, `instrument_hotspots-trace-output-2026-08-10_12.55.53/`
  (tool 4), and `profile_hotspots-output-2026-08-10_12.36.26/` / `profile_hotspots-output-2026-08-10_12.47.32/`
  (tool 3, sampling). Confirmed field-by-field what's in `metadata-N.json`, `functions-N.json`,
  `wall_clock-N.{json,txt}`, `sampling_wall_clock-N.{json,txt}`, `sampling_percent-N.{json,txt}`,
  `trip_count-N.{json,txt}`, `hotspots.txt`, and the `rocprofv3` CSVs (`kernel_trace`,
  `kernel_stats`, `domain_stats`, `agent_info`).
- Read `Friendly_Rocprof/README.md` and `CLAUDE.md` to confirm the tool 1-5 naming convention
  (tool 5 = `profile_hotspot_kernels.sh`, rocprof-compute GPU hardware-counter deep dive — not
  mentioned by the user but relevant as a future data source) and the project's plan-doc
  convention (`docs/plans/NN-slug.md`, next number is 10).

Key findings driving the plan:
- Per-rank MPI-call timing (exact) is present in tool 4's `wall_clock-N.json/txt` via
  `ROCPROFSYS_MPI_GOTCHA_ENABLED`; the same MPI call names show up in tools 1-3's
  `sampling_wall_clock`/`sampling_percent`, but only as statistically-sampled estimates —
  short/async MPI calls (`MPI_Isend`/`Irecv`) can be under- or mis-attributed.
- No PAPI/hardware-counter output exists in any current run (config flags like
  `ROCPROFSYS_PAPI_ARRAY_ENABLED` are present but `ROCPROFSYS_PAPI_EVENTS` is unset and no
  `papi_*` files exist) — so Instruction Scaling and IPC Scaling are **not computable today**
  from any of these outputs, only from a future run with PAPI counters explicitly configured.
- Dimemas-style ideal-network simulation doesn't exist here, so Serialisation Efficiency and
  Transfer Efficiency (the split of Communication Efficiency) are not computable — per the
  user's decision, these are simply documented as unavailable rather than approximated.
- Communication Efficiency *itself* (not its Dimemas sub-split) has a direct, non-simulated
  formula — `max(useful compute time) / max(total elapsed time)` across ranks — computable from
  real trace data alone. This is a more useful fact for the doc than treating all of
  Communication Efficiency as Dimemas-gated.
- Computation Efficiency, Instruction Scaling, and IPC Scaling are inherently scaling-study
  metrics: they compare a reference run against a scaled run, so they need **two or more**
  output directories as input, not just one.
- Friendly_Rocprof already computes per-rank/per-function avg/min/max/stddev across ranks (the
  CPU load-imbalance table in `extract_CPU_hotspots.py`, reused by `profile_hotspots.sh`'s
  combined report) and already does the "useful CPU time vs GPU-wait time" pool arithmetic
  (`profile_hotspots.sh`'s combined-pool calculation) — both are the exact building blocks
  Load Balance and Communication Efficiency need, and a future implementation should reuse them
  rather than re-deriving per-rank timing from scratch.

## Approach

Produce one reference document, not code — this stage is analysis/documentation only.

1. Write `Friendly_Rocprof/docs/pop_metrics_reference.md` containing:
   - A short intro linking to the POP page and naming the hierarchy (Global Efficiency →
     Parallel Efficiency × Computation Efficiency; Parallel Efficiency → Load Balance ×
     Communication Efficiency; Communication Efficiency → Serialisation Efficiency × Transfer
     Efficiency).
   - A table with one row per metric (Global Efficiency, Parallel Efficiency, Load Balance,
     Communication Efficiency, Serialisation Efficiency, Transfer Efficiency, Computation
     Efficiency, Instruction Scaling, IPC Scaling) and columns: formula, raw data required,
     available from tool 4 (instrumented+MPI trace)?, available from tools 1-3 (sampling)?,
     precision notes/caveats.
   - A callout that Computation Efficiency / Instruction Scaling / IPC Scaling need a
     multi-run scaling study (2+ output directories at different process/thread counts), not a
     single trace.
   - A callout listing what's confirmed absent from current outputs and why: PAPI/HW counters
     (flags set, no data), Dimemas simulation (not part of this toolchain), and two data-quality
     gaps observed directly in the example traces (blank `executable`/`total runtime` header
     fields in `hotspots.txt` on ROCm 7.0.2 — already a known README limitation; one missing
     per-rank `perfetto-trace-N.proto` file in the tool 4 example) as reminders that a real
     computation tool must handle missing/partial per-rank data rather than assume completeness.
   - A short "existing building blocks to reuse" note pointing at `extract_CPU_hotspots.py`'s
     load-imbalance logic and `profile_hotspots.sh`'s combined CPU/GPU pool arithmetic as the
     basis for a future `extract_pop_metrics.py`.
   - A short "future data sources" note: tool 5 (`profile_hotspot_kernels.sh`, rocprof-compute)
     exposes GPU hardware counters per kernel, which could feed a GPU-side analog of
     Instruction/IPC Scaling later, distinct from the classic CPU/PAPI-based version.
2. Per this repo's `CLAUDE.md` plan-documentation convention, once this plan is approved, also
   save this plan's verbatim text to `Friendly_Rocprof/docs/plans/10-pop-metrics-reference.md`
   (continuing the existing `01`-`09` sequence).
3. Present the same metric/availability table directly in the chat response (per user request
   for both a repo doc and a chat answer) — this is the final message of the turn, not a
   separate file.

No code changes, no new scripts, and no scaling-study run is performed in this step — this is
scoped purely to the reference document that a future implementation plan will build on.

## Verification

- Spot-check every "available" claim in the table against the specific file/field found during
  exploration (e.g. Load Balance's "available, tool 4" cites `wall_clock-N.json`'s per-rank
  inclusive/exclusive values; Instruction Scaling's "not available" cites the absence of any
  `papi_*` output file and the empty `ROCPROFSYS_PAPI_EVENTS`).
- Re-read `docs/pop_metrics_reference.md` after writing it and confirm every row's "raw data
  required" column matches a real field name from the explored files, not a generic restatement
  of the POP page.
- No runtime/test execution applies (documentation-only change).
