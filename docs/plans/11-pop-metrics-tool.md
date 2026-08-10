# POP metrics computation tool: `postprocess/extract_pop_metrics.py`

> **Implementation note (divergence):** the "Cross-run comparison" section below states
> `GE = scaled_metrics["PE"] * CompE`, "for the reference run itself, CompE = GE = 1 by
> definition". That last clause was a mistake caught during implementation: only
> `CompE_ref = 1` is trivially true (self-ratio); `GE_ref = PE_ref * 1 = PE_ref`, i.e. the
> reference run's Global Efficiency correctly equals its own (real, not-necessarily-1)
> Parallel Efficiency, not a hardcoded 1. Implemented that way in
> `postprocess/extract_pop_metrics.py`'s `write_report()`. See `docs/DEVELOPMENT_HISTORY.md`
> for the recorded divergence.

## Context

`docs/pop_metrics_reference.md` (written in the prior step) established which POP metrics are
computable from Friendly_Rocprof's existing output and what raw data each needs. The user now
wants the actual tool: given one rocprof-sys-family output directory, compute the single-run
metrics (Load Balance, Communication Efficiency, Parallel Efficiency); given additional
directories from the same scaling study, also compute Computation Efficiency and Global
Efficiency relative to the first directory as the reference (which trivially gets
CompE = GE = 1, since it's compared against itself). Both **strong scaling** (fixed global
problem size, more ranks) and **weak scaling** (fixed problem size per rank, more ranks) must be
supported, since they need different normalizations for Computation Efficiency.

Exploration this session (two Explore agents + direct source reads) established the concrete
building blocks to reuse and confirmed the exact semantics needed:

- `postprocess/extract_CPU_hotspots.py`'s `aggregate_per_rank(output_dir, unfiltered=False)`
  (source: `postprocess/extract_CPU_hotspots.py:360`) returns, per rank, a `{label: self_sum}`
  dict by default (exclusive/self time) — confirmed by reading the function directly. Self time
  is exactly what's needed to sum "time in MPI calls" or "time in GPU-sync-wait calls" per rank
  **without any double-counting from nested calls**, since self-time by construction partitions
  a rank's total time additively across every node in its call tree, regardless of nesting depth
  or naming. Calling it with `unfiltered=True` instead returns `{label: sum}` (inclusive) per
  rank — `max()` of those values is each rank's root/total wall-clock time (same "largest row is
  the root" heuristic `scan_ranks` already uses internally).
- `postprocess/extract_GPU_hotspots.py`'s `aggregate_per_rank` mirrors this shape for GPU kernel
  totals per rank (one rocprofv3 output file = one rank, kernels are leaves so there's no
  inclusive/exclusive distinction).
- `postprocess/extract_hotspots.py` already defines `SYNC_WAIT_LABELS = {"hipStreamSynchronize",
  "hipDeviceSynchronize"}` (line 26) and the exact "CPU total − GPU-wait overhead + GPU kernel
  total" arithmetic in `build_combined_view()` (lines 51-122) — reused here, but applied
  **per rank** instead of pooled across the whole run, since Load Balance and Communication
  Efficiency need per-rank granularity.
- There is **no existing MPI-call classification** anywhere in the codebase (`MPI_*`/`PMPI_*`
  labels are currently just ordinary CPU rows) — this tool introduces the first one, as a fixed
  module-level prefix set (matching the existing `GPU_API_PREFIXES`-style constant pattern in
  `extract_CPU_hotspots.py`), confirmed via user decision to be MPICH/Cray-MPICH-specific for
  now (documented as a known limitation, not CLI-configurable yet).
- No shared `common.py`/`utils.py` exists across `postprocess/` — every extractor is
  standalone-by-design and cross-tool reuse happens only via sibling `import extract_X_hotspots
  as x_tool` (as `extract_hotspots.py` already does). This tool follows the same pattern:
  `import extract_CPU_hotspots as cpu_tool`, `import extract_GPU_hotspots as gpu_tool`.
- Tests use plain `unittest` (`python3 -m unittest`, no pytest/CI config), with hand-crafted
  fixture directories under `postprocess/tests/fixtures/` (e.g. `mpi_2rank/` — 2-rank, but its
  `wall_clock-*.txt` rows are `main`/`_compute_stencil`/`_hipMemcpy` only, **no MPI-prefixed
  rows** — new fixtures with real `MPI_*`/`PMPI_*` rows are needed for this tool's tests).
  `test_extract_CPU_hotspots.py`'s style: `importlib.util` to load the module by file path,
  fixtures loaded from disk for I/O-touching tests, inline literal dicts for pure-function unit
  tests, comments justifying every hand-picked fixture value.
- Real example output confirms the directory layout to auto-detect: `profile_hotspots-output-*/`
  (tool 3) and `instrument_hotspots-scan-*/` (tool 4's aggregated-table output, **not** its
  `instrument_hotspots-trace-output-*/` raw-Perfetto-trace sibling, which has no `wall_clock*`
  tables at all) both already contain `rocprof-sys/` + `rocprofv3/` subdirectories directly.

User decisions from this planning round:
1. **Directory input**: one directory per run (reference first, then any number of scaled
   runs), each auto-inspected for `<dir>/rocprof-sys/` + `<dir>/rocprofv3/` subdirs, falling
   back to CPU-only if no `rocprofv3/` subdir is found — not `extract_hotspots.py`'s explicit
   two-positional-arg-per-run style.
2. **Weak-scaling Computation Efficiency** compares AVERAGE per-rank useful compute time
   (reference vs. scaled), not the total — the total is expected to grow with rank count under
   weak scaling even at perfect efficiency, so only the average is a meaningful ratio. Strong
   scaling keeps the POP page's literal formula: ratio of TOTAL (summed across ranks) useful
   compute time.
3. **MPI-prefix classification** is a fixed, hardcoded set for now (see below), documented in
   the tool's help text and `docs/pop_metrics_reference.md` as MPICH/Cray-MPICH-specific,
   expandable later rather than CLI-configurable today.

## Approach

### New file: `postprocess/extract_pop_metrics.py`

Follows the existing extractor pattern: `argparse` with `HELP_BLURB` +
`RawDescriptionHelpFormatter` (plain-language explanation first, ending with a link to the
rocprof-sys docs — same as the other four tools), sibling imports of `cpu_tool`/`gpu_tool`.

**CLI**:
```
extract_pop_metrics.py REF_DIR [SCALED_DIR ...] [--scaling {strong,weak}] [-o OUTPUT]
```
- `REF_DIR` (required): the reference run.
- `SCALED_DIR` (0 or more): additional runs in the same scaling study, compared against `REF_DIR`.
- `--scaling {strong,weak}`: required (argparse error with a clear message) only when at least
  one `SCALED_DIR` is given — meaningless for a single directory, where Computation/Global
  Efficiency are trivially 1 by definition (self-comparison).
- `-o`/`--output`: report path, same default-path convention as the other extractors.

**Module constants** (mirroring `GPU_API_PREFIXES` / `SYNC_WAIT_LABELS` style):
```python
MPI_PREFIXES = ("MPI_", "PMPI_", "MPIR_", "MPID_")  # MPICH / Cray-MPICH internals; documented
                                                      # limitation, not yet configurable
```

**Directory resolution** — `resolve_run_dirs(run_dir)` → `(cpu_dir, gpu_dir_or_None)`:
- If `<run_dir>/rocprof-sys/` and `<run_dir>/rocprofv3/` both exist → use both.
- Elif `<run_dir>/rocprof-sys/` exists alone → CPU-only.
- Elif `wall_clock*.txt` files exist directly under `run_dir` (tool-1-alone layout, no subdir
  nesting) → treat `run_dir` itself as the CPU dir.
- Elif only `<run_dir>/rocprofv3/` exists (GPU-only, no CPU timing at all) → fail with a clear
  error: POP metrics need CPU-side timing (point at a `profile_hotspots.sh` /
  `profile_CPU_hotspots.sh` / `instrument_hotspots.sh` scan output instead).

**Per-run metric computation** — `compute_run_metrics(run_dir)`:
1. `cpu_per_rank = cpu_tool.aggregate_per_rank(cpu_dir)` (self_sum, default) and
   `cpu_per_rank_incl = cpu_tool.aggregate_per_rank(cpu_dir, unfiltered=True)` (inclusive sum).
2. Per rank: `total_time = max(cpu_per_rank_incl[i].values())`.
3. Per rank: `comm_time = sum(v for label, v in cpu_per_rank[i].items() if label.startswith(MPI_PREFIXES))`.
4. If `gpu_dir` present: `gpu_api_overhead = sum(v for label, v in cpu_per_rank[i].items() if label in SYNC_WAIT_LABELS)`
   (imported from `extract_hotspots`), `gpu_total = sum(gpu_tool.aggregate_per_rank(gpu_dir)[i].values())`
   converted to seconds; `useful_compute = max(0, total_time - comm_time - gpu_api_overhead) + gpu_total`.
   Else: `useful_compute = max(0, total_time - comm_time)`.
5. Across ranks: `LB = mean(useful_compute) / max(useful_compute)`;
   `CommE = max(useful_compute) / max(total_time)`; `PE = LB * CommE`.
6. Return a dict: `{per_rank: [...], LB, CommE, PE, total_useful_compute (sum), avg_useful_compute (mean), num_ranks}`.

**Cross-run comparison** — `compute_scaling_metrics(ref_metrics, scaled_metrics, mode)`:
- `strong`: `CompE = ref_metrics["total_useful_compute"] / scaled_metrics["total_useful_compute"]`.
- `weak`: `CompE = ref_metrics["avg_useful_compute"] / scaled_metrics["avg_useful_compute"]`.
- `GE = scaled_metrics["PE"] * CompE` (for the reference run itself, `CompE = GE = 1` by
  definition — no division needed, just special-cased in the report writer).

**Report** (`write_report`, same text-report convention as the other tools): per-run table (LB,
CommE, PE, rank count) for every directory given, plus — only when 2+ directories are given — a
comparison table (CompE, GE per scaled run vs. the reference), and an explicit note of which
metrics are NOT included (Serialisation/Transfer Efficiency — no Dimemas; Instruction/IPC
Scaling — no PAPI counters in the input), pointing at `docs/pop_metrics_reference.md` for why.

No `scripts/` launcher is added — this tool only post-processes output directories that already
exist from earlier `profile_*`/`instrument_hotspots.sh` runs, same as `extract_hotspots.py`'s
own "works standalone against any existing output directory" mode.

### Tests: `postprocess/tests/test_extract_pop_metrics.py`

- New fixtures under `postprocess/tests/fixtures/`: at minimum `pop_mpi_2rank/` (reference,
  2 ranks, `wall_clock-N.txt` with real `MPI_Isend`/`PMPI_Waitall`/`MPIR_Typerep_icopy`-style
  rows at known self_sum values so LB/CommE/PE can be hand-computed and asserted exactly) and
  `pop_mpi_4rank/` (a "scaled" run, 4 ranks) to exercise the cross-run comparison in both
  `strong` and `weak` modes. Optionally one fixture with a paired `rocprofv3/` subdir to test
  the combined CPU+GPU pool path per rank.
- Follow `test_extract_CPU_hotspots.py`'s established style: `importlib.util` module loading,
  disk fixtures for I/O tests, inline literals for pure-function unit tests
  (`compute_run_metrics`'s inner math, `resolve_run_dirs`'s branching), comments justifying each
  hand-picked fixture number, cross-checks via `statistics.mean`/`pstdev` rather than hand-typed
  expected values where reasonable.
- Explicitly test: the self-time-based MPI/GPU-sync subtraction produces no double-counting
  regardless of call-tree nesting depth; `--scaling` required-only-when-multi-dir validation;
  the reference run's CompE/GE == 1 special case; GPU-dir-only input's clear error.

### Docs

Add a short "Computing POP metrics" section to `README.md` (same style/place as the other four
tools' sections) and update `docs/pop_metrics_reference.md`'s existing MPI-prefix caveat to
point at the new `MPI_PREFIXES` constant location, so the two docs stay consistent.

## Verification

- `python3 -m unittest` from the repo root (or `postprocess/tests/`) — all new tests pass
  alongside the existing suite (confirms nothing in the shared `cpu_tool`/`gpu_tool` modules was
  broken by how this tool calls them).
- `bash -n` doesn't apply (pure Python); run `python3 -m py_compile
  postprocess/extract_pop_metrics.py` as a basic syntax sanity check.
- Manually run `python3 postprocess/extract_pop_metrics.py
  ~/Documents/Porting_Adventure/Heat_Convection_Solver/profile_hotspots-output-2026-08-10_12.36.26
  ~/Documents/Porting_Adventure/Heat_Convection_Solver/profile_hotspots-output-2026-08-10_12.47.32
  --scaling strong` against the two real example directories gathered earlier this session, and
  sanity-check the printed LB/CommE/PE/CompE/GE values are all in `[0, ~1.2]` (values just above
  1 are plausible for CompE under super-linear effects, but LB/CommE/PE must stay in `[0, 1]`) —
  this can't be validated against ground truth on this CPU-only dev machine, but it confirms the
  tool runs end-to-end against real (not just fixture) data without crashing or producing
  nonsensical output.
