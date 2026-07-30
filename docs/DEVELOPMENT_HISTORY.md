# Development History

| Date | Change |
|---|---|
| 2026-07-30 | Project scaffolding and `CLAUDE.md` project rules established |
| 2026-07-30 | First tool: rocprof-sys CPU hotspots launcher + extractor |

## 2026-07-30 — Project scaffolding and rules

Initialized the repository with a `scripts/` directory (bash scripts wrapping `rocprofv3`/`rocprof-system`/`rocprof-compute`) and a `postprocess/` directory (tools for parsing/analyzing their output), plus a README and `.gitignore` for typical profiler output artifacts (`.csv`, `.json`, `.db`, `results/`).

Added `CLAUDE.md` defining the project scope and working constraints:
- Targets AMD ROCm's profiling stack (`rocprofv3`, `rocprof-system`, `rocprof-compute`), per https://rocm.docs.amd.com/en/latest/, with retro-compatibility to ROCm 7.0.2.
- Tools must work against arbitrary, unknown target codebases — the only assumed facts are: written in C/C++ or Fortran, depends on ROCm and an MPI library, and may use OpenMP (offload or CPU-parallel).
- The project is open source.
- The dev machine is CPU-only with no ROCm/GPU access and no path to the HPC system, so nothing here can be validated end-to-end locally — final validation against real hardware is done manually by the user. Local checks are limited to syntax/lint and logic tests against hand-crafted fixture data matching documented rocprof output schemas.
- Adopted the same plan-documentation workflow used in `Heat_Convection_Solver`: approved plans get written verbatim to `docs/plans/`, and `docs/DEVELOPMENT_HISTORY.md` (plus its regenerated `.docx`) is updated before every commit.

## 2026-07-30 — First tool: rocprof-sys CPU hotspots

Implemented the first tool per the approved plan in [docs/plans/01-rocprof-sys-hotspots.md](plans/01-rocprof-sys-hotspots.md): a quick "where's the time going, and what's still on the CPU" report for a first `rocprof-sys` profiling run, without needing the full Perfetto trace viewer.

Research into `rocprof-sys` (rocprofiler-systems, ex-Omnitrace) established that its text/JSON output only captures **host-side** timing — even rows that look like GPU work (`hipLaunchKernel`, etc.) are launch/API overhead, not real device execution time. True GPU kernel hotspots need a different tool (`rocprofv3 --stats --kernel-trace --summary`, already stable since ROCm 7.0.2) or Perfetto/rocpd trace parsing (rocpd itself isn't available before ROCm 7.1, so out of our 7.0.2 compatibility scope). Given that, this first tool deliberately scopes itself to CPU-side hotspots only — a GPU-only tool (built on `rocprofv3`) and a combined tool are planned as separate, later work.

Two pieces:
- `postprocess/rocprof_sys_hotspots.py` — standalone Python 3 (stdlib only) extractor. Parses rocprof-sys's pipe-delimited "timemory" text tables (e.g. `wall_clock-<pid>.txt`), aggregates by function name across however many files it finds (handles multiple MPI-rank files transparently), buckets entries into "CPU compute (offload candidates)" vs "GPU API/launch overhead" by name prefix, and writes a short `hotspots.txt`. Notable implementation detail: rocprof-sys's own docs describe the MPI row-prefix format as `|MM|NN>>>label`, which embeds an extra `|` inside what's conceptually one field — the parser handles this (and the similar case of hierarchy-indentation markers) generically by taking the last 11 pipe-delimited fields as the fixed numeric columns and treating everything before that as the label, rather than assuming a fixed field count.
- `scripts/rocprof_sys_profile.sh` — bash launcher wrapping `rocprof-sys-sample` (lightweight call-stack sampling, no binary instrumentation) with sane first-run defaults (flat profile, text+JSON output, Perfetto tracing off), and auto-invoking the extractor afterwards. Since `rocprof-sys-sample` wraps a single process, MPI usage is `mpirun -np N scripts/rocprof_sys_profile.sh -- ./app` (mpirun/srun goes *before* the script, not inside it) — the script detects `OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`SLURM_PROCID` so only rank 0 runs the extractor against the shared output directory.

Verified locally (no ROCm/GPU available on this dev machine, per `CLAUDE.md`'s environment constraints): `python3 -m unittest` against hand-crafted fixtures under `postprocess/tests/fixtures/` (18 tests covering label-cleaning, table parsing, cross-file aggregation, GPU/CPU bucketing, and end-to-end report generation), `bash -n` on the launcher (shellcheck not installed in this environment, so that check was skipped), and manual dry-run / stubbed-`rocprof-sys-sample` runs exercising flag parsing, env var construction, and the rank-0-only summary gating. Real end-to-end validation against actual `rocprof-sys` output is left to the user on the HPC system.
