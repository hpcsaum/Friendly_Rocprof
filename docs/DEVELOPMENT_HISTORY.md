# Development History

| Date | Change |
|---|---|
| 2026-07-30 | Project scaffolding and `CLAUDE.md` project rules established |
| 2026-07-30 | First tool: rocprof-sys CPU hotspots launcher + extractor |
| 2026-07-30 | rocprof-sys hotspots: %-of-total column, selectable output, header metadata |
| 2026-07-31 | Second tool: rocprofv3 GPU kernel hotspots launcher + extractor |

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

## 2026-07-30 — rocprof-sys hotspots: %-of-total column, selectable output, header metadata

Extended the CPU hotspots tool with three follow-up requests:

1. **`%total` column**, computed purely from timemory data: each aggregated entry's share of a `total_runtime` denominator, itself the sum, across all scanned files, of that file's own single largest `SUM` value. A file's largest `SUM` is — barring unusual instrumentation — its outermost/root scope, since inclusive time only grows going up the call stack; this holds whether the file is a hierarchical or an already-flattened profile, without needing to guess the entry-point function's name (which the target codebase's language/build system might vary, e.g. Fortran's `MAIN__` vs C's `main`).
2. **Selectable output**: `-n/--top N` (unchanged default of 20), plus new `--threshold PCT` (only entries at/above PCT% of total runtime — falls back to showing everything, with a note, if total runtime couldn't be computed) and `--all` (no truncation), as a mutually-exclusive argparse group. `scripts/rocprof_sys_profile.sh` forwards whichever of the three the user passed through to the extractor unchanged.
3. **Best-effort header metadata** — executable name, run date/time, total runtime, and MPI rank count — read from `metadata.json` in the rocprof-sys output directory (never from the timemory files themselves, per the request). Since `metadata.json`'s schema isn't documented anywhere found during research, field lookup is a best-effort, case-insensitive search over several plausible key names (one level of nested dicts deep), with two concrete fallbacks grounded in documented rocprof-sys behavior rather than guesswork: run date/time falls back to rocprof-sys's own default output-subdirectory naming pattern (`%F_%H.%M`, e.g. `2025-01-21_07.40`) if present in the path, and rank count falls back to the number of distinct PIDs among the scanned `<component>-<pid>.txt` filenames. Any field that still can't be determined is left blank — never an error, per the request.

Extended `postprocess/tests/fixtures/mpi_2rank/` with a `metadata.json` fixture (exercising both a direct top-level key and one nested under a `settings` object) and left `single_rank/` without one, to test the all-fields-blank-plus-PID-fallback path. Added 15 new tests (33 total) covering the total-runtime computation, all three selection modes (including the threshold/unknown-total-runtime fallback), and metadata guessing with and without `metadata.json` present.

## 2026-07-31 — Second tool: rocprofv3 GPU kernel hotspots

Implemented the second tool per the approved plan in [docs/plans/02-rocprofv3-gpu-hotspots.md](plans/02-rocprofv3-gpu-hotspots.md): the GPU-kernel-only counterpart to the first tool, completing the roadmap item flagged when tool 1 was built (`rocprof-sys` never has real device execution time; `rocprofv3` does).

Research into `rocprofv3` (rocprofiler-sdk), cross-checked against the ROCm 7.0.2-tagged docs and the tool's own source on GitHub, found it considerably more straightforward to build on than `rocprof-sys`:
- `rocprofv3 --kernel-trace --stats --output-format csv -- <app>` needs no instrumentation/rebuild step, and writes a `kernel_stats.csv` whose schema was confirmed directly from the tool's source (`generateCSV.cpp`, `statistics.cpp`): `Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev`. rocprofv3 also already aggregates by kernel name internally (confirmed in `generateStats.cpp`) — no per-file dedup needed on our end, unlike `rocprof-sys`'s hierarchical text tables.
- Its own default output format is `rocpd` (SQLite), not CSV — `--output-format csv` must always be passed explicitly, so the launcher hardcodes it rather than exposing it as an option.
- Default output path is `<output_dir>/<hostname>/<pid>_kernel_stats.csv` (PID-based naming avoids MPI-rank collisions the same way `rocprof-sys` does) — the extractor globs recursively (`<output_dir>/**/*_kernel_stats.csv`) to find these one level down.
- No `metadata.json`-equivalent exists for ROCm 7.0.2 — the one candidate (`--output-config` → `<pid>_config.json`) is a post-7.0.2 addition, confirmed absent from the 7.0.2-tagged docs. The header's executable/run-datetime/total-runtime fields therefore stay blank on 7.0.2 in practice; only the MPI-rank-count fallback (distinct PIDs among scanned filenames) is reliable regardless of version.

Two pieces, mirroring tool 1's shape for a consistent tool family:
- `postprocess/rocprofv3_hotspots.py` — standalone Python 3 (stdlib only) extractor. Recursively finds `*_kernel_stats.csv`, aggregates by kernel name across however many files it finds, recomputes a global `%total` (sum of that kernel's `TotalDurationNs` across files, divided by the sum of every row's `TotalDurationNs` across every file — no root-detection ambiguity here, since `kernel_stats.csv` has no call-stack hierarchy at all) and a recomputed per-call average, and writes a single-table `hotspots.txt` (no CPU/GPU split needed — this tool is GPU-kernel-only by design). Same `-n/--top | --threshold | --all` selection modes as tool 1, duplicated rather than shared, since each extractor is meant to stand alone.
- `scripts/rocprofv3_profile.sh` — bash launcher wrapping `rocprofv3` with `--kernel-trace --stats --summary --truncate-kernels --output-format csv` and auto-invoking the extractor afterward, using the same MPI pattern and rank-0-only summary gating as tool 1's launcher.

Also updated tool 1's footer note (`postprocess/rocprof_sys_hotspots.py`, plus its checked-in sample `hotspots.txt` fixtures and the corresponding test assertion) to point at `scripts/rocprofv3_profile.sh` now that it exists, instead of a raw `rocprofv3` command line.

Verified locally (no ROCm/GPU available, same constraint as tool 1): `python3 -m unittest` against new hand-crafted fixtures under `postprocess/tests/fixtures/rocprofv3_*` (22 new tests; 55 total across both tools), including one row with scientific-notation values to exercise float parsing, and a `<pid>_config.json` fixture (with one directly-keyed and one nested-under-an-object field) to test the header-guessing path. `bash -n` on the launcher (shellcheck still not installed here). Manual verification used a stubbed `rocprofv3` on `PATH` (same technique as tool 1) to confirm the missing-dependency error, dry-run command construction (`--output-format csv` always present), a full run producing a real `hotspots.txt` from copied fixture data, and rank-0-only summary gating under a simulated `OMPI_COMM_WORLD_RANK=1`. Real end-to-end validation against actual `rocprofv3` output is left to the user on the HPC system.
