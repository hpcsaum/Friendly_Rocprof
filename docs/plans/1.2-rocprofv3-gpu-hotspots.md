# Second tool: rocprofv3 GPU kernel hotspots

## Context

Per the roadmap agreed when building the first tool ([docs/plans/01-rocprof-sys-hotspots.md](../../Documents/Porting_Adventure/Friendly_Rocprof/docs/plans/01-rocprof-sys-hotspots.md)): `rocprof-sys` only ever gives host-side (CPU) timing, even for rows that look like GPU work. Real GPU kernel execution hotspots need `rocprofv3`. This plan builds that second, GPU-only tool, mirroring the first tool's shape (launcher + standalone extractor, same CLI conventions) for a consistent "family" of tools.

Fresh research into `rocprofv3` (cross-checked against ROCm 7.0.2-tagged docs, the rocprofiler-sdk GitHub source, and AMD's own blog examples) found it's actually **much more straightforward** than `rocprof-sys` for this purpose:

- `rocprofv3 --kernel-trace --stats --output-format csv -- <app>` needs no binary instrumentation/rebuild (transparent interception via ROCm's tool-registration mechanism) and writes a `kernel_stats.csv` with a schema **confirmed directly from the tool's own source** (`generateCSV.cpp`, `statistics.cpp`): `Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev` — only `Name` is quoted; the numeric columns can be plain-decimal *or* scientific notation depending on magnitude (Python's `float()` handles both natively).
- **rocprofv3 already aggregates by kernel name for you** (confirmed in `generateStats.cpp`: the stats map is keyed only on kernel name, summed across every stream/queue/agent) — no cross-row dedup needed within one file, unlike `rocprof-sys`'s hierarchical text tables.
- **Default output format is `rocpd` (SQLite), not CSV** — `--output-format csv` must always be passed explicitly, or the tool won't produce the CSV we parse at all.
- Default output path is `<output_dir>/%hostname%/%pid%_kernel_stats.csv` — PID-based naming, confirmed (via AMD's own MPI how-to page) to avoid collisions across MPI ranks with no extra work, exactly like `rocprof-sys`. The extractor needs a **recursive** glob (`<output_dir>/**/*_kernel_stats.csv`) to find these, since they're nested one level under a hostname subdirectory rather than sitting flat in `output_dir` like `rocprof-sys`'s files.
- **No `metadata.json`-equivalent exists for ROCm 7.0.2.** The one candidate (`--output-config` → `<pid>_config.json`, which does carry the executed command and init/fini timestamps) is confirmed **absent from the 7.0.2-tagged docs** — it's a post-7.0.2 addition. So for our compatibility target, executable name/run-datetime/total-runtime header fields will realistically stay blank most of the time; only the MPI-rank-count fallback (distinct PIDs among scanned filenames, same technique as tool 1) has a reliable, version-independent data source. This is worth calling out to the user but doesn't change the design — same graceful-blank behavior as tool 1, it's just less often filled in here.
- `Percentage` in `kernel_stats.csv` is per-file (share of that single run's total kernel time) — not documented in prose, but corroborated across multiple examples. When aggregating multiple ranks' files, recompute a global `%total` the same way tool 1 does: sum of `TotalDurationNs` across the aggregated kernel's rows, divided by the sum of *every* row's `TotalDurationNs` across *all* scanned files (no root-detection ambiguity here — `kernel_stats.csv` has no hierarchy at all, every row is already a distinct, fully-summed kernel).

Given this is meaningfully simpler than tool 1 (clean documented CSV, native aggregation, a native percentage), the design directly reuses tool 1's now-established conventions rather than inventing new ones: same launcher/extractor split, same `-n/--top | --threshold | --all` selection modes, same best-effort/blank-on-missing header metadata philosophy, same rank-0-only auto-summary gating under MPI. The **scope stays GPU-kernel-only** (no HIP-API host-side bucket) — that's what makes this tool distinct from tool 1 and from the later combined tool.

## Design

### 1. `postprocess/rocprofv3_hotspots.py` — standalone extractor

Usage: `rocprofv3_hotspots.py <rocprofv3-output-dir> [-o hotspots.txt] [-n N | --threshold PCT | --all]` (default: top 20, same argparse mutually-exclusive group as tool 1).

Logic:
- Recursively glob `<output_dir>/**/*_kernel_stats.csv`.
- Parse each with Python's `csv` module (handles the single-quoted-field format natively); columns are `Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev` — use `Name`, `Calls`, `TotalDurationNs` only (the per-file `Percentage`/`AverageNs`/min/max aren't meaningful after cross-file aggregation and get recomputed).
- Aggregate by kernel `Name` across all files found: sum `Calls`, sum `TotalDurationNs`.
- `total_gpu_time_ns = sum of TotalDurationNs across every row in every scanned file`; each aggregated kernel's `%total = aggregated_TotalDurationNs / total_gpu_time_ns * 100` (blank/`n/a` if `total_gpu_time_ns` is 0, i.e. nothing was scanned — but that case is already the "nothing to report" fail-fast).
- Recompute `avg = aggregated_TotalDurationNs / aggregated_Calls` for display (µs), since averaging averages across files would be wrong.
- Apply the same `select_entries`-style top/threshold/all logic as tool 1 (reuse the same behavior, don't need to import cross-module — small enough to duplicate the ~15 lines, consistent with tool 1's file being fully standalone).
- Write `hotspots.txt`:
  - Header: generated timestamp, source directory, files scanned (relative to output_dir, since they're nested), executable/run-datetime/total-runtime (best-effort from `<pid>_config.json` if `--output-config` was used — check for it, parse leniently, blank if absent; call out in a header comment that this is realistically almost always blank on ROCm 7.0.2), MPI ranks (distinct PIDs parsed from the `<pid>_kernel_stats.csv` filenames — reliable regardless of ROCm version).
  - Section "GPU kernel hotspots" — table of `#, total(s), %total, calls, avg(us), kernel name`.
  - Footer note: this covers GPU kernel execution time only, not host-side (HIP API / launch overhead) time — point at `scripts/rocprof_sys_profile.sh` for that (completing the cross-reference tool 1 already has pointing here).
- Fail fast with a clear message if the directory doesn't exist or no `*_kernel_stats.csv` was found (nothing to report — e.g. user forgot `--kernel-trace --stats --output-format csv`).

### 2. `scripts/rocprofv3_profile.sh` — launcher

Usage: `rocprofv3_profile.sh [-o OUTPUT_DIR] [--top N | --threshold PCT | --all] [--no-summary] [--dry-run] -- <command...>`

Same MPI pattern as tool 1 (rocprofv3 also wraps a single process transparently, no instrumentation step): `mpirun -np 4 scripts/rocprofv3_profile.sh -o results/run1 -- ./app arg1 arg2`.

Behavior:
1. Parse flags (mirroring tool 1's parsing style); require a trailing command after `--`.
2. Check `rocprofv3` is on `PATH` — fail fast with the same style of clear error as tool 1.
3. Build and run: `rocprofv3 --kernel-trace --stats --summary --truncate-kernels --output-format csv -d "$OUTPUT_DIR" -- "$@"` (`--output-format csv` is non-negotiable — the extractor only understands CSV, and rocprofv3's own default is `rocpd`/SQLite; `--truncate-kernels` for readable names, matching AMD's own recommended usage; `--summary` is a free bonus console printout, harmless to leave on). `--dry-run` prints the command instead of running it, same as tool 1.
4. After the wrapped command exits, same rank-0-only gating as tool 1 (`OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`SLURM_PROCID`) before invoking `postprocess/rocprofv3_hotspots.py "$OUTPUT_DIR" <selection-args>`; same graceful warn-and-skip if `python3` is missing or the extractor fails.
5. Propagate the wrapped command's exit code.

### Files

- `scripts/rocprofv3_profile.sh` (new)
- `postprocess/rocprofv3_hotspots.py` (new)
- `postprocess/tests/fixtures/rocprofv3_single_rank/<hostname>/1234_kernel_stats.csv` and `postprocess/tests/fixtures/rocprofv3_mpi_2rank/<hostname>/{2001,2002}_kernel_stats.csv` — hand-crafted, matching the source-verified schema (plain + scientific-notation numeric examples, to exercise the float parsing), nested under a hostname subdirectory to match rocprofv3's real default layout.
- `postprocess/tests/test_rocprofv3_hotspots.py` — `unittest` (stdlib only) covering: recursive CSV discovery, cross-file aggregation by kernel name, `%total`/avg recomputation, selection modes, and the `no *_kernel_stats.csv found` fail-fast.
- `docs/plans/02-rocprofv3-gpu-hotspots.md` — this approved plan, written verbatim per `CLAUDE.md`'s workflow.
- `README.md` — add the new tool alongside the existing rocprof-sys one.
- `docs/DEVELOPMENT_HISTORY.md` (+ regenerated `.docx`) — updated before the commit, per `CLAUDE.md`.

## Verification

- `bash -n scripts/rocprofv3_profile.sh` (and `shellcheck` if available, same caveat as tool 1 if not).
- `python3 -m unittest postprocess/tests/test_rocprofv3_hotspots.py -v` against the hand-crafted fixtures.
- Manual dry run with a stubbed `rocprofv3` on `PATH` (same technique used to verify tool 1, since no ROCm/GPU exists in this environment): confirm flag construction, `--output-format csv` is always present, and rank-0-only summary gating.
- Run the extractor directly against the fixtures and inspect the generated `hotspots.txt`, same as was done for tool 1.
