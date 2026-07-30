# First tool: rocprof-sys hotspots (CPU + GPU-offload-candidate view)

## Context

`Friendly_Rocprof`'s goal is user-friendly wrappers around ROCm's profiling stack for porting/optimizing unknown C/C++/Fortran + MPI(+OMP) codebases. The very first workflow to support is: a user profiles an unfamiliar codebase for the first time with `rocprof-sys` (rocprofiler-systems, ex-Omnitrace) and wants a quick answer to "where is the time going, and which of that time is still on the CPU (i.e. a candidate to offload)?" — without having to learn `rocprof-sys`'s ~90 env vars or open the full Perfetto trace viewer.

Research into rocprof-sys (docs at rocm.docs.amd.com, cross-checked against the ROCm 7.0.2-tagged docs and AMD training material) established the following, which drives the design:

- **Lightweight sampling mode exists and is the right default for a first profile**: `rocprof-sys-sample` does statistical call-stack sampling with no binary instrumentation step, is documented as "fully compatible with MPI", and its config (`ROCPROFSYS_USE_SAMPLING`, `ROCPROFSYS_SAMPLING_FREQ`, `ROCPROFSYS_PROFILE`, `ROCPROFSYS_FLAT_PROFILE`, `ROCPROFSYS_TEXT_OUTPUT`, `ROCPROFSYS_JSON_OUTPUT`) is confirmed stable and unchanged between ROCm 7.0.2 and latest.
- **CPU-side timing lands in well-documented, stable "timemory" text tables** (`wall_clock-<pid>.txt` etc.), one pipe-delimited row per function: `LABEL|COUNT|DEPTH|METRIC|UNITS|SUM|MEAN|MIN|MAX|VAR|STDDEV|% SELF`. This schema is confirmed by a literal quoted example in the archived docs and is the most reliable, version-stable data source available.
- **The JSON twin's exact field names are not documented** (only inferred from a downstream project's reader) — the text table is the safer parse target.
- **GPU kernel launch calls (`hipLaunchKernel`, etc.) DO appear in this same CPU table, but their `SUM` is host-side launch overhead, not real device execution time** — true device kernel duration is only in the Perfetto trace or a separate, undocumented-schema `roctracer-<pid>.txt`. rocpd (SQLite) output doesn't exist until ROCm 7.1, so it's out of scope for our 7.0.2 compatibility requirement.
- Because of this, a first "hotspots" tool has a natural, honest scope: **rank CPU-side hotspots reliably from the timemory text tables, split into "GPU API/launch overhead" vs "everything else (offload candidates)" by name prefix, and explicitly caveat that true GPU device time requires a different data source** — rather than pretending to report GPU kernel hotspots from data that doesn't actually contain them.
- MPI ranks each get their own PID-suffixed output files in the same shared output directory (no per-rank subdirectory by default) — the hotspots tool must aggregate across however many `wall_clock*.txt` files it finds.

Per the two scope questions resolved with the user: build **both** a launcher and an extractor, as two independently-usable pieces (the extractor must work standalone on any existing rocprof-sys output directory), with the launcher optionally auto-invoking the extractor at the end. The extractor is **Python 3, stdlib-only** (no pip dependencies — reliable everywhere python3 is available on HPC systems); the launcher is bash, consistent with the rest of the project.

**Roadmap note:** the hotspots concept will eventually have three tools — this CPU-only one (rocprof-sys), a GPU-only one (built on `rocprofv3 --stats --kernel-trace --summary`, which already has documented, stable output back to ROCm 7.0.2), and a combined tool merging both. This plan covers **only the CPU-only tool**; the GPU-only and combined tools are separate, later work and out of scope here. `hotspots.txt`'s footer caveat should point at the GPU-only tool once it exists, but for now just points at running `rocprofv3` directly.

## Design

### 1. `postprocess/rocprof_sys_hotspots.py` — standalone extractor

Usage: `rocprof_sys_hotspots.py <rocprof-sys-output-dir> [-o hotspots.txt] [-n TOP_N]` (default `TOP_N=20`, default output `hotspots.txt` written into the given dir unless `-o` overrides the path).

Logic:
- Glob the given directory (non-recursive, matches how rocprof-sys writes files directly into `OUTPUT_PATH[/TIMESTAMP]`) for `*.txt` files, skip known non-timing files (`available.txt`, `instrumented.txt`, `excluded.txt`, `overlapping.txt`), and for each remaining file, auto-detect whether it's a timemory table by checking for the `LABEL|COUNT|DEPTH|METRIC|UNITS|SUM|...` header row before attempting to parse — skip silently (not fatal) if it doesn't match, since rocprof-sys's exact file set varies by what components a given run enabled.
- For each parsed row: strip the thread/rank prefix (`|NN>>>` or `|MM|NN>>>`) and hierarchy indentation (`|_` repeated per `DEPTH`) from `LABEL` to get a clean function name; keep `COUNT`, `SUM`, `% SELF`.
- Aggregate by clean function name across **all** files found (summing `COUNT` and `SUM`) — this is how multiple MPI-rank files and multiple threads combine into one global ranking.
- Classify each aggregated entry: name matching a GPU-API prefix (`hip`, `hsa`, `roctx`, `kfd`, `rocdecode`, `rocjpeg`, case-insensitive) or originating from a file whose name contains `roctracer`/`hsa` → **"GPU API / launch overhead"** bucket; everything else → **"CPU compute (offload candidates)"** bucket.
- Sort each bucket by summed `SUM` descending, take top `TOP_N`.
- Write `hotspots.txt`:
  - Header: source directory, list of files scanned, generation time.
  - Section "Top CPU compute hotspots (candidates for GPU offload)" — table of rank/function/total time (s)/calls/%self.
  - Section "Top GPU API / launch overhead" — same columns, with a note that these are host-side call overhead, not device execution time.
  - Footer caveat: true GPU kernel execution time isn't present in this data — check the Perfetto `.proto` trace (path noted if found in the directory) or run `rocprofv3 --stats --kernel-trace --summary` for GPU-side kernel hotspots.
  - If a `rocpd`-style `.db` is found, note its path (informational only — not parsed, since its schema is undocumented and ROCm-7.1+-only).
- Fail fast with a clear message if the directory doesn't exist or no timemory-format file was found at all (nothing to report); exit 0 with a note in the file if e.g. the GPU bucket is empty (that's a normal/valid outcome, not an error).

### 2. `scripts/rocprof_sys_profile.sh` — launcher

Usage: `rocprof_sys_profile.sh [-o OUTPUT_DIR] [-f FREQ_HZ] [--top N] [--no-summary] [--dry-run] -- <command...>`

Key design point from the MPI research: `rocprof-sys-sample` wraps **one** process, and the documented MPI pattern is `mpirun -np N rocprof-sys-sample -- ./app`. So this script wraps a single command the same way — **the user (or their job script) puts `mpirun`/`srun` before the call to this script**, e.g.:
```bash
mpirun -np 4 scripts/rocprof_sys_profile.sh -o results/run1 -- ./app arg1 arg2
```
Each rank then independently execs `rocprof-sys-sample <flags> -- ./app arg1 arg2`, and since all ranks share the same `-o` value, ROCm's own default per-PID file naming keeps them from colliding in the shared output directory — no per-rank logic needed in this script.

Behavior:
1. Parse flags; require a trailing command after `--`.
2. Check `rocprof-sys-sample` is on `PATH` — fail fast with a clear "install/module-load ROCm" message if missing (per `CLAUDE.md`'s "make dependencies explicit" rule).
3. Export `ROCPROFSYS_OUTPUT_PATH` (default `rocprof-sys-hotspots-output`), `ROCPROFSYS_TEXT_OUTPUT=1`, `ROCPROFSYS_JSON_OUTPUT=1`, `ROCPROFSYS_FLAT_PROFILE=1` (flat profile is what makes the extractor's aggregation-by-name meaningful instead of double-counting nested call-tree rows), `ROCPROFSYS_TRACE=0` (skip the heavy Perfetto trace by default, since the goal is the lightweight text summary — the full-trace workflow is a separate, later tool).
4. Build and run: `rocprof-sys-sample -f "$FREQ_HZ" -- "$@"` (`--dry-run` prints the constructed command and exported env vars instead of executing — this is also how the script gets tested here, since no ROCm/GPU is available on this dev machine).
5. After the wrapped command exits, if `--no-summary` wasn't given: check python3 is available (warn-and-skip, don't fail the whole run, if it's missing — the profiling data is already safely on disk either way) and whether this is rank 0 by checking `OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`SLURM_PROCID` (treat as rank 0 / always run if none of these are set, i.e. a plain non-MPI invocation) — only rank 0 invokes `postprocess/rocprof_sys_hotspots.py "$OUTPUT_DIR" -n "$TOP_N"`, avoiding every rank racing to summarize the same shared directory.
6. Propagate the wrapped command's exit code.

### Files

- `scripts/rocprof_sys_profile.sh` (new)
- `postprocess/rocprof_sys_hotspots.py` (new)
- `postprocess/tests/fixtures/` — hand-crafted sample `wall_clock-*.txt` files matching the documented schema (one single-rank flat-profile example, one 2-rank MPI hierarchical example) for the extractor to run against, since no real rocprof-sys install exists here.
- `postprocess/tests/test_rocprof_sys_hotspots.py` — `unittest`-based (stdlib only) tests against those fixtures: verifies parsing, prefix-stripping, cross-file aggregation, GPU-vs-CPU bucketing, and top-N sorting.
- `docs/plans/01-rocprof-sys-hotspots.md` — this approved plan, written verbatim per `CLAUDE.md`'s workflow.
- `docs/DEVELOPMENT_HISTORY.md` (+ regenerated `.docx`) — updated before the commit, per `CLAUDE.md`.

## Verification

- `bash -n scripts/rocprof_sys_profile.sh` and `shellcheck scripts/rocprof_sys_profile.sh` (if shellcheck is installed; note in the commit if it isn't and this step was skipped).
- `python3 -m unittest postprocess/tests/test_rocprof_sys_hotspots.py -v` — exercises the extractor against the hand-crafted fixtures.
- Manual dry run: `scripts/rocprof_sys_profile.sh --dry-run -- ./some_binary arg1` and with a fake `OMPI_COMM_WORLD_RANK` exported, to confirm flag parsing, env var construction, and the rank-0-only summary gating all behave as designed — this is as far as local verification can go without ROCm/GPU access (per `CLAUDE.md`'s environment-constraints section); real end-to-end validation against actual `rocprof-sys` output is left to the user on the HPC system.
- Run the extractor directly against the fixtures directory and eyeball the generated `hotspots.txt` for correct sections/formatting.
