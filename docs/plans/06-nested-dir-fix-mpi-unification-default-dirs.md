# Fix nested-output-dir bug, unify MPI invocation across all 4 tools, rename default dirs

> **Post-implementation correction (2026-08-03):** this plan didn't address where
> `profile_hotspots.sh` (tool 3) writes its final combined `hotspots.txt` relative to the
> `-o DIR` the user passes. It was inheriting `extract_hotspots.py`'s own default
> (`args.rocprof_sys_dir/hotspots.txt`, i.e. nested at `DIR/rocprof-sys/hotspots.txt`), not the
> root of `DIR` — inconsistent with tools 1/2, where `hotspots.txt` always lands directly under
> the directory the user specified. Fixed by having `profile_hotspots.sh` pass an explicit
> `-o "$OUTPUT_DIR/hotspots.txt"` to the extractor instead of relying on its default. See
> `docs/DEVELOPMENT_HISTORY.md` for the corresponding entry.

## Context

**Bug report.** The user copied a real `rocprof-sys-sample` output directory from the HPC
system into `Heat_Convection_Solver/rocprof-sys-hotspots-output` and ran tool 1's extractor
against it. It failed with `error: no rocprof-sys timemory text table found`. The real directory
has an extra level: rocprof-sys wrote every per-process file one level deeper, inside an
auto-generated `2026-08-03_09.24`-style timestamped subdirectory (rocprof-sys's documented
default `ROCPROFSYS_TIME_OUTPUT` behavior — `%F_%H.%M` strftime naming, already anticipated by
this codebase's `TIME_OUTPUT_DIR_RE` constant, just never actually handled in the file-scanning
code). `extract_CPU_hotspots.py`'s `aggregate()`/`aggregate_per_rank()` only `glob.glob()` one
level, so they never see files one level down. Tool 3 and tool 4 both call into
`extract_CPU_hotspots.aggregate()` directly, so fixing it there fixes all three. Tool 2 never had
this problem — rocprofv3 nests under a hostname directory, and that extractor already globs
recursively (`**/*_kernel_stats.csv`, `recursive=True`).

**Stale-directory request, and where it led.** The user also asked for every script's default
output directory to include a time value, so re-running a tool with no `-o` can't silently mix a
new run's files with a stale previous run's. The obvious way to do that (`$(date ...)` in the
default) is unsafe today because tools 1-3 are invoked as `mpirun -np N script.sh -- app` — N
independent copies of the same bash script, which must all agree on one shared output directory
(rocprof-sys/rocprofv3 tell ranks apart by PID in the filename, not by directory). If rank
startup is skewed across a second boundary, independent per-rank `date` calls could disagree and
split one experiment across two directories.

**This reframed the invocation model itself.** The user then asked whether all four tools could
be invoked the same way tool 4 already is: one single call to the script, with the MPI launch
command passed as data via `--mpi "mpirun -np 4"`, rather than the script being wrapped
externally by `mpirun`. Checking why tool 4 works that way: it *has* to, because its binary
rewrite step must run exactly once, never once per rank. But that constraint is specific to the
rewrite step — it doesn't apply to tools 1-3 at all. `rocprof-sys-sample`/`rocprofv3` each wrap
"exactly one process" the same way regardless of whether their immediate parent process is bash
or `mpirun` — `mpirun -np 4 script.sh -- app` (today) and `mpirun -np 4 rocprof-sys-sample -- app`
(proposed) fork the same N processes either way. So tools 1-3 can adopt tool 4's exact
`--mpi "<launch command>"` convention with no loss of capability, and it has three concrete
benefits: (1) one consistent invocation style across all four tools instead of two, (2) it
removes today's per-script `RANK="${OMPI_COMM_WORLD_RANK:-...}"` guard entirely — the script now
runs as a single top-level driver process regardless of N, so "only run the summary once" is
just "run the summary" (no more depending on an MPI implementation setting the right rank-id env
var to avoid it running N times), and (3) it makes the stale-directory fix trivial — since each
tool now runs as exactly one process no matter how many MPI ranks it drives, a plain
`$(date +%F_%H.%M.%S)` default has no cross-rank race to worry about at all. This supersedes the
scheduler-job-id fallback design from the earlier round of questions — that complexity was
solving a problem (cross-process default-directory agreement) that no longer exists once there's
only one process making the default in the first place.

**Naming request.** Default directory names today are named after the underlying AMD tool
(`rocprof-sys-hotspots-output`, `rocprofv3-hotspots-output`, ...), inconsistent with this
project's own naming rule (scripts are named after what they do, not the AMD tool behind them —
see `CLAUDE.md`). The user asked to make default directory names match their script names.

## Part A — fix the nested-subdirectory bug (`postprocess/extract_CPU_hotspots.py`)

Make every filesystem scan in this module recurse, mirroring `extract_GPU_hotspots.py`'s existing
pattern exactly:

- `aggregate()`: `glob.glob(os.path.join(output_dir, "*.txt"))` →
  `glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True)`.
- `aggregate_per_rank()`: same change.
- `find_extra_artifacts()`: same recursive change for both the `*.proto` and `*.db` globs.
- `load_metadata()`: switch from reading exactly `output_dir/metadata.json` to a recursive glob
  for that filename, taking the first match if any — still returns `{}` (never errors) if none
  found.
- `guess_run_datetime(metadata, output_dir, scanned_files=())`: add the new `scanned_files` param
  (default keeps existing 2-arg call sites/tests working). Try the existing `output_dir`-string
  regex search first (keeps current behavior/tests passing), then fall back to searching
  `os.path.dirname(f)` for each `f` in `scanned_files` — needed because the timestamped directory
  is now *discovered* via recursive glob rather than passed in as part of `output_dir` itself.
- `gather_run_info(output_dir, scanned_files)`: pass `scanned_files` through to
  `guess_run_datetime`.

No changes needed in `extract_hotspots.py` (tool 3) or `select_hotspot_functions.py` (tool 4) —
both call `extract_CPU_hotspots.aggregate()` directly and inherit the fix automatically. No
changes needed in `extract_GPU_hotspots.py` (tool 2) — already recursive.

### Tests
Add fixture `postprocess/tests/fixtures/mpi_2rank_dated_subdir/`: same two ranks' `wall_clock-*.txt`
content as `mpi_2rank`, but nested one level inside a `2026-08-03_09.24/` subdirectory (plus a
top-level `metadata.json`), directly reproducing the reported bug. Add cases to
`test_extract_CPU_hotspots.py` (`aggregate()`/`aggregate_per_rank()` find the nested files;
`guess_run_datetime()`/`gather_run_info()` recover the date from the nested path; `write_report()`
succeeds instead of raising `SystemExit`), plus one end-to-end case each in
`test_extract_hotspots.py` and `test_select_hotspot_functions.py` against the same fixture,
proving tools 3 and 4 are fixed too.

## Part B — unify MPI invocation: `--mpi "<launch cmd>"` everywhere, no more "wrap the script"

Change `scripts/profile_CPU_hotspots.sh`, `scripts/profile_GPU_hotspots.sh`,
`scripts/profile_hotspots.sh` to match tool 4's existing convention exactly, and remove the "put
mpirun/srun in front of this script" documented pattern entirely (one convention across all four
tools, not two).

**`profile_CPU_hotspots.sh`**: add `--mpi "<launch cmd>"` option (parsed and split into an array
the same way `instrument_hotspots.sh` already does: `read -ra MPI_ARR <<< "$MPI_STR"`). Change
`CMD=(rocprof-sys-sample -f "$FREQ_HZ" -- "$@")` to
`CMD=("${MPI_ARR[@]}" rocprof-sys-sample -f "$FREQ_HZ" -- "$@")`. Remove the
`RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"` guard around the summary step —
now unconditional, since the script is a single driver process regardless of `-np`. Update the
header comment and `usage()` text to describe `--mpi`, dropping the "put mpirun before the
script" language (mirror tool 4's wording). Update the `--dry-run` block to reflect `MPI_ARR` in
the printed command, same style as tool 4's dry-run preview.

**`profile_GPU_hotspots.sh`**: identical shape — add `--mpi`, build `MPI_ARR`, change
`CMD=(rocprofv3 ... -- "$@")` → `CMD=("${MPI_ARR[@]}" rocprofv3 ... -- "$@")`, remove the RANK
guard, update comment/usage/dry-run.

**`profile_hotspots.sh`**: add `--mpi`, forward it to both sub-launcher calls instead of expecting
external wrapping — e.g. `MPI_FORWARD=(); [[ -n "$MPI_STR" ]] && MPI_FORWARD=(--mpi "$MPI_STR")`,
then `"$CPU_LAUNCHER" --no-summary "${MPI_FORWARD[@]}" -o "$CPU_DIR" -- "$@"` and same for
`$GPU_LAUNCHER`. Remove its RANK guard too. Update comment/usage/dry-run (including the `--mpi`
forwarding preview for both sub-calls).

**`instrument_hotspots.sh`** (tool 4): its *final trace run* (`RUN_CMD=("${MPI_ARR[@]}"
rocprof-sys-run -- ...)`) is unchanged — it already wraps the raw AMD binary directly, the same
model being extended to tools 1-3's raw AMD-tool invocations. But its *auto-profiling step*
currently wraps `mpirun` around `profile_hotspots.sh` itself:
`PROFILE_CMD=("${MPI_ARR[@]}" "$HOTSPOTS_LAUNCHER" ...)` — that's exactly the old "wrap the
script" convention being removed. Once `profile_hotspots.sh` accepts `--mpi` itself, this needs
to change to forward instead of wrap:
`PROFILE_CMD=("$HOTSPOTS_LAUNCHER" "${MPI_FORWARD[@]}" "${SELECTION_ARGS[@]}" "${NO_SUMMARY_ARGS[@]}" -o "$OUTPUT_DIR" -- "${APP_ARGS[@]}")`
(building `MPI_FORWARD` the same way as above). No other change needed in this script for Part B.

### README.md
Rewrite the "MPI" callouts in the CPU/GPU/combined-tool sections (currently
`mpirun -np 4 scripts/profile_CPU_hotspots.sh -o results/run1 -- ./app arg1 arg2`) to the `--mpi`
form, matching the selective-instrumentation section's existing style
(`scripts/instrument_hotspots.sh trace --mpi "mpirun -np 4" -- ./app arg1 arg2`). Flag this as a
breaking change to the MPI invocation convention for tools 1-3 in `docs/DEVELOPMENT_HISTORY.md`.

## Part C — rename default output directories to match script names; make them timestamped

Now that each tool is a single-process driver regardless of `-np` (Part B), a plain
`$(date +%F_%H.%M.%S)` suffix is race-free — no scheduler-job-id fallback needed.

Change each script's hardcoded default (computed once, before option parsing, so an explicit
`-o`/`--output-dir` still fully overrides it):
- `profile_CPU_hotspots.sh`: `rocprof-sys-hotspots-output` → `profile_CPU_hotspots-output-$(date +%F_%H.%M.%S)`
- `profile_GPU_hotspots.sh`: `rocprofv3-hotspots-output` → `profile_GPU_hotspots-output-$(date +%F_%H.%M.%S)`
- `profile_hotspots.sh`: `rocprof-combined-hotspots-output` → `profile_hotspots-output-$(date +%F_%H.%M.%S)`
  (its internal `$OUTPUT_DIR/rocprof-sys` / `$OUTPUT_DIR/rocprofv3` split is unchanged — those
  label which underlying tool's data lives where, not the tool's own name, so out of scope here)
- `instrument_hotspots.sh`: compute the timestamp once into a local var and reuse it for both
  defaults, so they agree for one invocation:
  `rocprof-sys-instrument-scan` → `instrument_hotspots-scan-<ts>`,
  `rocprof-sys-instrumented-trace-output` → `instrument_hotspots-trace-output-<ts>`

Update each script's `-h`/`--help` text: the shown default is now an example (includes a
timestamp), not a literal fixed string.

### `.gitignore`
Current `rocprof-sys-*-output/` already misses `rocprofv3-hotspots-output` and
`rocprof-combined-hotspots-output` (pre-existing gap, unrelated to this change) — and every name
is changing anyway. Replace it with one pattern per tool, suffixed for the timestamp:
```
profile_CPU_hotspots-output-*/
profile_GPU_hotspots-output-*/
profile_hotspots-output-*/
instrument_hotspots-scan-*/
instrument_hotspots-trace-output-*/
```

## Tests / Verification

- `python3 -m unittest discover postprocess/tests` — full suite (119 existing + new Part A
  tests).
- `bash -n` on all four scripts.
- Manual (Part A): confirm `extract_CPU_hotspots.py`, `extract_hotspots.py`, and
  `select_hotspot_functions.py` all now succeed against the new nested-subdirectory fixture
  instead of raising `SystemExit`.
- Manual (Part B): `--dry-run` each of the four scripts once with no `--mpi` (confirm command
  unchanged/no MPI prefix) and once with `--mpi "mpirun -np 4"` (confirm the printed command is
  prefixed accordingly) — for `profile_hotspots.sh`, confirm the preview shows `--mpi` forwarded
  to both sub-launcher lines; for `instrument_hotspots.sh`, confirm its auto-profiling preview
  line now shows `profile_hotspots.sh --mpi "..."` (forwarded) instead of
  `mpirun -np 4 profile_hotspots.sh` (wrapped).
- Manual (Part C): run each script with `--dry-run` and no `-o`, twice in a row a few seconds
  apart, confirm the two printed default directories differ and match the new
  `<script-name>-output-<timestamp>` shape.

## Docs
Write this plan verbatim to `docs/plans/06-nested-dir-fix-mpi-unification-default-dirs.md` after
approval. Update `README.md` per Part B above. Update `docs/DEVELOPMENT_HISTORY.md`/`.docx` with a
new timeline row + narrative section covering: the nested-output-dir bug and fix, the MPI
invocation convention unification (noting it's a breaking change for tools 1-3), and the
default-directory rename/timestamping — before committing.
