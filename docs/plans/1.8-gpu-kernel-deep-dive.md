# Tool 5: profile only the biggest GPU kernels with rocprof-compute

## Context

Tools 1-4 cover CPU-side sampling (rocprof-sys), GPU kernel timing (rocprofv3), a combined view,
and selective CPU-function instrumentation. The project's own scope (`CLAUDE.md`, `README.md`)
has always named a third AMD tool this suite would eventually wrap: `rocprof-compute` (the ROCm
7.x rename of Omniperf) — hardware-counter-level GPU kernel profiling. Tool 5 is the first
concrete implementation of that.

The user's ask: reuse the "hotspots → instrument → trace" *pipeline shape* tool 4 established
(find hotspots first, then run an expensive, detailed tool scoped to just those hotspots) but for
GPU kernels instead of CPU functions — run tool 2 (`profile_GPU_hotspots.sh`) to find the biggest
kernels, then build and run a `rocprof-compute` command restricted to just those, instead of
profiling every kernel the app launches (which is slower and noisier).

**Why this isn't literally tool 4's two-subcommand shape.** Tool 4 splits into `instrument`
(build a rewritten binary, stop) and `trace` (build, then run it) because the Dyninst binary
rewrite is a real, separate, reusable, expensive artifact. `rocprof-compute` has no such build
step — `rocprof-compute profile` *is* the run; there's nothing to "build" ahead of time. Per the
user's confirmed choice, tool 5 is a **single-mode script**: find kernels → run a `-k`-filtered
`rocprof-compute profile` → auto-run `rocprof-compute analyze` and surface its own output (no
custom parsing needed — `analyze` is already a human-facing summary tool, unlike rocprof-sys's
raw text tables). `--dry-run` (same as every other tool here) covers "show me the command
without running it."

**rocprof-compute facts this design depends on** (verified against the `rocm-7.0.2` tag of
`ROCm/rocprofiler-compute`'s own `src/argparser.py`/`src/rocprof_compute_base.py`, not just
rendered docs, since flag semantics matter here):
- `rocprof-compute profile -n <name> [-p <path>] [-k <substr1> <substr2> ...] [-d <n>] --
  <app> [args]`. `-n/--name` is functionally required (the tool `sys.exit`s without it).
  `-k/--kernel` is a **profile-time collection filter** (genuinely reduces what gets counted,
  not just a post-hoc report filter) matched as a **substring**, not a regex — space-separated
  in one `-k` invocation (`nargs="+"`), not repeated flags. No escaping needed (unlike
  rocprof-sys-instrument's regex `-R`), since rocprofv3's kernel names (already truncated by
  tool 2's own `--truncate-kernels`) just need to appear as a substring of themselves.
- `-d/--dispatch <n>`, combined with `-k`, selects **which numbered occurrence of each
  `-k`-matched kernel** to profile (1-based — `-d 1` is each kernel's first dispatch). **Per the
  user's own direct, hands-on correction** (overriding what the ROCm 7.0.2-tagged
  `docs/how-to/profile/mode.rst` literally says — "Dispatch filtering is based on the *global*
  dispatch index of kernels in a run" — which the user has confirmed by repeated manual use is
  not how it behaves when combined with `-k`; docs can be stale/imprecise here, direct usage
  wins): this script always adds a fixed `-d 2` alongside `-k`, so every selected kernel is
  profiled on its **second** call, deliberately skipping the first (first-touch/page-fault
  overhead makes a kernel's first dispatch unrepresentative of its steady-state cost). Not
  exposed as a configurable flag in v1 — hardcoded, per the user's explicit direction that this
  is enough for a beginner tool.
- Kernels dispatched only **once** in the whole run have no second occurrence, and per the user
  are "irrelevant anyway in the optimization process" — `select_hotspot_kernels.py` drops any
  kernel with `count < 2` from its selection entirely (not a fallback to occurrence 1 — just
  excluded), so the `-k` list passed to `rocprof-compute` never includes a kernel `-d 2` couldn't
  find. This exclusion (and `-d 2` itself) is skippable via `--all-dispatches` — see below.
- **Escape hatch: `--all-dispatches`** (opt-in, off by default). Drops `-d 2` entirely, so
  `rocprof-compute` profiles *every* dispatch of each `-k`-matched kernel (also lifts the
  `count < 2` exclusion, since without `-d` a single-call kernel is fine again). For a kernel
  called N times, this multiplies however many counter-collection passes rocprof-compute already
  needs (see the kernel-replay note above) by N — potentially turning a short profiling run into
  one that takes many times longer. When this flag is set, the script prints a loud warning
  (not just help text) before running:
  `"warning: --all-dispatches profiles every call of each selected kernel, not just the 2nd --
  for a kernel called N times this can multiply profiling time by roughly N (on top of
  rocprof-compute's own multi-pass counter replay). Only use this for a small test case
  specifically sized for rocprof-compute; for a normal/long-running application, drop this flag
  and let the default (-d 2, second call only) keep runtime bounded."` Same warning text goes in
  `-h`/`--help` and the README.
- `-p/--path` controls where output goes. Left default, it writes to
  `./workloads/<name>/<gpu-model>/` (auto-appending a GPU-model subdirectory rocprof-compute
  detects at runtime — not knowable ahead of time). **Given explicitly, that auto-append is
  skipped** — output goes exactly to the given path. This script always passes `-p` explicitly
  (never relies on the default), so the exact path is known ahead of time for the later
  `analyze -p <path>` call, without needing to detect/guess a GPU-model string.
- `rocprof-compute analyze -p <workload-dir> [-o <file>]` — plain CLI text table; `-o` fully
  redirects it to a file (nothing to stdout when given), no HTML/JSON report format exists at
  this ROCm version. This script won't pass `-o`; instead it pipes `analyze`'s stdout through
  `tee` so the user sees it immediately *and* it's saved to a file — simpler than a two-pass
  run-then-cat, and doesn't depend on `-o`'s own behavior.
- **Multi-rank MPI profiling is a real rocprof-compute feature, but its output-path templating
  (`%rank%`/`%hostname%`/`--output-directory`, with automatic per-rank isolation to avoid
  collisions) is confirmed absent through rocprofiler-compute 3.4.0** — the newest version with
  an actual git tag (ships with ROCm 7.2.x). Checked directly: `src/argparser.py` at every
  `rocm-7.x` tag through `rocm-7.2.4`, the `develop`/`amd-staging` branches' current HEAD, and
  their `docs/how-to/profile/mode.rst` source — zero occurrences of `%rank%`/`output-directory`/
  `mpirun`/`distributed` anywhere. The feature genuinely exists on AMD's live "latest" rendered
  docs page (re-verified with a literal, skeptical re-fetch, not a paraphrase) — but that content
  isn't in any branch or tag reachable from the public repo, so **no exact "added in version
  X.Y.Z" can be pinned down**; it's newer than 3.4.0 and not yet in a released version I can
  check. Without that isolation, `mpirun -n 4 rocprof-compute profile -p WORKLOAD_DIR -- app` on
  an unsupported version would have every rank write into the same `-p` path with no separation
  — a real collision risk, not theoretical, if silently allowed.

  **Design (per the user's explicit direction): probe the installed tool's actual capability at
  runtime, don't hardcode a version-number threshold** — self-correcting regardless of which
  exact version adds the feature, and avoids asserting a threshold this research couldn't
  actually confirm. Concretely, before doing anything else:
  1. `ROCPROF_COMPUTE_VERSION="$(rocprof-compute -v)"` — for display in messages only, not the
     decision itself.
  2. `rocprof-compute profile --help 2>&1 | grep -q '%rank%'` — the capability probe. `%rank%`
     is the specific, unambiguous marker of the multi-rank output-templating feature (confirmed
     present verbatim in AMD's docs description of it); a generic `--output-directory` string
     match risks false positives, `%rank%` doesn't.
  3. Only if `--mpi` was given: determine the actual rank count generically (works for any
     launcher, not just mpirun) by running the user's own `--mpi` launch command against a
     trivial no-op instead of the real app — `"${MPI_ARR[@]}" echo __rank_probe__` — and counting
     how many `__rank_probe__` lines come back (any MPI launcher spawns N copies of whatever
     command it's given, so N echoes = N ranks). If the probe produces zero matching lines (e.g.
     a malformed `--mpi` string), **abort rather than guess** — silently assuming 1 rank when the
     true count is unknown is exactly the unsafe path this check exists to avoid.
  4. Decision: rank count `== 1` → proceed regardless of capability (no collision possible with
     one rank). Rank count `> 1` and capability probe positive → proceed. Rank count `> 1` and
     capability probe negative → hard `SystemExit`/exit 1 with the user's own message, extended
     with the concrete numbers: `"error: MPI is not supported with this version of
     rocprof-compute ($ROCPROF_COMPUTE_VERSION detected $RANK_COUNT ranks; no %rank%
     output-isolation support found in 'rocprof-compute profile --help') -- rerun with a
     single-rank launch, or upgrade rocprof-compute."`
  5. This check runs early, right after option parsing and before even calling into tool 2 —
     no point spending time on hotspot discovery if the run is about to be refused at the
     rocprof-compute step. It also runs (and can abort) under `--dry-run`, since both probes are
     cheap/side-effect-free relative to the real workload (a `--version`/`--help` call, and a
     trivial `echo` under the user's own launch command) and a `--dry-run` preview should reflect
     the same go/no-go outcome a real run would reach — call this out explicitly in `-h`/`--help`
     and the README so it's not a surprise that `--dry-run` still runs something real under
     `--mpi`.

**Deliberately out of scope for v1** (call out in the plan doc, not silently skipped):
- No `--check-instrumented`-equivalent "did rocprof-compute actually capture these kernels"
  cross-check. Tool 4's version works because `rocprof-sys-instrument` writes a ground-truth
  `instrumented.json`; rocprof-compute's raw per-run output schema for confirming which `-k`
  substrings actually matched anything isn't researched here. A kernel that matches nothing
  (typo, or it just didn't run this time) fails silently for now — worth a follow-up once
  rocprof-compute's raw output format is understood.
- No custom parsing/reformatting of `analyze`'s output — see above, it's surfaced verbatim.

## Design

### New files

**`postprocess/select_hotspot_kernels.py`** — GPU-kernel analog of `select_hotspot_functions.py`,
much simpler since there's no regex-escaping concern:
- `labels_from_output_dir(rocprofv3_dir, top=None, threshold=None, show_all=False,
  require_multiple_calls=True)` — calls `extract_GPU_hotspots.aggregate(rocprofv3_dir)` then
  `.select_entries(entries, total_ns, top=, threshold=, show_all=)` (no `rank_by`/`unfiltered` —
  no self-vs-inclusive concept for GPU kernels); when `require_multiple_calls` (the default),
  **drops any entry with `e["count"] < 2`** (no second dispatch for `-d 2` to target — see
  Context); `--all-dispatches` passes `require_multiple_calls=False` to keep single-call kernels
  eligible too, since `-d` won't be used at all in that mode. Returns
  `sorted({e["label"] for e in selected})`. `SystemExit` if nothing scanned, mirroring
  `select_hotspot_functions.py`'s own error message shape; note this is a *different* condition
  than "nothing scanned" — a directory can scan fine and still end up with zero eligible kernels
  under the default filter, which should just mean an empty result, not an error (the launcher
  decides what to do with an empty `-k` list — see Pipeline step 3).
- `labels_from_report(report_path)` — mirrors `labels_from_report`'s exact technique: find the
  line containing `"GPU kernel hotspots"` (present in both tool 2's own report header,
  `"GPU kernel hotspots -- showing ..."` in `extract_GPU_hotspots.py`'s `write_report`, and
  tool 3's combined-report table 3 header, `"=== 3. GPU kernel hotspots (rocprofv3 run) --
  showing ... ==="` in `extract_hotspots.py` — one substring covers both, exactly like
  `"CPU compute hotspots"` already does on the CPU side), then the next line ending in
  `"kernel"` (the column header — `extract_GPU_hotspots.format_table()`'s header row is
  `#, total(s), %total, calls, avg(us), kernel`, 6 columns), then read rows until blank,
  `line.split(maxsplit=5)` (kernel name is the last field, `parts[5].strip()`, since a demangled
  C++ kernel signature can contain spaces) — **`maxsplit=5` here, not the CPU extractor's
  `maxsplit=6`, because this table has one fewer column** (no self-time metric). Also parses
  `parts[3]` (the `calls` column) as an int and applies the same `require_multiple_calls`-gated
  `count < 2` drop as `labels_from_output_dir` (same parameter, same default), for consistency
  regardless of which resolution path is used. `SystemExit` on a missing section/header/empty
  table, same shape as the CPU version.
- No `escape_for_instrument_regex`/`find_lost_functions`/`--check-instrumented` — not needed
  (substring match, no rewrite step, no ground-truth JSON to check against — see Context).
- CLI: `--report FILE` / `--output-dir DIR` (mutually exclusive, one required) plus
  `-n/--top`, `--threshold`, `--all`, `--all-dispatches` (passes `require_multiple_calls=False`
  through to whichever resolution function runs); prints one kernel name per line (no regex
  column needed, unlike tool 4's `label<TAB>regex` pairs) since `rocprof-compute -k` takes plain
  substrings.

**`scripts/profile_hotspot_kernels.sh`** — single-mode launcher, same idioms as every script here
(`MPI_ARR`/`MPI_FORWARD` split, `set +e`/capture/`set -e` exit-code idiom, `--dry-run` preview,
`command -v`-based dependency checks):
```
Usage: profile_hotspot_kernels.sh [options] -- <command> [args...]
```
- `-o, --output-dir DIR` — where tool 2's auto-profiling scan writes (default
  `profile_hotspot_kernels-scan-<timestamp>`); skipped entirely if `--report` is given (same
  `if [[ -z "$REPORT" ]]` gate as tool 4).
- `--report FILE` — reuse an existing `hotspots.txt` (from tool 2 or tool 3) instead of
  auto-profiling.
- `--top N` / `--threshold PCT` / `--all` — forwarded to both the auto-profiling call (tool 2)
  and `select_hotspot_kernels.py`; **default stays tool 2's own top-20** (unlike tool 4's
  deliberate `--threshold 1` override) since the user's own framing was "the 20 biggest kernels."
- `--workload-dir DIR` — rocprof-compute's own output location, forwarded as `-p`; default
  `profile_hotspot_kernels-workload-<timestamp>`. The `-n` name rocprof-compute requires is
  derived from this (`$(basename "$WORKLOAD_DIR")`) rather than exposed as a second flag —
  one concept, one flag.
- `--no-summary` — skip the auto `rocprof-compute analyze` step (same flag name as tools 1-4's
  "skip the auto-post-processing" convention, even though here "summary" means "run analyze").
- `--all-dispatches` — drop the default `-d 2` (and its `count >= 2` selection requirement),
  profiling every dispatch of each selected kernel instead of just its second call. Off by
  default. Prints the loud runtime-multiplication warning from Context immediately when set
  (before the dependency/MPI checks even run, so it's the very first thing the user sees), in
  both the real run and `--dry-run`.
- `--mpi "<launch cmd>"` — forwarded to both the tool-2 auto-profiling call and the
  `rocprof-compute profile` invocation itself, gated by the version-capability/rank-count check
  from Context (runs early, before step 1 below, and can abort the whole script). Help text and
  README document the check and its rationale (rocprof-compute's own multi-rank output isolation
  can't be assumed present; this script verifies it rather than guessing).
- `--dry-run`, `-h/--help`.

Pipeline:
1. Dependency check `rocprof-compute` (+ `python3`). Then the version-capability/rank-count gate
   from Context above — runs even under `--dry-run`; exits here if it decides to abort.
2. If no `--report`: call `"$GPU_LAUNCHER" --no-summary "${MPI_FORWARD[@]}" "${SELECTION_ARGS[@]}" -o "$OUTPUT_DIR" -- "$@"` (`GPU_LAUNCHER="$SCRIPT_DIR/profile_GPU_hotspots.sh"`) — identical call shape to how tool 4 calls tool 3.
3. Resolve kernel names: build `ALL_DISPATCHES_ARGS=(); [[ "$ALL_DISPATCHES" -eq 1 ]] &&
   ALL_DISPATCHES_ARGS=(--all-dispatches)` (same array-flag pattern as everywhere else), then
   `--report` → `python3 "$SELECTOR" --report "$REPORT" "${ALL_DISPATCHES_ARGS[@]}"`; else →
   `python3 "$SELECTOR" --output-dir "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" "${ALL_DISPATCHES_ARGS[@]}"`
   (reading straight from rocprofv3's `kernel_stats.csv`, not from a written report — same shape
   as tool 4's `--output-dir "$OUTPUT_DIR/rocprof-sys"` call). Read newline-separated kernel
   names into a `KERNELS` bash array (simpler than tool 4's tab-pair parsing, since there's no
   second regex column). Excludes single-call kernels unless `--all-dispatches` was given (see
   Design above).
   If `${#KERNELS[@]} -eq 0`: `SystemExit`/exit 1 with
   `"error: no hotspot kernel is called more than once in this run -- nothing eligible to
   profile with -d 2 (rerun with --all-dispatches to include single-call kernels too)"` rather
   than silently falling through to an unscoped or empty `rocprof-compute` invocation.
4. ```bash
   DISPATCH_ARGS=()
   [[ "$ALL_DISPATCHES" -ne 1 ]] && DISPATCH_ARGS=(-d 2)
   PROFILE_CMD=("${MPI_ARR[@]}" rocprof-compute profile -n "$WORKLOAD_NAME" -p "$WORKLOAD_DIR" \
                 -k "${KERNELS[@]}" "${DISPATCH_ARGS[@]}" -- "$@")
   ```
5. Run it (`set +e`/capture/`set -e`, same idiom as every other tool).
6. Unless `--no-summary`: `rocprof-compute analyze -p "$WORKLOAD_DIR" | tee "$WORKLOAD_DIR/analysis.txt"`.
7. `exit "$APP_EXIT"`.

### Docs

- New README.md section "### GPU kernel deep-dive — `profile_hotspot_kernels.sh`", following
  the exact structural pattern of tools 1-4's sections (plain-language paragraph + non-MPI/MPI
  bash examples + standalone-tool invocation line), explicitly noting: (a) this needs a real
  discrete GPU to run at all (no dry-run substitute possible on this dev machine), (b) the
  substring-match caveat for `-k`, and that every selected kernel is profiled on its **second**
  call only (`-d 2`, default) to skip first-touch/page-fault overhead on the first dispatch, with
  kernels called only once excluded entirely — plus the `--all-dispatches` escape hatch and its
  runtime-multiplication warning, spelled out in full (not just a one-liner, given the severity),
  (c) the runtime capability/rank-count check under `--mpi` and
  why it exists (rocprof-compute's per-rank output isolation is a real but not-yet-generally-
  released feature — confirmed absent through rocprofiler-compute 3.4.0/ROCm 7.2.x — so this
  script verifies the installed tool's own `--help` output rather than assuming either way, and
  will refuse to run a multi-rank job it can't confirm is safe), (d) the "no cross-check that
  requested kernels actually got profiled" limitation.
- Write this plan verbatim to `docs/plans/08-gpu-kernel-deep-dive.md` after approval.
- `docs/DEVELOPMENT_HISTORY.md`/`.docx` updated before committing, per `CLAUDE.md`'s workflow.

## Tests

`postprocess/tests/test_select_hotspot_kernels.py`, mirroring `test_select_hotspot_functions.py`'s
structure against the **existing** GPU fixtures (`rocprofv3_single_rank`, `rocprofv3_mpi_2rank`,
`rocprofv3_no_data` — no new fixtures needed, these already have real kernel names):
- `labels_from_output_dir`: top-N/threshold/`--all` selection against `rocprofv3_mpi_2rank`;
  `no_timing_data`-equivalent (`rocprofv3_no_data`) raises `SystemExit`. The `count < 2` exclusion
  is already exercisable on an existing fixture without adding a new one:
  `rocprofv3_single_rank`'s `__hipRegisterFatBinary` row has `Calls=1` — assert it's present in
  `extract_GPU_hotspots.aggregate()`'s raw output, absent from `select_hotspot_kernels`'s default
  (`require_multiple_calls=True`) selection, and present again with `require_multiple_calls=False`
  (i.e. `--all-dispatches`).
- `labels_from_report`: against a report freshly generated by both
  `extract_GPU_hotspots.write_report()` (tool 2's own report) and `extract_hotspots.write_report()`
  (tool 3's combined report, to prove the same parser covers both formats) — same dual-format
  proof `test_select_hotspot_functions.py` already does for the CPU side. Include one regression
  case with a fabricated multi-word kernel signature (e.g. a templated name with a space) to
  pin down the `maxsplit=5` column count, same spirit as the CPU extractor's own regression test
  for its `maxsplit=6`.

## Verification

- `python3 -m unittest discover postprocess/tests` (full suite).
- `bash -n scripts/profile_hotspot_kernels.sh`.
- `--dry-run` with and without `--mpi`, with and without `--report`, with and without
  `--all-dispatches`, confirming: the tool-2 auto-profiling call is skipped when `--report` is
  given; `-k ... -d 2` both appear together whenever at least one eligible kernel was resolved,
  and the script errors out (per Pipeline step 3) rather than printing an empty/unscoped command
  when zero kernels have `count >= 2`; with `--all-dispatches`, `-d` is absent from the printed
  command, the runtime-multiplication warning prints, and a single-call kernel (e.g. a stubbed
  fixture kernel with `Calls=1`) shows up in `-k` where it wouldn't otherwise; `-p`/`-n` are
  consistent between the printed `profile` and `analyze` commands; `--mpi` prefixes both the
  tool-2 call and the `rocprof-compute profile` call but never the `analyze` call (no MPI concept
  applies to reading already-collected data).
- Using the stubbed `rocprof-compute` binary on `PATH` (same technique as prior tools' manual
  verification): stub `-v`/`--version` to print a fake version string, stub `profile --help` to
  print text with/without a `%rank%` line, and stub the rank-probe's `echo` passthrough — confirm
  the gate proceeds when rank count is 1 regardless of capability; proceeds when rank count > 1
  and the stub's `--help` includes `%rank%`; aborts with the documented error when rank count > 1
  and `%rank%` is absent; aborts (doesn't guess) when the rank-probe produces no matching output.
- No real end-to-end run is possible here (no GPU/ROCm) — state this plainly in the PR/commit,
  same as every prior tool in this project; real validation is the user's, on the target system.
