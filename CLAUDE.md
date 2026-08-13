# CLAUDE.md

## Project

Open-source bash scripts and post-processing tools built around AMD's ROCm profiling stack — `rocprofv3`, `rocprof-system`, and `rocprof-compute`. Reference documentation: https://rocm.docs.amd.com/en/latest/. Everything developed here must stay retro-compatible with ROCm 7.0.2, even as newer ROCm versions are targeted going forward.

The tools assume **no prior knowledge of the codebase being profiled**, beyond:
- Written in C/C++ or Fortran
- Depends on ROCm and an MPI library
- May contain OpenMP directives, either `target` offload or CPU-based OpenMP parallelism

Don't design scripts around assumptions specific to any one target application (build system, source layout, naming conventions) — treat the profiled code as an opaque binary/launch command plus whatever rocprof/rocprof-system/rocprof-compute exposes.

## User-facing conventions (naming and `-h`/`--help`)

Every script under `scripts/` and every post-processing tool under `postprocess/` must follow both of these — they apply to all future tools, not just the ones that exist today:

- **Name the tool after what it does, not the AMD tool it wraps.** The underlying `rocprofv3`/`rocprof-sys`/`rocprof-compute` name belongs in code comments and in the `-h`/`--help` text (see below), never in the file name or in the report/output it produces. Established pattern: launcher scripts (`scripts/`) are named `profile_<what>` (e.g. `profile_CPU_hotspots.sh`), post-processing tools (`postprocess/`) are named `extract_<what>` (e.g. `extract_CPU_hotspots.py`) — follow this pattern for new tools unless there's a good reason not to.
- **`-h`/`--help` must lead with a short (~10-15 lines), jargon-free explanation** of the tool's purpose and its main limitation, written for someone who has never heard of `rocprofv3`/`rocprof-sys`/`rocprof-compute` — before the existing options/flags table, not replacing it. End that explanation with one line naming the specific AMD tool used under the hood and a link to its official ROCm documentation (e.g. "Under the hood, this uses AMD's rocprofv3 — see <link> for details."), so a curious user can go straight to the authoritative source. For bash scripts this is a paragraph prepended to the `usage()` heredoc; for Python tools, pass a dedicated help string to `argparse` (with `formatter_class=argparse.RawDescriptionHelpFormatter`) instead of reusing the module docstring — keep the more technical module docstring in place as a code comment for future maintainers.

## Environment constraints

This dev machine is CPU-only (4 cores, no GPU, no ROCm install, no HPC access) — see hardware constraint notes. Consequences:
- Nothing here can be run against a real GPU or validated end-to-end against actual `rocprofv3`/`rocprof-system`/`rocprof-compute` output in this environment.
- Final validation on real HPC hardware is done manually by the user — do not claim a script "works" or is "verified" based only on local testing; local testing here can only confirm shell syntax, argument parsing, control flow, and logic against synthetic/mocked inputs (e.g. hand-crafted sample CSV/JSON matching the documented output schema).
- When MPI is exercised locally (e.g. testing a launcher script), cap `-np` at 4.

## Test

Since real rocprof tooling isn't available here, prefer:
- `bash -n <script>` / `shellcheck <script>` for syntax and lint checks
- Exercising script logic against small hand-crafted fixture files that mimic real rocprof output formats (documented schema, not invented), placed under a fixtures/test directory
- Making dependencies on external tools (`rocprofv3`, `rocprof-compute`, MPI launchers, etc.) explicit and checked at script startup, with a clear error if missing, so failures on the target machine are diagnosable

## Plan documentation

Plans are numbered `<major>.<minor>-<slug>.md` under `docs/plans/`. The major version marks a
distinct phase of work — `1.x` is the original tool suite (one plan per tool/feature); `2.x` is the
`postprocess/` consolidation refactor. After a plan is approved in plan mode, write its full
verbatim text to a new file continuing the *current* major version's minor sequence (e.g. the plan
after `2.1-postprocess-consolidation-refactor.md` is `2.2-<slug>.md`). Only bump the major version
when starting a genuinely distinct phase of work, not for every plan. If the as-built code later
diverges from the approved plan (e.g. a bug found during implementation changes the design), don't
edit the plan file itself — add a short header note pointing to what changed and where, and record
the divergence in [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md).

## Code comments

- **Comments describe current behavior, not history.** Say what a function does and why *the code
  needs it to work correctly* (a non-obvious invariant, a subtle constraint) — not the story of how
  it got that way, which real-data investigation motivated it, which plan introduced it, or what an
  earlier version did instead. That narrative belongs in
  [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md), not the source file; a comment that
  reads like a changelog entry has drifted from its job.
- **Write for a new developer joining the project, not a project historian.** Avoid comments that
  only make sense with full context of past sessions or plans ("see docs/plans/...", "confirmed via
  real test_apps HPC data", "this fixes the bug where..."). If a comment name-drops a specific
  investigation or a prior bug instead of just stating the current, standing rule, rewrite it.
- **Every module gets a top-of-file docstring** covering: its scope (what it owns, what it
  explicitly does not), the functions it exposes, and its general design philosophy (e.g.
  "classification only, no tree surgery" for a noise-filtering module) — enough for a new developer
  to navigate the file without reading every function first.

Before running any `git commit`, update [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) (new timeline row(s) + a narrative subsection) to cover the work being committed, regenerate `docs/DEVELOPMENT_HISTORY.docx` from it (`pandoc docs/DEVELOPMENT_HISTORY.md -o docs/DEVELOPMENT_HISTORY.docx -M title="Development History" --standalone`), and include both files in the commit alongside the code changes.
