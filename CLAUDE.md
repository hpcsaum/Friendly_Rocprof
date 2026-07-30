# CLAUDE.md

## Project

Open-source bash scripts and post-processing tools built around AMD's ROCm profiling stack — `rocprofv3`, `rocprof-system`, and `rocprof-compute`. Reference documentation: https://rocm.docs.amd.com/en/latest/. Everything developed here must stay retro-compatible with ROCm 7.0.2, even as newer ROCm versions are targeted going forward.

The tools assume **no prior knowledge of the codebase being profiled**, beyond:
- Written in C/C++ or Fortran
- Depends on ROCm and an MPI library
- May contain OpenMP directives, either `target` offload or CPU-based OpenMP parallelism

Don't design scripts around assumptions specific to any one target application (build system, source layout, naming conventions) — treat the profiled code as an opaque binary/launch command plus whatever rocprof/rocprof-system/rocprof-compute exposes.

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

After a plan is approved in plan mode, write its full verbatim text to a new numbered file in `docs/plans/` (e.g. `docs/plans/01-<slug>.md`), continuing the existing sequence. If the as-built code later diverges from the approved plan (e.g. a bug found during implementation changes the design), don't edit the plan file itself — add a short header note pointing to what changed and where, and record the divergence in [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md).

Before running any `git commit`, update [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) (new timeline row(s) + a narrative subsection) to cover the work being committed, regenerate `docs/DEVELOPMENT_HISTORY.docx` from it (`pandoc docs/DEVELOPMENT_HISTORY.md -o docs/DEVELOPMENT_HISTORY.docx -M title="Development History" --standalone`), and include both files in the commit alongside the code changes.
