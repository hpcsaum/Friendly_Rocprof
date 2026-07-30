# Development History

| Date | Change |
|---|---|
| 2026-07-30 | Project scaffolding and `CLAUDE.md` project rules established |

## 2026-07-30 — Project scaffolding and rules

Initialized the repository with a `scripts/` directory (bash scripts wrapping `rocprofv3`/`rocprof-system`/`rocprof-compute`) and a `postprocess/` directory (tools for parsing/analyzing their output), plus a README and `.gitignore` for typical profiler output artifacts (`.csv`, `.json`, `.db`, `results/`).

Added `CLAUDE.md` defining the project scope and working constraints:
- Targets AMD ROCm's profiling stack (`rocprofv3`, `rocprof-system`, `rocprof-compute`), per https://rocm.docs.amd.com/en/latest/, with retro-compatibility to ROCm 7.0.2.
- Tools must work against arbitrary, unknown target codebases — the only assumed facts are: written in C/C++ or Fortran, depends on ROCm and an MPI library, and may use OpenMP (offload or CPU-parallel).
- The project is open source.
- The dev machine is CPU-only with no ROCm/GPU access and no path to the HPC system, so nothing here can be validated end-to-end locally — final validation against real hardware is done manually by the user. Local checks are limited to syntax/lint and logic tests against hand-crafted fixture data matching documented rocprof output schemas.
- Adopted the same plan-documentation workflow used in `Heat_Convection_Solver`: approved plans get written verbatim to `docs/plans/`, and `docs/DEVELOPMENT_HISTORY.md` (plus its regenerated `.docx`) is updated before every commit.
