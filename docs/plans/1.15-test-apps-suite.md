# Test-app suite for broader calltree/filter validation

## Context

Noise-filtering substrings across this project's postprocessing tools were built and validated
against real HPC data from exactly one target application, `Heat_Convection_Solver` (Cray Fortran
+ Cray's OpenACC/HIP-offload runtime + Cray MPICH). This isn't only a calltree concern: the base
`GPU_API_PREFIXES` list lives in `extract_CPU_hotspots.py` and is reused by `extract_calltree.py`/
`extract_calltree_traced.py`/`calltree_common.py`, and the combined `extract_hotspots.py` and
`extract_pop_metrics.py` tools depend on the same classification for correct CPU-vs-GPU-vs-noise
attribution — so any gap in the filter substrings affects the hotspots and POP-metrics tools too,
not just the call trees. This shows up directly in the code: `COMPILER_RUNTIME_SUBSTRINGS` and
`KERNEL_LAUNCH_LABEL_SUBSTRINGS`' `__cray_start_acc_kernel` entry are both explicitly commented as
Cray-specific, and no GNU/AMD compiler-runtime noise, no OpenMP-target-offload launch/runtime
symbols, no standard-parallelism (stdpar) offload symbols, and no Open MPI-specific internal
prefixes (`ompi_`/`opal_`/`orte_`, distinct from MPICH's `mpid_`/`mpidi_` already covered) have
ever been observed or filtered — in any of these tools. The
user wrote `Tests_parameters.txt` to define the intended coverage: 3 required languages
(Fortran/C/C++), 3 required compilers (Gnu/Cray/amd), required programming models (HIP, OpenMP
offload, standard parallelism for C++), 2 MPI distributions (OpenMPI, CrayMPI/MPICH) — with every
test doing real MPI communication and launching real kernels, complex enough to exercise the
calltree features, specifically so the filter-substring list can be expanded from real coverage
instead of one app's incidental structure.

This is explicitly a **two-phase** effort:
- **Phase 1 (this plan)**: design and write a small suite of real, buildable HPC apps under
  `test_apps/`, plus build/run documentation. This dev machine is CPU-only with no ROCm/HPC
  access (per `CLAUDE.md`), so nothing here can be compiled, run, or validated end-to-end
  locally — building/running on real HPC hardware is the user's manual next step, same as every
  other "final validation" step in this project.
- **Phase 2 (separate, future plan, blocked on Phase 1's real HPC output)**: once the user has
  built and run these apps and has real `rocprof-sys`/`rocprofv3` output to inspect, hand-craft
  new synthetic fixtures encoding whatever new patterns are found (GNU/AMD compiler-runtime
  noise, OpenMP-target/stdpar offload symbols, Open MPI internal prefixes, etc.) and extend the
  filter substrings accordingly wherever they actually live — `extract_CPU_hotspots.py`'s
  `GPU_API_PREFIXES` (shared by the hotspots and POP-metrics tools too, not just calltree) as well
  as `calltree_common.py`/`extract_calltree.py`'s calltree-specific tiers. Per the user's explicit
  decision this session,
  real captured profile directories are **never committed** — they stay transient, inspected
  manually, and translated into hand-crafted fixtures matching the existing
  "documented-schema, not invented" convention (`CLAUDE.md`, `test_calltree_common.py`'s
  `make_row()` convention). This plan does not attempt Phase 2.

**User's scope decisions this session**, which shape the design below:
1. One app per required language (Fortran, C, C++) — not a full language x compiler x model x
   MPI cross-product.
2. Compiler (Gnu/Cray/amd) and MPI distribution (OpenMPI/CrayMPI) are **build-time and launch-time
   choices only** — the same source must build under all three compilers and run under either MPI
   distribution with zero source changes, since on real HPC systems this is just which modules are
   loaded / which wrapper compilers and launch command are used.
3. Programming model (HIP / OpenMP target offload / stdpar) is tested via **separate compilation
   units within each app** (one module per backend), not separate apps.
4. Suite lives at `test_apps/` inside `Friendly_Rocprof` (not a sibling repo like
   `Heat_Convection_Solver`).
5. Kokkos and the Python language variant are optional per `Tests_parameters.txt` and are
   explicitly deferred, not attempted in this pass.

## Shared app design (same pattern in all three apps)

- **Workload**: a small MPI-decomposed iterative stencil, same spirit as `Heat_Convection_Solver`
  but deliberately minimal — each rank owns a slice of a 1D/2D array, exchanges halo data with
  neighbor ranks every iteration, and reduces a residual/checksum across all ranks.
- **Call depth** (so the resulting tree is non-trivial, per `Tests_parameters.txt`'s "complex
  enough to test the calltree features" requirement): `main` → `run_simulation` (iteration loop)
  → `step`, and `step` calls three real subroutines each iteration:
  - `exchange_halo` — point-to-point MPI (`MPI_Isend`/`MPI_Irecv`/`MPI_Wait`) with neighbor ranks
  - `reduce_residual` — collective MPI (`MPI_Allreduce`)
  - `launch_backend_kernel` — dispatches into whichever offload backend module is active
  This gives MPI P2P, MPI collective, and GPU-kernel-launch call sites all under one real,
  multi-level call chain per app.
- **Backends as separate compilation units**, each its own source file (`kernel_hip.*`,
  `kernel_omp.*`, `kernel_stdpar.*` for the C++ app only). By default a run exercises **all**
  backends sequentially within one execution (`--backend=hip|omp|stdpar|all`, default `all`) —
  maximizing how much one HPC job can teach the filter-substring inventory, since one profiling
  run then contains every backend's kernel-launch shape in the same call tree. `--backend=<one>`
  stays available for isolating a single backend's noise when tuning filters later.
- **Compiler axis**: one `Makefile` per app driven by a `COMPILER` variable (`gnu`/`cray`/`amd`),
  selecting the right compiler/wrapper names per toolchain, with override-able `MPICC`/`MPIFC`/
  `MPICXX`/`OFFLOAD_ARCH` variables (sensible defaults, documented as site-tunable) — consistent
  with this repo's "don't assume anything about the target site" principle (`CLAUDE.md`). Same
  source, three `make COMPILER=...` invocations, three binaries.
- **MPI-distribution axis**: no source or Makefile change at all — on Cray systems `cc`/`ftn`/`CC`
  already bundle whichever `craype-network-*`/MPI modules are loaded; elsewhere `mpicc`/`mpif90`/
  `mpicxx` resolve to whatever OpenMPI or MPICH install is active. Which distro actually runs is
  purely which modules were loaded before building, and which `--mpi "<launch cmd>"` string is
  passed to the existing `scripts/profile_*.sh` launchers at run time (e.g. `--mpi "srun -n 4"` vs
  `--mpi "mpirun -np 4"`) — reusing that existing convention exactly, no new plumbing needed.

## Per-app specifics

- **`test_apps/fortran_app/`**: `main.f90`, `kernel_omp.f90` (OpenMP `target` offload directives),
  `kernel_hip.cpp` + `kernel_hip_binding.f90` (HIP kernel launched from Fortran via
  `iso_c_binding`, mirroring the real Cray-Fortran-calls-HIP pattern this project already targets).
- **`test_apps/c_app/`**: `main.c`, `kernel_omp.c` (OpenMP `target` offload), `kernel_hip.cpp`
  (HIP kernel with an `extern "C"` launch entry point called from the C `main`).
- **`test_apps/cpp_app/`**: `main.cpp`, `kernel_omp.cpp` (OpenMP `target` offload), `kernel_hip.cpp`
  (HIP kernel), `kernel_stdpar.cpp` (`std::for_each`/`std::transform_reduce` with
  `std::execution::par_unseq` — GPU-offloaded via amdclang++'s `--hipstdpar` when built with
  `COMPILER=amd`; runs CPU-only under `gnu`/`g++`, since GNU has no GPU stdpar offload today —
  documented as a known caveat, still useful for exercising the call-tree shape and CPU-side
  stdpar runtime symbols). Kokkos module explicitly **not** built this pass (optional per
  `Tests_parameters.txt`) — noted in the app's README as a future extension point.

## `test_apps/README.md`

- Matrix table (language × compiler × MPI distro × programming model) showing what's covered and
  how compiler/MPI are build/launch-time axes rather than separate code paths.
- Build instructions per compiler: example `module load` lines for `PrgEnv-cray`/`PrgEnv-gnu`/
  `PrgEnv-amd` + a ROCm module, then `make COMPILER=cray|gnu|amd` per app.
- Run instructions reusing the existing `scripts/profile_hotspots.sh`/`profile_CPU_hotspots.sh`/
  etc. launchers unchanged, with `--mpi "srun -n 4"` or `--mpi "mpirun -np 4"` (note: this dev
  machine's 4-core cap only applies to local smoke-testing of the *build*, not to real HPC
  allocations).
- An explicit statement of the two-phase plan: this suite's job is to *produce* real captured
  output for the user to inspect manually; expanding `calltree_common.py`'s filter substrings from
  what's actually observed is Phase 2, a separate follow-up once real data exists — not attempted
  here.

## Out of scope for this plan

- Actually building/running anything on HPC (no access from this environment).
- Expanding any tool's filter substrings themselves — `calltree_common.py`, `extract_calltree.py`,
  or the shared `GPU_API_PREFIXES` in `extract_CPU_hotspots.py` (used by the hotspots and
  POP-metrics tools too) — all of that is Phase 2, blocked on real data from this suite.
- SLURM/PBS job-script templates — no existing convention in this repo to extend; the user wraps
  the `make` + `profile_*.sh` commands in their own site's job script.
- Kokkos module and the optional Python language variant (both explicitly optional in
  `Tests_parameters.txt`).

## Verification

- No Python is touched by this plan (all new files are Fortran/C/C++/Makefiles), so the existing
  `python3 -m unittest discover` suite is unaffected and doesn't need re-running for this change.
- Verification here is necessarily limited to careful manual code review (call structure, MPI
  usage, `iso_c_binding`/`extern "C"` correctness) and each Makefile handling all three
  `COMPILER` values with a clear error message on a missing/undetected tool (matching this
  project's existing "explicit dependency check, clear error if missing" convention, e.g. in
  `scripts/profile_CPU_hotspots.sh`) — this dev machine has no compilers/ROCm to actually build or
  run these apps, so real build/run verification is deferred to the user on HPC hardware, per this
  project's established constraint against claiming something is "verified" without real hardware.
- Ask the user to report back (or share captured output files from) their first HPC run once done,
  so Phase 2 (filter-substring expansion across the hotspots, calltree, and POP-metrics tools)
  can begin from real data.
