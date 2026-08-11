# test_apps

Small, real, buildable HPC applications used to validate this repo's
postprocessing tools (`extract_CPU_hotspots.py`, `extract_hotspots.py`,
`extract_pop_metrics.py`, `extract_calltree.py`, `extract_calltree_traced.py`)
against a broader set of real profiler output than any one target
application can provide. Their scope is defined by
[`../Tests_parameters.txt`](../Tests_parameters.txt).

## Why this exists

Every noise-filtering substring these tools currently know about
(`GPU_API_PREFIXES` in `extract_CPU_hotspots.py`, and the calltree-specific
tiers in `calltree_common.py`/`extract_calltree.py`) was discovered from one
target application's real data (Cray Fortran + Cray's OpenACC/HIP-offload
runtime + Cray MPICH). This suite exists to run real MPI + real GPU-kernel
workloads under compilers, programming models, and an MPI distribution that
application never used, so new noise patterns (GNU/AMD compiler-runtime
symbols, OpenMP-target/stdpar offload runtime symbols, Open MPI's own
`ompi_`/`opal_`/`orte_` internal prefixes, etc.) can be found and filtered
too.

**This is a two-phase effort.** This directory is Phase 1 — producing real
captured `rocprof-sys`/`rocprofv3` output for a human to inspect. Phase 2 —
actually expanding the filter substrings in the tools listed above — happens
separately, afterward, once that real output exists. This dev environment
has no ROCm/HPC access (see the repo's `CLAUDE.md`), so building and running
these apps is a manual step done by you on real HPC hardware; nothing here
has been compiled or run end-to-end.

Real captured profile directories are **not** committed to this repo, even
after Phase 2 — per this project's existing testing convention, only
hand-crafted fixtures matching a documented real schema live under
`postprocess/tests/fixtures/`. Inspect your real captures locally, then
report back (or share the relevant excerpts) so new fixtures and filter
entries can be added from what was actually observed.

## Coverage matrix

One app per required language; compiler and MPI distribution are build-time
and launch-time choices only (same source, no code changes), and programming
model is tested via separate compilation units (backends) within each app.

| | Fortran (`fortran_app/`) | C (`c_app/`) | C++ (`cpp_app/`) |
|---|---|---|---|
| Gnu / Cray / amd compiler | ✅ same source, `make COMPILER=...` | ✅ | ✅ |
| OpenMPI / CrayMPI-MPICH | ✅ same binary, launch-command choice | ✅ | ✅ |
| HIP backend | ✅ via `iso_c_binding` | ✅ | ✅ |
| OpenMP target-offload backend | ✅ | ✅ | ✅ |
| Standard-parallelism (stdpar) backend | — (not a Fortran model) | — (not a C model) | ✅ |
| Kokkos backend | — | — | deferred (optional) |

Python is also optional per `Tests_parameters.txt` and is deferred along
with Kokkos — not attempted in this pass.

Every app does real MPI communication (point-to-point halo exchange +
`MPI_Allreduce` each iteration) and launches real GPU kernels through a
multi-level call chain (`main`/`program` → `run_simulation` → `step` →
`exchange_halo`/`launch_backend_kernel`/`reduce_residual`), so the resulting
call trees are non-trivial — see each app's own `README.md` for its exact
structure.

## Build

Each app has its own `Makefile`, driven by a `COMPILER` variable:

```
cd test_apps/c_app        && make COMPILER=gnu    # or cray, amd
cd test_apps/fortran_app  && make COMPILER=cray
cd test_apps/cpp_app      && make COMPILER=amd
```

Load whatever compiler/MPI/ROCm modules your site needs *before* running
`make` — for example:

```
module load PrgEnv-cray craype-accel-amd-gfx90a rocm
module load PrgEnv-gnu rocm
module load PrgEnv-amd rocm
```

The exact module names are site-specific; nothing here assumes a particular
site's naming. Each `Makefile` checks its required compilers are on `PATH`
and fails with a clear message naming what's missing, rather than a raw
compiler-not-found error mid-build. Every build variable (`MPICC`/`MPIFC`/
`MPICXX`/`OFFLOAD_ARCH`/etc.) can be overridden on the `make` command line if
your site's names differ from the defaults — see the comments at the top of
each `Makefile`.

## Run

MPI distribution is purely which modules were loaded before building and
which launch command you use here — no rebuild needed to switch between
OpenMPI and CrayMPI/MPICH. Profile each app the same way you'd profile any
other target, using the existing launcher scripts unchanged. Run these from
the repo root (`Friendly_Rocprof/`):

```
scripts/profile_hotspots.sh --mpi "srun -n 4" -o results/cpp_app_amd_cray_all \
  -- test_apps/cpp_app/cpp_app all

scripts/profile_hotspots.sh --mpi "mpirun -np 4" -o results/fortran_app_gnu_ompi_all \
  -- test_apps/fortran_app/fortran_app all
```

(This dev machine's 4-core cap on local `-np`/`-n` only matters if you're
smoke-testing a build locally before moving to a real HPC allocation — it
doesn't apply to your actual HPC run.)

Each app defaults to `backend=all`, running every one of its backends'
full simulation in a single execution, so one profiling run's captured
output contains every backend's kernel-launch shape in the same call tree —
maximizing what a single HPC job teaches the filter-substring inventory.
Pass a specific backend name (see each app's `README.md`) to isolate one
backend's noise when tuning filters later.
