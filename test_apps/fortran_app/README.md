# fortran_app

Small MPI-decomposed 1D stencil in Fortran, exercising two GPU-offload
backends (HIP via `iso_c_binding` interop, OpenMP `target`) as separate
compilation units. See [../README.md](../README.md) for the full matrix
this suite covers and general build/run instructions.

The HIP backend is a real C++/HIP kernel (`kernel_hip.cpp`) called from
Fortran through a small `bind(C)` interface module
(`kernel_hip_binding.f90`) — the same interop pattern real Cray-Fortran +
HIP codes use, and the reason this project's calltree tools were originally
built against exactly this pattern.

## Build

```
make COMPILER=gnu   # or cray, amd
```

## Run

```
mpirun -np 4 ./fortran_app [backend] [local_n] [niter]
```

- `backend`: `hip`, `omp`, or `all` (default `all` — runs every backend's
  full simulation, one after another, in a single execution so one
  profiling run captures every backend's kernel-launch shape at once)
- `local_n`: per-rank array size (default 1024)
- `niter`: iterations (default 5)
