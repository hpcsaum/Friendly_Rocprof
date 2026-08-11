# cpp_app

Small MPI-decomposed 1D stencil in C++, exercising three GPU-offload
backends (HIP, OpenMP `target`, and standard-parallelism `std::execution`)
as separate compilation units. See [../README.md](../README.md) for the
full matrix this suite covers and general build/run instructions.

The stdpar backend (`kernel_stdpar.cpp`) is GPU-offloaded via amdclang++'s
`--hipstdpar` flag under `COMPILER=amd`; under `gnu`/`cray` it still
compiles and runs, but on the CPU only (via libstdc++'s parallel STL), since
neither toolchain has a GPU stdpar-offload path today. A Kokkos backend is
explicitly deferred (optional per `Tests_parameters.txt`) — a natural future
addition here as `kernel_kokkos.cpp`, same pattern as the other three.

## Build

```
make COMPILER=gnu   # or cray, amd
```

## Run

```
mpirun -np 4 ./cpp_app [backend] [local_n] [niter]
```

- `backend`: `hip`, `omp`, `stdpar`, or `all` (default `all` — runs every
  backend's full simulation, one after another, in a single execution so
  one profiling run captures every backend's kernel-launch shape at once)
- `local_n`: per-rank array size (default 1024)
- `niter`: iterations (default 5)
