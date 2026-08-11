# c_app

Small MPI-decomposed 1D stencil in C, exercising two GPU-offload backends
(HIP, OpenMP `target`) as separate compilation units. See
[../README.md](../README.md) for the full matrix this suite covers and
general build/run instructions.

## Build

```
make COMPILER=gnu   # or cray, amd
```

## Run

```
mpirun -np 4 ./c_app [backend] [local_n] [niter]
```

- `backend`: `hip`, `omp`, or `all` (default `all` — runs every backend's
  full simulation, one after another, in a single execution so one
  profiling run captures every backend's kernel-launch shape at once)
- `local_n`: per-rank array size (default 1024)
- `niter`: iterations (default 5)
