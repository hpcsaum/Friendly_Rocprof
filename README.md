# Friendly Rocprof

User-friendly bash scripts and post-processing tools for AMD ROCm's profiling stack
(`rocprofv3`, `rocprof-sys`, `rocprof-compute`), aimed at porting and optimizing
C/C++/Fortran + MPI(+OpenMP) codebases. Retro-compatible with ROCm 7.0.2. See
[CLAUDE.md](CLAUDE.md) for project scope and constraints.

## Layout

- `scripts/` — bash launcher scripts wrapping rocprof tools
- `postprocess/` — tools for parsing and analyzing rocprof output

## Tools

### rocprof-sys CPU hotspots

Get a quick "where is the time going, and what's still on the CPU" view from a
first `rocprof-sys` profiling run, without opening the full Perfetto trace viewer.

```bash
# non-MPI
scripts/rocprof_sys_profile.sh -o results/run1 -- ./app arg1 arg2

# MPI: put mpirun/srun before the script, one rank per rocprof-sys-sample instance
mpirun -np 4 scripts/rocprof_sys_profile.sh -o results/run1 -- ./app arg1 arg2
```

This runs `rocprof-sys-sample` (lightweight call-stack sampling, no binary
instrumentation needed) and, once it finishes, generates `results/run1/hotspots.txt`
listing the top CPU-side hotspots (candidates for GPU offload) and top GPU-API/launch
overhead calls. Note: this only covers CPU-side timing — true GPU kernel execution
time needs a different tool (`rocprofv3`, planned separately).

The extractor also works standalone against any existing `rocprof-sys` output directory:

```bash
python3 postprocess/rocprof_sys_hotspots.py <rocprof-sys-output-dir> [-n TOP_N] [-o report.txt]
```
