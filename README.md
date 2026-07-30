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
overhead calls, each with its share of total measured runtime. Note: this only covers
CPU-side timing — true GPU kernel execution time needs a different tool (`rocprofv3`,
planned separately). The report's header also includes the executable name, run
date/time, total runtime, and MPI rank count when `rocprof-sys` happened to record
them (best-effort from its `metadata.json`, since none of that lives in the timing
data itself) — any field it couldn't find is just left blank.

By default the report lists the top 20 entries per section, ranked by total time.
Pass `--top N` for a different count, `--threshold PCT` to instead list every
entry at or above PCT% of total runtime, or `--all` to list everything with no
truncation (works on both the launcher and the extractor):

```bash
scripts/rocprof_sys_profile.sh -o results/run1 --threshold 5 -- ./app arg1
```

The extractor also works standalone against any existing `rocprof-sys` output directory:

```bash
python3 postprocess/rocprof_sys_hotspots.py <rocprof-sys-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all]
```
