# Friendly Rocprof

User-friendly bash scripts and post-processing tools for AMD ROCm's profiling stack
(`rocprofv3`, `rocprof-sys`, `rocprof-compute`), aimed at porting and optimizing
C/C++/Fortran + MPI(+OpenMP) codebases. Retro-compatible with ROCm 7.0.2. See
[CLAUDE.md](CLAUDE.md) for project scope and constraints.

## Layout

- `scripts/` — bash launcher scripts wrapping rocprof tools
- `postprocess/` — tools for parsing and analyzing rocprof output

## Tools

Tool names describe what each one does, not which AMD tool sits behind it —
that detail is in the code comments, not the name or the output, since these
scripts are meant to be usable without knowing anything about `rocprofv3`/
`rocprof-sys`. Run any script with `-h`/`--help` for a plain-language
explanation of its purpose and limits before you use it.

### CPU hotspots — `profile_CPU_hotspots.sh`

Get a quick "where is the time going, and what's still on the CPU" view,
without opening a full trace viewer (under the hood: a `rocprof-sys` run).

```bash
# non-MPI
scripts/profile_CPU_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: put mpirun/srun before the script, one rank per profiled process
mpirun -np 4 scripts/profile_CPU_hotspots.sh -o results/run1 -- ./app arg1 arg2
```

This runs a lightweight call-stack sample (no binary instrumentation needed)
and, once it finishes, generates `results/run1/hotspots.txt` listing the top
CPU-side hotspots (candidates for GPU offload) and top GPU-API/launch
overhead calls, each with its share of total measured runtime. Note: this
only covers CPU-side timing — true GPU kernel execution time needs the GPU
tool below. The report's header also includes the executable name, run
date/time, total runtime, and MPI rank count when available (best-effort,
since none of that lives in the timing data itself) — any field it couldn't
find is just left blank.

By default the report lists the top 20 entries per section, ranked by total
time. Pass `--top N` for a different count, `--threshold PCT` to instead
list every entry at or above PCT% of total runtime, or `--all` to list
everything with no truncation (works on both the launcher and the extractor):

```bash
scripts/profile_CPU_hotspots.sh -o results/run1 --threshold 5 -- ./app arg1
```

The extractor also works standalone against any existing `rocprof-sys` output directory:

```bash
python3 postprocess/extract_CPU_hotspots.py <rocprof-sys-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all]
```

### GPU hotspots — `profile_GPU_hotspots.sh`

Get the real GPU-side counterpart: which kernels actually spend time
executing *on the device*, ranked by total time (under the hood: a
`rocprofv3` run).

```bash
# non-MPI
scripts/profile_GPU_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: put mpirun/srun before the script, same convention as the CPU tool
mpirun -np 4 scripts/profile_GPU_hotspots.sh -o results/run1 -- ./app arg1 arg2
```

No rebuild or instrumentation needed either, and generates
`results/run1/hotspots.txt` ranking GPU kernels by total device execution
time — genuine device time, unlike the CPU tool's view above. Same
`--top`/`--threshold`/`--all` selection and best-effort header
(executable/run-datetime/total-runtime/MPI-rank-count) as the CPU tool,
though on ROCm 7.0.2 the metadata file that header is read from
(`--output-config`) doesn't exist yet, so those fields will usually be blank
there — only the MPI rank count (derived from output filenames) is reliably
available on 7.0.2.

```bash
python3 postprocess/extract_GPU_hotspots.py <rocprofv3-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all]
```

### Combined CPU+GPU hotspots — `profile_hotspots.sh`

Runs both of the above against the same command (one after the other — each
tool wraps a whole process, so they can't run concurrently) and merges them
into one coherent ranking, so you don't have to eyeball two separate reports
to figure out whether your top bottleneck is a CPU function or a GPU kernel.

```bash
# non-MPI
scripts/profile_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: same convention as the other two tools
mpirun -np 4 scripts/profile_hotspots.sh -o results/run1 -- ./app arg1 arg2
```

Combining two separately-measured runs isn't as simple as adding their
totals: the CPU run's total already includes time spent blocked inside
`hipStreamSynchronize`/`hipDeviceSynchronize`/a synchronous `hipMemcpy` —
i.e. the CPU literally waiting for the GPU — which is the *same* physical
time the GPU run counts again from the device side as kernel execution. To
avoid double-counting that overlap, the CPU run's "GPU API / launch
overhead" bucket (the one `extract_CPU_hotspots.py` already separates from
its "CPU compute" bucket) is subtracted out before the two totals are added:
`combined pool = (CPU total − GPU API/overhead) + GPU kernel total`. The
report shows this arithmetic explicitly rather than hiding it.

The generated `hotspots.txt` has four tables: (1) the fused CPU+GPU ranking
against that combined pool — the headline answer; (2) CPU compute hotspots
exactly as `extract_CPU_hotspots.py` would report them standalone; (3) GPU
kernel hotspots exactly as `extract_GPU_hotspots.py` would report them
standalone; (4) the GPU API/launch-overhead bucket that was subtracted out
of table 1, so you can see precisely what got removed and why. Same
`--top`/`--threshold`/`--all` selection as the other two tools, applied to
every table.

The extractor takes both tools' output directories directly and does **not**
check that they came from the same executable or test case — that's on you
(garbage in, garbage out):

```bash
python3 postprocess/extract_hotspots.py <rocprof-sys-output-dir> <rocprofv3-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all]
```
