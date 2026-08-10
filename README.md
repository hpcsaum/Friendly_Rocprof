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

# MPI: pass the launch command as data via --mpi, one rank per profiled process
scripts/profile_CPU_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
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

CPU entries are ranked by **self time** (a function's own work, not counting
time spent in whatever it calls) rather than inclusive/cumulative time — a
function that just calls the next thing (`main`, a thin wrapper, ...) won't
crowd out the ones actually doing the work. This needs real call-tree data,
so the launcher runs with `ROCPROFSYS_FLAT_PROFILE=0` (hierarchical mode) —
somewhat more overhead than the old flat-profile default; set
`ROCPROFSYS_FLAT_PROFILE=1` yourself beforehand if you need the lighter,
self-time-blind mode back for an overhead-sensitive run. Pass `--unfiltered`
(launcher and extractor both) for the old inclusive-time ranking.

By default the report lists the top 20 entries per section, ranked by self
time. Pass `--top N` for a different count, `--threshold PCT` to instead
list every entry at or above PCT% of total runtime, or `--all` to list
everything with no truncation (works on both the launcher and the extractor):

```bash
scripts/profile_CPU_hotspots.sh -o results/run1 --threshold 5 -- ./app arg1
```

For MPI runs, the report also includes a **CPU load imbalance** table:
each function's average/min/max time and how much it varies across
ranks (std_dev), including MPI calls — a rank that never called a
function counts as 0.0 for that rank rather than being left out, so a
function that only runs on some ranks shows up as maximally imbalanced.
This table is its own independent ranking by `std_dev`, using the same
`--top`/`--all` flags; `--threshold PCT` here means *coefficient of
variation* (`std_dev / avg >= PCT%`) instead of % of total runtime, since
a runtime-based cutoff has no equivalent meaning for a std_dev ranking.
Skipped (with a one-line note) if fewer than 2 ranks were profiled.

The extractor also works standalone against any existing `rocprof-sys` output directory:

```bash
python3 postprocess/extract_CPU_hotspots.py <rocprof-sys-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all] [--unfiltered]
```

### GPU hotspots — `profile_GPU_hotspots.sh`

Get the real GPU-side counterpart: which kernels actually spend time
executing *on the device*, ranked by total time (under the hood: a
`rocprofv3` run).

```bash
# non-MPI
scripts/profile_GPU_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: same --mpi convention as the CPU tool
scripts/profile_GPU_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
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

For MPI runs, the report also includes a **GPU kernel load imbalance**
table, same shape and `--top`/`--threshold`(coefficient-of-variation)/`--all`
semantics as the CPU tool's load-imbalance table above.

### Combined CPU+GPU hotspots — `profile_hotspots.sh`

Runs both of the above against the same command (one after the other — each
tool wraps a whole process, so they can't run concurrently) and merges them
into one coherent ranking, so you don't have to eyeball two separate reports
to figure out whether your top bottleneck is a CPU function or a GPU kernel.

```bash
# non-MPI
scripts/profile_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: same --mpi convention as the other two tools
scripts/profile_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
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

The generated `hotspots.txt` has six tables: (1) the fused CPU+GPU ranking
against that combined pool — the headline answer; (2) CPU compute hotspots
exactly as `extract_CPU_hotspots.py` would report them standalone; (3) GPU
kernel hotspots exactly as `extract_GPU_hotspots.py` would report them
standalone; (4) the GPU API/launch-overhead bucket that was subtracted out
of table 1, so you can see precisely what got removed and why; (5) CPU load
imbalance across ranks, reusing tool 1's own load-imbalance logic; (6) GPU
kernel load imbalance across ranks, reusing tool 2's. Same
`--top`/`--threshold`/`--all` selection as the other two tools, applied to
every table (tables 5-6 rank by std_dev, with `--threshold` meaning
coefficient of variation there instead of % of runtime). CPU-side entries
(tables 1, 2, and 5) rank by self time, same as tool 1 — pass `--unfiltered`
for the old inclusive-time view.

The extractor takes both tools' output directories directly and does **not**
check that they came from the same executable or test case — that's on you
(garbage in, garbage out):

```bash
python3 postprocess/extract_hotspots.py <rocprof-sys-output-dir> <rocprofv3-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all] [--unfiltered]
```

### Selective instrumentation — `instrument_hotspots.sh`

Once you already know your hotspots (from `profile_hotspots.sh` above),
instrumenting every function in the binary to get a detailed trace is slow
and distorts the very timing you're trying to measure. This tool builds a
`rocprof-sys-instrument` binary rewrite covering *only* the hotspot
functions, then (in `trace` mode) runs it to produce a full trace — kernels,
OpenMP regions, and MPI calls are captured automatically too, not just the
hotspot functions themselves.

Two modes, sharing the same build: `instrument` builds the instrumented
binary and stops; `trace` does the same build, then immediately runs it.
`trace` never assumes a binary was already built by a separate `instrument`
call — every invocation builds its own.

```bash
# build only, auto-profiling with the default 1% threshold
scripts/instrument_hotspots.sh instrument -- ./app arg1 arg2

# build + run to produce a full trace, reusing a report you already have
scripts/instrument_hotspots.sh trace --report results/run1/hotspots.txt -- ./app arg1 arg2
```

Auto-profiling picks hotspot functions by self time, same as tool 1 — pass
`--unfiltered` for the old inclusive-time selection (which tends to pick
pass-through wrapper functions rather than the ones doing real work).

Same `--mpi "<launch cmd>"` convention as the other three tools, but the
binary rewrite itself must happen exactly once, not once per rank — this
script places `--mpi` wherever it's actually needed (the auto-profiling
run, and `trace` mode's final run) and never in front of the one-time
build step, which doesn't execute your program at all:

```bash
scripts/instrument_hotspots.sh trace --mpi "mpirun -np 4" -- ./app arg1 arg2
```

After the rewrite, this tool checks `rocprof-sys-instrument`'s own
`instrumented.json` output against the functions you asked for and warns
(without stopping anything) about any that didn't actually make it into the
binary — inlining, optimization, or a name mismatch can all cause that.

```bash
python3 postprocess/select_hotspot_functions.py --output-dir <rocprof-sys-output-dir> [-n TOP_N | --threshold PCT | --all] [--unfiltered]
python3 postprocess/select_hotspot_functions.py --report results/run1/hotspots.txt
```

### GPU kernel deep-dive — `profile_hotspot_kernels.sh`

Once `profile_GPU_hotspots.sh` (above) has told you which GPU kernels are the biggest, this
tool goes deep on just those: it builds and runs a `rocprof-compute` command that collects
detailed hardware-counter data (occupancy, cache behavior, memory bandwidth, and more) for
*only* those kernels, instead of every kernel the application launches.

```bash
# non-MPI: auto-finds hotspots, then profiles them in detail
scripts/profile_hotspot_kernels.sh -- ./app arg1 arg2

# reuse a report you already have instead of re-profiling
scripts/profile_hotspot_kernels.sh --report results/run1/hotspots.txt -- ./app arg1 arg2
```

Each selected kernel is profiled on its **second** call only, not its first — a kernel's first
dispatch is usually slower than its steady-state cost (first-touch memory-allocation penalties,
page faults, and similar one-time overhead), so profiling it would give a skewed picture. A
kernel that only ran once has no second call to target and is left out entirely by default.

`--all-dispatches` overrides this and profiles *every* call of every selected kernel instead
(also bringing single-call kernels back in). Only use this for a small test case specifically
sized for this kind of profiling: for a kernel called N times, this can multiply how long
profiling takes by roughly N, on top of the multiple passes `rocprof-compute` may already need
per kernel to collect every counter it wants. For a normal, long-running application, leave
this off.

```bash
scripts/profile_hotspot_kernels.sh --all-dispatches -- ./small_test_case
```

After profiling, this auto-runs `rocprof-compute analyze` and shows its own output directly
(saved alongside the raw data too) — no extra parsing step, since `analyze` is already meant
to be read directly. Pass `--no-summary` to skip that and inspect the raw workload directory
yourself.

**MPI note:** `rocprof-compute`'s own support for safely profiling multiple MPI ranks at once
(so ranks don't overwrite each other's output) is a real feature in some version of the tool,
but is confirmed **absent** through `rocprofiler-compute` 3.4.0 (the version ROCm 7.2.x ships) —
no publicly released version could be confirmed to have it. Rather than assume either way, this
script checks your installed `rocprof-compute`'s own `profile --help` output for evidence of
that support, and separately determines how many ranks your `--mpi` launch command actually
produces. If more than one rank is detected and the installed version shows no sign of
supporting that safely, the script refuses to run rather than risk silent data corruption from
concurrent ranks writing into the same directory.

```bash
scripts/profile_hotspot_kernels.sh --mpi "mpirun -np 4" -- ./app arg1 arg2
```

There's currently no cross-check confirming that a requested kernel actually got profiled (the
way `instrument_hotspots.sh` checks against `rocprof-sys-instrument`'s own `instrumented.json`)
— a kernel that matches nothing (a typo, or one that just didn't run this time) fails silently
for now.

```bash
python3 postprocess/select_hotspot_kernels.py --output-dir <rocprofv3-output-dir> [-n TOP_N | --threshold PCT | --all] [--all-dispatches]
python3 postprocess/select_hotspot_kernels.py --report results/run1/hotspots.txt
```

### Computing POP metrics — `extract_pop_metrics.py`

Post-processing only, no launcher script: point this at output directories you've already
produced with the tools above, and it computes POP (Performance Optimization and Productivity
Centre of Excellence, https://pop-coe.eu/node/69)-inspired parallel efficiency metrics — Load
Balance, Communication Efficiency, and Parallel Efficiency from one directory; Computation
Efficiency and Global Efficiency too, given more directories from the same scaling study,
compared against the first as reference.

```bash
# single run: Load Balance / Communication Efficiency / Parallel Efficiency only
python3 postprocess/extract_pop_metrics.py results/run1

# scaling study: also computes Computation Efficiency / Global Efficiency vs. run1
python3 postprocess/extract_pop_metrics.py results/run1 results/run2 results/run4 --scaling strong
```

`--scaling {strong,weak}` is required whenever more than one directory is given: **strong**
scaling (fixed global problem size, more ranks) compares the *total* useful compute time summed
across ranks; **weak** scaling (fixed problem size per rank, more ranks) compares the *average*
per-rank useful compute time instead, since weak scaling's total is expected to grow with rank
count even at perfect efficiency.

Each directory can be either a `profile_hotspots.sh`/`instrument_hotspots.sh`-style combined
output (auto-detected `rocprof-sys/` + `rocprofv3/` subdirectories) or a plain
`profile_CPU_hotspots.sh`-style CPU-only directory — GPU kernel time is folded into the "useful
compute" pool (same double-counting-safe arithmetic as `extract_hotspots.py`) whenever a paired
`rocprofv3/` directory is found.

Not every POP metric is computed: Serialisation Efficiency and Transfer Efficiency need a
Dimemas-style ideal-network simulation (not part of this toolchain), and Instruction/IPC Scaling
need PAPI hardware counters (not present in `rocprof-sys` output unless
`ROCPROFSYS_PAPI_EVENTS` was explicitly configured for the run). See
[docs/pop_metrics_reference.md](docs/pop_metrics_reference.md) for the full picture of what's
computable and why. Communication time is classified by function-name prefix
(`MPI_`/`PMPI_`/`MPIR_`/`MPID_`) — MPICH/Cray-MPICH only for now.

Four more columns, project-specific extensions rather than official POP metrics (see
[docs/pop_metrics_reference.md](docs/pop_metrics_reference.md#gpu-specific-extensions-not-part-of-the-official-pop-catalog)),
appear automatically whenever the data supports them, no extra flags needed. Three appear
whenever a paired `rocprofv3/` directory is found, even for a single run: **`GPU-Off`** (GPU
Offload Efficiency — how much of the critical-path rank's time is still CPU-only *compute*,
excluding communication), **`GPU-Util`** (GPU Utilization — the GPU's raw share of wall-clock
time, which *does* drop when communication grows and starves the GPU of work, unlike `GPU-Off`),
and **`GPU-LB`** (GPU Load Balance — imbalance between GPUs specifically, separate from `LB`'s
whole CPU+GPU pool). The fourth, **`GPU-Eff`** (GPU Efficiency — whether the GPU's own contribution
scaled, isolated from `CompE`'s whole-pool view; a low value in strong scaling flags the per-rank
problem size shrinking below what keeps the GPU saturated), appears only in a scaling study where
both the reference and compared run have paired GPU data.
