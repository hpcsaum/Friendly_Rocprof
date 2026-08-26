# Friendly Rocprof

User-friendly bash scripts and post-processing tools for AMD ROCm's profiling stack
(`rocprofv3`, `rocprof-sys`, `rocprof-compute`), aimed at porting and optimizing
C/C++/Fortran + MPI(+OpenMP) codebases. Retro-compatible with ROCm 7.0.2. See
[CLAUDE.md](CLAUDE.md) for project scope and constraints.

## Layout

- `scripts/` — bash launcher scripts that run the profiling tools for you and generate a
  report automatically.
- `postprocess/tools/` — the post-processing scripts that turn raw profiler output into the
  reports described below. Each one also works standalone against output you already have.
  See [postprocess/README.md](postprocess/README.md) if you're extending or debugging them.

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

This generates `results/run1/hotspots.txt`, listing the top CPU-side hotspots
(candidates for GPU offload) and top GPU-API/launch overhead calls, each with
its share of total measured runtime — plus `calltree.txt`, an indented call
tree of the same run (see below). No rebuild or instrumentation needed. Note:
this only covers CPU-side timing — true GPU kernel execution time needs the
GPU tool below. The report's header also includes the executable name, run
date/time, total runtime, and MPI rank count when available.

Entries are ranked by **self time** (a function's own work, not counting
time spent in whatever it calls) by default — pass `--unfiltered` for the
older inclusive-time ranking, where a function that just calls the next
thing (`main`, a thin wrapper, ...) can crowd out the ones doing real work.

By default the report lists the top 20 entries per section. Pass `--top N`
for a different count, `--threshold PCT` to instead list every entry at or
above PCT% of total runtime, or `--all` to list everything with no
truncation (works on both the launcher and the extractor):

```bash
scripts/profile_CPU_hotspots.sh -o results/run1 --threshold 5 -- ./app arg1
```

For MPI runs, the report also includes a **CPU load imbalance** table:
each function's average/min/max time and how much it varies across ranks,
so you can spot work that's unevenly distributed. Skipped (with a note) if
fewer than 2 ranks were profiled.

The extractor also works standalone against any existing `rocprof-sys` output directory:

```bash
python3 postprocess/tools/extract_CPU_hotspots.py <rocprof-sys-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all] [--unfiltered]
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

No rebuild or instrumentation needed here either. Generates
`results/run1/hotspots.txt` ranking GPU kernels by total device execution
time — genuine device time, unlike the CPU tool's view above. Same
`--top`/`--threshold`/`--all` selection and best-effort header as the CPU
tool, and a **GPU kernel load imbalance** table for MPI runs.

```bash
python3 postprocess/tools/extract_GPU_hotspots.py <rocprofv3-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all]
```

### Combined CPU+GPU hotspots — `profile_hotspots.sh`

Runs both of the above against the same command and merges them into one
coherent ranking, so you don't have to eyeball two separate reports to
figure out whether your top bottleneck is a CPU function or a GPU kernel.

```bash
# non-MPI
scripts/profile_hotspots.sh -o results/run1 -- ./app arg1 arg2

# MPI: same --mpi convention as the other tools
scripts/profile_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
```

The generated `hotspots.txt` shows one fused CPU+GPU ranking as the headline
answer, followed by the CPU-only and GPU-only views it was built from (so
you can see exactly what fed into it) and load-imbalance tables for both.
Also generates `calltree.txt`, same as the CPU tool. Same
`--top`/`--threshold`/`--all`/`--unfiltered` flags as the other tools,
applied consistently across every table.

The extractor takes both tools' output directories directly and does **not**
check that they came from the same executable or test case — that's on you:

```bash
python3 postprocess/tools/extract_hotspots.py <rocprof-sys-output-dir> <rocprofv3-output-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all] [--unfiltered]
```

### Call tree — `extract_calltree.py`

Where the hotspots reports above give you a flat ranked list, this one shows the actual
**call tree** — real function nesting, so you can see *what called what*, not just which
functions took the most time. Generated automatically alongside `hotspots.txt` by
`profile_CPU_hotspots.sh`, `profile_hotspots.sh`, and `instrument_hotspots.sh trace` — or
run standalone against any of their output directories:

```bash
python3 postprocess/tools/extract_calltree.py <output-dir> [-o calltree.txt] [--max-depth N] \
  [--show-gpu-api] [--show-rocprofsys-internals] [--show-mpi-internals] \
  [--show-compiler-runtime] [--show-all-internals]
```

This is the **sampling-based** variant (the default): it shows the tree's true depth,
including real intermediate frames that were never explicitly instrumented, at the cost of
timing that's only statistically approximate. The raw sampled stack also contains internal
noise (rocprof-sys's own instrumentation, GPU-API calls, MPI library internals,
compiler-runtime helpers) — hidden by default, revealed one category at a time with the
`--show-*` flags above, or all at once with `--show-all-internals`. `--max-depth N`
truncates the tree for readability (stating how many further nodes were hidden, not
silently dropping them).

When a paired `rocprofv3/` directory is available (from `profile_hotspots.sh`, or
`instrument_hotspots.sh trace`), real GPU kernel data is nested into the tree at the CPU
code that launched it, when that can be determined.

### Call tree (exact, shallower) — `extract_wallclock_calltree.py`

The faster, simpler alternative: built from rocprof-sys's own instrumented data instead of
sampling, so every call shown has **exact** timing — but the tree is only as deep as
rocprof-sys's own instrumentation boundaries, so a real but never-instrumented intermediate
frame is invisible.

```bash
python3 postprocess/tools/extract_wallclock_calltree.py <output-dir> [-o wallclock_calltree.txt] [--max-depth N] [--show-gpu-api]
```

Same GPU-kernel attachment and `--max-depth` behavior as `extract_calltree.py` above;
GPU-API/runtime noise is hidden by default (`--show-gpu-api` to reveal it).

### Hotspot caller chains — `extract_hotspot_callers.py`

Where the calltree tools above walk *down* from the program's entry point, this one walks
*up*: pick your top N hotspots, then for each one, show exactly how the program reached it
— the chain of callers from the entry point down to that function, for every distinct place
it was called from. Useful when a hot function's name alone doesn't say enough (a generic
helper, something called from more than one place) and you need to see its actual call
path(s).

```bash
python3 postprocess/tools/extract_hotspot_callers.py <output-dir> [gpu_dir] [-o hotspot_callers.txt] \
  [-n TOP_N | --threshold PCT | --all] [--unfiltered] [--max-depth N]
```

Pass a second, `rocprofv3` directory (or a combined run directory containing both
`rocprof-sys/` and `rocprofv3/`) to rank CPU functions and GPU kernels together and get a
hot kernel's own caller chain traced back through whichever CPU call site launched it —
omit it for a CPU-only report. `--max-depth N` here truncates each chain to its N nearest
callers, counting *up* from the hotspot itself, not down from the root.

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

```bash
# build only, auto-profiling with the default 1% threshold
scripts/instrument_hotspots.sh instrument -- ./app arg1 arg2

# build + run to produce a full trace, reusing a report you already have
scripts/instrument_hotspots.sh trace --report results/run1/hotspots.txt -- ./app arg1 arg2

# MPI: same --mpi convention as the other tools
scripts/instrument_hotspots.sh trace --mpi "mpirun -np 4" -- ./app arg1 arg2
```

Auto-profiling picks hotspot functions by self time, same as the CPU hotspots tool — pass
`--unfiltered` for the old inclusive-time selection. The raw selection is then widened a bit so
the resulting trace's shape stays meaningful: each selected function's immediate real caller is
pulled in too (`--ancestor-depth`, default 1; 0 disables it), and a hot GPU kernel's real CPU
owner is pulled in even if that owner wasn't itself a hotspot (wired automatically from the
auto-profiling scan's own `rocprofv3/` data). After the rewrite, this tool checks which requested
functions actually made it into the binary and warns (without stopping anything) about any that
didn't — inlining, optimization, or a name mismatch can all cause that.

```bash
python3 postprocess/tools/select_instrumented_functions.py --output-dir <rocprof-sys-output-dir> [-n TOP_N | --threshold PCT | --all] [--unfiltered] [--gpu-output-dir <rocprofv3-output-dir>] [--ancestor-depth N]
python3 postprocess/tools/select_instrumented_functions.py --report results/run1/hotspots.txt
```

If you also want automatic hotspots/calltree reports built from the trace instead of just the raw
`.proto` output, use `profile_traced_hotspots.sh` below instead of `instrument_hotspots.sh trace`.

### Selective instrumentation + reports — `profile_traced_hotspots.sh`

Does everything `instrument_hotspots.sh trace` does above, then goes further: converts the trace to
CSV and builds an actual `hotspots.txt` + `calltree.txt` from it — the "just give me the reports"
version, for when you don't want to run the conversion and report tools by hand afterward. Use
`instrument_hotspots.sh` directly instead if you only want the raw trace (e.g. to open in
Perfetto's own UI).

```bash
# build + trace + convert + report, in one invocation
scripts/profile_traced_hotspots.sh -- ./app arg1 arg2

# skip straight to an existing trace directory -- no executable needed
scripts/profile_traced_hotspots.sh --trace-report results/run1/trace

# MPI: same --mpi convention as the other tools
scripts/profile_traced_hotspots.sh --mpi "mpirun -np 4" -- ./app arg1 arg2
```

Two independent selection concepts happen to share flag names with `instrument_hotspots.sh`'s own
flags: bare `--top`/`--threshold`/`--all`/`--unfiltered` control the **final report** (how many
entries appear in `hotspots.txt`); `--instrument-top`/`--instrument-threshold`/`--instrument-all`/
`--instrument-unfiltered` control which functions get **instrumented** in the first place, for
expert users who want that to diverge from the report — left at the ≥1% default if none of the
`--instrument-*` flags are given. `--trace-report DIR` skips the sample/instrument/trace steps
entirely, and skips the CSV conversion too if `DIR` already has converted files.

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

# MPI: same --mpi convention as the other tools
scripts/profile_hotspot_kernels.sh --mpi "mpirun -np 4" -- ./app arg1 arg2
```

Each selected kernel is profiled on its **second** call only by default — a kernel's first
dispatch is usually slower (first-touch memory-allocation penalties, page faults) and would
give a skewed picture. Pass `--all-dispatches` to profile every call of every selected
kernel instead; only do this for a small test case, since it can multiply profiling time
substantially.

```bash
scripts/profile_hotspot_kernels.sh --all-dispatches -- ./small_test_case
```

After profiling, this auto-runs `rocprof-compute analyze` and shows its own output directly
(saved alongside the raw data too). Pass `--no-summary` to skip that and inspect the raw
workload directory yourself. If your MPI launch command targets more than one rank and your
installed `rocprof-compute` doesn't support that safely, the script refuses to run rather
than risk silent data corruption.

```bash
python3 postprocess/tools/select_hotspot_kernels.py --output-dir <rocprofv3-output-dir> [-n TOP_N | --threshold PCT | --all] [--all-dispatches]
python3 postprocess/tools/select_hotspot_kernels.py --report results/run1/hotspots.txt
```

### Kernel deep-dive from an existing trace — `profile_traced_hotspot_kernels.sh`

Combines the two tools above: if you already have a trace directory (from `instrument_hotspots.sh
trace` or `profile_traced_hotspots.sh`), this skips the fresh `rocprofv3` scan
`profile_hotspot_kernels.sh` would otherwise run — it resolves the biggest GPU kernels straight
from the trace's own recorded data, then profiles just those with `rocprof-compute`, the same way.

```bash
# resolve hotspot kernels from an existing trace, then profile them in detail
scripts/profile_traced_hotspot_kernels.sh --trace-dir results/run1/trace -- ./app arg1 arg2

# restrict kernel selection to one window of the trace (e.g. a steady-state iteration)
scripts/profile_traced_hotspot_kernels.sh --trace-dir results/run1/trace --time-range 20:45 -- ./app arg1 arg2

# MPI: same --mpi convention as the other tools
scripts/profile_traced_hotspot_kernels.sh --trace-dir results/run1/trace --mpi "mpirun -np 4" -- ./app arg1 arg2
```

Same `--top`/`--threshold`/`--all`, `--all-dispatches` (2nd-call-only by default), `--no-summary`,
and MPI-safety-gate behavior as `profile_hotspot_kernels.sh`. If the trace directory doesn't have
converted CSVs yet, they're produced first (via `convert_trace_to_csv.py`).

```bash
python3 postprocess/tools/select_hotspot_kernels.py --trace-dir results/run1/trace [-n TOP_N | --threshold PCT | --all] [--all-dispatches] [--time-range 20:45]
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
python3 postprocess/tools/extract_pop_metrics.py results/run1

# scaling study: also computes Computation Efficiency / Global Efficiency vs. run1
python3 postprocess/tools/extract_pop_metrics.py results/run1 results/run2 results/run4 --scaling strong
```

`--scaling {strong,weak}` is required whenever more than one directory is given: **strong**
scaling (fixed global problem size, more ranks) compares total useful compute time across
ranks; **weak** scaling (fixed problem size per rank, more ranks) compares the average
per-rank time instead. Each directory can be a combined CPU+GPU output or a CPU-only
directory — GPU kernel time is folded in automatically whenever a paired `rocprofv3/`
directory is found, and a few extra GPU-specific columns (offload efficiency, GPU
utilization, GPU load balance) appear automatically too.

Not every POP metric is computable with this toolchain (some need hardware/network data this
project doesn't collect) — see [docs/pop_metrics_reference.md](docs/pop_metrics_reference.md)
for the full picture of what's computed and why.

## Trace-based tools

The tools above all read `rocprof-sys`'s and `rocprofv3`'s already-aggregated summary output. This
project also has a second family of tools that reads a full **trace** instead: every individual
call and GPU kernel dispatch, with real timestamps, not a pre-summed total. That gives some reports
information the summary-based tools can't have (e.g. exact CPU-to-GPU kernel correlation, real
per-event categories instead of a name-based guess) at the cost of needing a trace-mode run to
begin with.

A trace-mode run is produced by any `rocprof-sys` run with `ROCPROFSYS_TRACE=1` set -- `trace` mode
in `instrument_hotspots.sh` above is one convenient way to get one, but not the only source; any
`rocprof-sys` trace experiment works. Either way, the raw output is a directory of per-rank Perfetto
`.proto` trace files, which `convert_trace_to_csv.py` turns into the flat CSV files the three tools
below read:

```bash
python3 postprocess/tools/convert_trace_to_csv.py <rocprof-sys-trace-dir> [-o <csv-dir>] [--unfiltered] [--trace-processor PATH]
```

By default this writes the `-gpu`/`-mpi`/`-other` partitioned trio per rank -- the shape needed for
exact GPU-kernel-to-launch-site correlation in the tools below. `--unfiltered` additionally writes a
plain, combined per-rank file with no GPU-arg columns -- cheaper, but a downstream tool loses exact
kernel placement if that's the only file present for a rank. This tool doesn't manage Perfetto's own
`trace_processor_shell` tool itself -- point it at an already-installed one with `--trace-processor
PATH` or `$FRIENDLY_ROCPROF_TRACE_PROCESSOR`, or leave both unset to fall back to PATH; see
https://perfetto.dev/docs/analysis/trace-processor for how to obtain it. Once you have that CSV
directory:

```bash
python3 postprocess/tools/extract_trace_hotspots.py <trace-csv-dir> [-o report.txt] [-n TOP_N | --threshold PCT | --all] [--unfiltered] [--time-range RANGE]
python3 postprocess/tools/extract_trace_calltree.py <trace-csv-dir> [-o calltree.txt] [--max-depth N] [--time-range RANGE] \
  [--show-gpu-api] [--show-rocprofsys-internals] [--show-mpi-internals] [--show-compiler-runtime] [--show-all-internals]
python3 postprocess/tools/extract_trace_pop_metrics.py <trace-csv-dir> [<more-trace-csv-dirs>...] [--scaling {strong,weak}] [-o report.txt] [--time-range RANGE]
```

- **`extract_trace_hotspots.py`** — the trace-based equivalent of `extract_hotspots.py` above: one
  combined CPU+GPU ranking, plus per-rank load imbalance. Since a trace already ties every kernel
  dispatch to the exact host call that launched it, there's nothing to separately reconcile the way
  the summary-based CPU/GPU tools have to.
- **`extract_trace_calltree.py`** — the trace-based call tree: real function nesting like
  `extract_calltree.py`/`extract_wallclock_calltree.py` above, but with GPU kernel dispatches
  nested in at the exact CPU call site that launched them (a trace's own `corr_id` links each
  dispatch to its host launch call directly, so this is never a guess). When that exact call site
  turns out to be a generic entry point shared by every kernel launch in the program (common under
  OpenMP `target` offloading), a kernel whose name embeds its owning function is instead placed at
  the real CPU call instance that was actually running right before it, found via the trace's own
  timestamps.
- **`extract_trace_pop_metrics.py`** — the trace-based equivalent of `extract_pop_metrics.py`
  above, with the same `--scaling {strong,weak}` convention for multi-run studies; GPU-specific
  columns always appear here (a trace always has both CPU and GPU visibility from one source,
  unlike the summary-based tool's paired-`rocprofv3`-directory case).

All three tools above also share `--time-range RANGE`, e.g. `--time-range 5:12.5` or
`--time-range 0:5,20:` (comma-separated, each half optional: `START:` runs to the end, `:END` runs
from the start) — restricts the report to one or more time windows, in seconds, letting you exclude
startup/teardown or focus on a single iteration. A function straddling a window boundary still
counts for the portion of its duration inside the window; `extract_trace_calltree.py`'s tree cuts a
subtree with no overlap anywhere within it, while keeping the chain to any surviving descendant
intact. Every report's header always states the window it reflects (the full run by default).

## Customizing noise filtering

Most tools above that read CPU-side data (both hotspots extractors, both calltree tools, the
caller-chains tool, the instrumentation-selection tool, and the POP-metrics tool — check a
tool's own `-h`/`--help` if you're unsure) classify rows using a bundled, editable set of
patterns: MPI internals, GPU-API calls, compiler-runtime helpers, and rocprof-sys's own
instrumentation wrappers are recognized and hidden or collapsed by default. If your
application uses a library or naming convention the bundled patterns don't recognize, pass a
small JSON file via `--extra-noise-config path/to/config.json` (or set
`$FRIENDLY_ROCPROF_NOISE_CONFIG` once for every run) to add, remove, or disable patterns
without touching any code. See [postprocess/README.md](postprocess/README.md) for the file
format and how this fits into the rest of the architecture.
