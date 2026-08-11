# Calltree tool: `postprocess/extract_calltree.py`

> **Update:** the tool this plan describes was renamed to
> `postprocess/extract_calltree_traced.py` (its default output is now
> `calltree_traced.txt`), and its rendering/kernel-attribution internals were
> refactored to share `postprocess/calltree_common.py` with a new tool. The
> name `extract_calltree.py` was reassigned to a new sampling-based tool with
> deeper (but noisier, statistically-approximate) call trees. See
> [14-sampling-calltree-tool.md](14-sampling-calltree-tool.md) and
> [DEVELOPMENT_HISTORY.md](../DEVELOPMENT_HISTORY.md) for details.

## Context

Plan 09 ([docs/plans/09-calltree-ancestry-thread-classification.md](09-calltree-ancestry-thread-classification.md))
built `attach_ancestry()` in `extract_CPU_hotspots.py` explicitly as "the reusable first step for
a future real call-tree view (walk `parent` links to render nesting/indentation)," but no tool
ever consumed it that way — every existing tool (`extract_CPU_hotspots.py`, `extract_hotspots.py`,
`extract_pop_metrics.py`) reads `scan_ranks()`'s output, which **merges same-label rows across
the whole file** for ranking purposes. The user wants that deferred feature built now: an actual
indented call tree (function nesting, not a flat ranked list), for both a sampling+rocprofv3
input (tool 1/3) and a rocprof-sys-trace input (tool 4).

Three research threads (this session) settled the open design questions:

1. **`scan_ranks()` cannot be reused for this.** Its final merge (keyed by `(label, gpu)`) is
   confirmed destructive to tree identity — two calls to the same function at different tree
   positions collapse into one summed entry. A calltree tool needs a new code path calling
   `parse_table_file()` + `attach_ancestry()` directly per file and rendering from the raw,
   unmerged per-node rows (`parent`/`depth`/`thread_id` intact).
2. **Multi-thread root detection has a gap worth working around, not fixing upstream**: a second
   OS thread's root row sometimes has `parent=None` without `is_thread_root=True` set (when its
   DEPTH resets to 0 rather than nesting one level under its spawn call — confirmed empirically
   against `multi_metric_rank` vs `gpu_spawned_thread` fixtures). Robust fix for a render-time
   consumer: treat **every row with `parent is None`** as its own tree root, independent of
   `is_thread_root` — `is_thread_root` stays reserved for `classify_gpu()`'s existing ancestry
   inheritance, unrelated to root enumeration.
3. **GPU kernel integration cannot be per-dispatch-precise from this toolchain's data.**
   `rocprofv3`'s `Correlation_Id` does link a kernel dispatch to its launching HIP API call (via a
   separate `hip_api_trace.csv`, confirmed real and documented), but `rocprof-sys`'s text tables
   have no per-call timestamps to match against — genuine per-dispatch placement needs the binary
   Perfetto trace, which (per research) has no stdlib-friendly Python parser: the `perfetto` pip
   package downloads a compiled `trace_processor` binary from Google's servers on first use, and
   the lower-level protobuf route needs the `protobuf` runtime plus Perfetto's large vendored
   schema — both break this project's "stdlib only, works air-gapped on an HPC login node" rule
   for the first time. **User's decision: build on the scan-directory/text-table route now (zero
   new dependencies), defer real trace parsing as explicit future work.**
4. **Tool 4's scan directory reliably pairs with real rocprofv3 kernel data** (confirmed: its
   launcher is `profile_hotspots.sh`, the *combined* CPU+GPU one, not the CPU-only one), **but its
   CPU-side `hipLaunchKernel`-style anchor point may be sparse or missing** (`ROCPROFSYS_USE_ROCM`
   is only set for the final instrumented `trace` run, not the scan step — confirmed by reading
   `scripts/instrument_hotspots.sh` directly). So: same kernel-placement logic as tool 1/3, with
   the same honest fallback when no anchor exists.
5. **One shared tool, not two** — tool 1/3 output and tool 4's scan directory turned out to have
   *identical* format and precision (both are `rocprof-sys-sample` + `rocprofv3` output; tool 4's
   "scan" step just *is* a `profile_hotspots.sh` run). Building two files for genuinely identical
   input would duplicate logic for no reason. A future genuine trace-parsing tool (reading the
   binary Perfetto trace) would be a clearly separate, differently-named tool once it exists.

## Approach

### New file: `postprocess/extract_calltree.py`

Follows the established pattern: `argparse` + `HELP_BLURB` + `RawDescriptionHelpFormatter`,
sibling imports `import extract_CPU_hotspots as cpu_tool` / `import extract_GPU_hotspots as
gpu_tool`, same directory auto-detection as `extract_pop_metrics.py`'s `resolve_run_dirs()`
(duplicated locally per this codebase's existing "small helpers are duplicated across standalone
tools" convention, not cross-imported).

**CLI**:
```
extract_calltree.py OUTPUT_DIR [-o OUTPUT] [--max-depth N] [--show-gpu-api]
```
- `OUTPUT_DIR`: one rocprof-sys(+rocprofv3) output directory — works identically whether it's a
  `profile_hotspots.sh`/`profile_CPU_hotspots.sh` output or an `instrument_hotspots.sh trace`
  scan directory (auto-detected `rocprof-sys/`+`rocprofv3/` subdirs, or flat CPU-only layout).
- `--max-depth N`: optional; omitted means print the whole tree, no truncation.
- `--show-gpu-api`: opt-in to also show GPU-API/runtime rows (`hip`/`hsa`/`roctx`/`kfd`/
  `rocdecode`/`rocjpeg`/`rocr`-prefixed, via `cpu_tool.classify_gpu()`) instead of pruning them —
  default is hidden, matching tool 3's own "CPU compute hotspots" bucket (user code + MPI calls
  survive automatically, since `classify_gpu()` never classifies `MPI_`/`PMPI_`/`MPIR_`/`MPID_`
  labels as GPU — no separate MPI allow-list needed).
- `-o`/`--output`: report path, default `<OUTPUT_DIR>/calltree.txt`.

**Per-rank tree construction** — `build_rank_tree(cpu_dir)`:
- Glob `wall_clock-<pid>.txt` per rank (reusing `cpu_tool.PID_SUFFIX_RE`); if absent for a given
  rank, fall back to `sampling_wall_clock-<pid>.txt` for that rank *entirely* — do **not**
  interleave rows from both metric types into one tree (their parent-links come from two
  independently-reconstructed call orders; splicing them would be structurally incoherent, unlike
  `scan_ranks()`'s per-label merge which only ever combines flat aggregates).
- For each rank's chosen file: `rows = cpu_tool.parse_table_file(path)`, then
  `cpu_tool.attach_ancestry(rows)`, then tag each row via `cpu_tool.classify_gpu(row, path)` —
  same sequence `scan_ranks()` already uses internally, just without its merge step.
- Enumerate roots: every row with `parent is None` starts an independent tree (handles both
  multi-thread shapes found in research, see point 2 above — no dependency on `is_thread_root`
  for this).

**Rendering** — `render_tree(root, max_depth, show_gpu_api)`:
- Depth-first walk via `parent`/child links (build a `children` list per node once, from the flat
  `rows` list, since `attach_ancestry()` only stores the parent pointer). Indent by 2 spaces per
  relative depth (`node_depth - subtree_root_depth`, not the raw file `DEPTH` column, so a second
  thread's subtree — which may start at DEPTH 2, per the `gpu_spawned_thread` fixture shape — is
  truncated relative to *its own* root, not the file's absolute depth numbering).
- A GPU-API-classified node (unless `--show-gpu-api`) prunes its entire subtree from the render —
  real user code is never expected nested inside GPU-runtime-internal calls (e.g. the
  `hipStreamCreate → hip::hipStreamCreate → hip::ihipStreamCreate` chain from the
  `gpu_api_nested_chain` fixture is pure GPU-runtime housekeeping, correctly removed wholesale).
- Each visible node prints `label  [calls=N, self=Xs, total=Ys]` (count, self_sum, sum — already
  on every row from `parse_table_file()`, no new computation needed).
- `--max-depth` truncation: once relative depth exceeds N, stop descending and print one line
  noting how many further levels/nodes were hidden there (no silent truncation, matching this
  project's existing "state what was cut, don't hide it" convention from `select_entries()`'s
  `top N of M` descriptions).

**GPU kernel integration** — `attach_kernel_summaries(roots, gpu_dir)`, only when a paired
`rocprofv3/` directory resolves:
- Walk the *raw, unfiltered* row list (before pruning) for rows matching a `KERNEL_LAUNCH_LABELS`
  prefix set (`hipLaunchKernel`, `hipModuleLaunchKernel`, `hipExtLaunchKernel`,
  `hipLaunchKernelGGL`, `hipExtModuleLaunchKernel`, `hipGraphLaunch` — documented as a best-effort
  known-launch-API list, same "not exhaustive" caveat style as this codebase's existing
  best-effort constants).
- For each match, walk `parent` links up to the nearest ancestor that will actually be *shown*
  (skips over anything itself GPU-API-classified, so this works correctly whether or not
  `--show-gpu-api` is set) — that ancestor is the attachment point.
- **Exactly one distinct attachment point found** → insert one synthetic child node there:
  `[GPU kernels -- rocprofv3]  [calls=N, total=Ys]`, itself expanded with one child per kernel
  name (from `gpu_tool.aggregate_per_rank()`, sorted by total time) — subject to the same
  `--max-depth` truncation as any other node.
- **Multiple distinct attachment points** → per the user's suggestion, don't just duplicate the
  full total with a disclaimer — use each site's own launch-call **count** (already on every row,
  no new data needed) as a proportional weight, and split the rocprofv3 kernel aggregate across
  sites accordingly: `weight_i = sum(count for launch-family rows whose nearest-visible-ancestor
  is site i)`; site *i* is attributed `total_kernel_data * (weight_i / sum(all weights))` of both
  the total time and each individual kernel name's count/time (uniformly scaled, since there's no
  finer-grained signal for which kernel *names* went where — see caveat below). Node label states
  the estimate basis plainly, e.g. `[GPU kernels -- rocprofv3, ~60% estimate: this site issued
  300/500 observed launch calls]`. This is a real assumption (roughly one kernel dispatch per
  launch call, uniform across sites) — documented explicitly as an estimate, not a measurement,
  in both the node label and the report's caveats section (e.g. `hipGraphLaunch`-driven code,
  where one call can dispatch many kernels, would skew this). Falls back to the old "not
  correlated, full total shown at each site" behavior only if every site's weight is zero (found
  a launch-family label with `count=0`, which shouldn't normally happen but is handled rather
  than dividing by zero).
- **No attachment point found at all** (the sparse-anchor case tool 4's scan directory can hit) →
  append a separate top-level section after all rank trees: `=== GPU kernels (rocprofv3) -- no
  launch call site found in CPU tree ===`, one subsection per rank with kernel data.

### Launcher script integration

Per the user's request, `calltree.txt` should be generated automatically alongside `hotspots.txt`
by the existing launcher scripts, not just available as a standalone `postprocess/` invocation.
Confirmed by reading the scripts directly:

- **`scripts/profile_CPU_hotspots.sh`** (tool 1, CPU-only): its existing auto-summary block
  (`if [[ "$RUN_SUMMARY" -eq 1 ]]; then python3 "$EXTRACTOR" "$OUTPUT_DIR" ...`, near the end of
  the script) gets a second, analogous call: `python3 "$CALLTREE_EXTRACTOR" "$OUTPUT_DIR" -o
  "$OUTPUT_DIR/calltree.txt" "${CALLTREE_ARGS[@]}"`, same `command -v python3` guard, same
  warn-not-fail-the-script behavior on extractor failure. No paired `rocprofv3/` directory exists
  at this script's level, so `extract_calltree.py` naturally produces a CPU+MPI-only tree with no
  kernel integration — exactly "tool 1 without GPU support," as the user specified.
- **`scripts/profile_hotspots.sh`** (tool 3, combined): same pattern, added right after its
  existing `extract_hotspots.py` call, pointed at `"$OUTPUT_DIR"` (the parent directory containing
  both `$CPU_DIR`/`$GPU_DIR` as `rocprof-sys/`/`rocprofv3/` subdirs) — matches
  `extract_calltree.py`'s own auto-detection, so kernel integration applies automatically.
  Note: this script already calls `profile_CPU_hotspots.sh --no-summary` internally (confirmed at
  the `"$CPU_LAUNCHER" --no-summary ... -o "$CPU_DIR"` line) — so tool 1's own new auto-calltree
  step is correctly suppressed during a tool-3 run, avoiding a stray, prematurely-CPU-only
  `calltree.txt` inside the `rocprof-sys/` subdirectory before the combined one is written.
- **`scripts/instrument_hotspots.sh`** (tool 4, trace mode): **no code change needed for the
  scan directory** — its scan step's launcher is confirmed to be `profile_hotspots.sh` itself
  (`HOTSPOTS_LAUNCHER="$SCRIPT_DIR/profile_hotspots.sh"`), so once tool 3 gains the auto-calltree
  step, tool 4's scan directory inherits `calltree.txt` automatically, for free, the same way it
  already inherits `hotspots.txt` today. (If the user passes `--report` to skip the scan step
  entirely, no `calltree.txt` is generated either — consistent with today's `hotspots.txt`
  behavior in that same case.) Only addition here: forward new `--max-depth`/`--show-gpu-api`
  flags through to the scan step's `PROFILE_CMD`, mirroring how `SELECTION_ARGS`/`UNFILTERED_ARGS`
  are already forwarded there.

New pass-through flags added to `profile_CPU_hotspots.sh`, `profile_hotspots.sh`, and
`instrument_hotspots.sh`: `--max-depth N` and `--show-gpu-api`, collected into a `CALLTREE_ARGS`
array and forwarded only to the calltree extractor call (the hotspots extractor doesn't understand
them) — same pattern as the existing `SELECTION_ARGS`/`UNFILTERED_ARGS` forwarding.

Auto-calltree reuses the **same** `RUN_SUMMARY`/`--no-summary` gate as the existing hotspots
auto-summary, rather than adding a separate `--no-calltree` toggle — one "skip auto-analysis"
switch, not two, matching how `--no-summary` already reads ("skip auto-running the hotspots
extractor afterwards" → becomes "...the hotspots and calltree extractors afterwards").

### Docs

- Add a "Calltree" section to `README.md`, matching the existing tools' style, explicitly stating
  the filtering equivalence to tool 3's "CPU compute hotspots" bucket and the kernel-placement
  honesty caveats (structural, not per-dispatch).
- No separate new doc file for the deferred-trace-parsing note — this plan document itself is the
  record (saved verbatim to `docs/plans/` per this repo's convention, plans are kept): real
  Perfetto-trace-based calltree (exact instrumented data, genuine call ordering, real HIP API call
  sites) is deferred future work requiring a new non-stdlib dependency (see point 3 in Context) —
  tracked here, not silently dropped.

### Tests

New `postprocess/tests/test_extract_calltree.py`, reusing existing fixtures where possible:
- Tree construction/rendering: `mpi_2rank` (basic multi-rank tree, MPI calls visible by default).
- Multi-root detection: `gpu_spawned_thread` (is_thread_root-flagged case) and `multi_metric_rank`
  (DEPTH-resets-to-0 case) — both must produce a correctly-separated second tree.
- GPU-API pruning: `gpu_api_nested_chain` (whole 3-level chain removed by default, present with
  `--show-gpu-api`).
- Kernel integration, single anchor: `gpu_sync_wait` (has a real `hipLaunchKernel` row) paired
  with `rocprofv3_single_rank`.
- Kernel integration, multiple anchors: a new small fixture with two distinct `hipLaunchKernel`-
  family call sites at different launch counts (e.g. 300 vs 200 calls), asserting the kernel
  aggregate splits proportionally (60/40) between them, not duplicated in full at each.
- Kernel integration, no-anchor fallback: a CPU-only fixture with no `hipLaunchKernel`-family row
  paired with a GPU fixture — likely needs one small new fixture combo (mirroring
  `pop_combined_2rank`'s existing pairing style from the POP metrics tool).
- `--max-depth` truncation: assert deeper nodes replaced by a stated hidden-count line, not
  silently dropped.

## Verification

- `python3 -m unittest discover -s postprocess/tests` — full suite green.
- Per the user's request, manually run `extract_calltree.py` standalone against the real,
  already-generated `Heat_Convection_Solver` directories (not just fixtures) once it's built:
  all three existing tool-3 outputs (`profile_hotspots-output-2026-08-04_08.58.53/`,
  `profile_hotspots-output-2026-08-10_12.36.26/`, `profile_hotspots-output-2026-08-10_12.47.32/`)
  and the one tool-4 scan directory (`instrument_hotspots-scan-2026-08-10_12.55.53/`) — confirming
  MPI calls visible, GPU-API noise hidden by default, kernel integration anchoring cleanly or
  falling back honestly depending on the directory, and `--max-depth` visibly truncating with a
  stated count. Save the resulting `calltree.txt` for each and deliver them to the user for
  review (same as the POP metrics tool's real-data validation earlier this session), rather than
  just reporting pass/fail.
- `bash -n` on all three modified scripts (`profile_CPU_hotspots.sh`, `profile_hotspots.sh`,
  `instrument_hotspots.sh`) for syntax sanity, plus their `--dry-run` mode (already supported by
  at least `profile_hotspots.sh`) to confirm the new calltree invocation is constructed correctly
  without needing real `rocprof-sys`/`rocprofv3` on this CPU-only dev machine — real end-to-end
  confirmation that `calltree.txt` actually lands next to `hotspots.txt` after a real run is left
  to the user on the HPC system, same as every prior script change in this project.
