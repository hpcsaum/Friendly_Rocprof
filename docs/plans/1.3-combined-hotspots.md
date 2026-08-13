# Third tool: combined CPU+GPU hotspots (double-counting-safe)

## Context

Tools 1 and 2 each answer half the "where's the time going" question: [postprocess/rocprof_sys_hotspots.py](../../Documents/Porting_Adventure/Friendly_Rocprof/postprocess/rocprof_sys_hotspots.py) ranks CPU-side hotspots from a `rocprof-sys` run, [postprocess/rocprofv3_hotspots.py](../../Documents/Porting_Adventure/Friendly_Rocprof/postprocess/rocprofv3_hotspots.py) ranks GPU kernel hotspots from a separate `rocprofv3` run. This third tool runs both profilers against the same command (one after the other — each tool wraps a process, they can't run concurrently against one binary) and merges their two ranked lists into **one** hotspots table, so the user gets a single prioritized view regardless of whether the top offender happens to be a CPU function or a GPU kernel.

Two design problems were worked through with the user in conversation before this plan, and the design below is the direct result:

**Problem 1 — naive "average the two totals" doesn't work.** The user's first idea was: average the two runs' total times, rescale every entry as if its run's total were that average, and use the rescaled numbers as the combined data. This is a mathematical no-op — rescaling every entry in a run by the same constant `k = avg/T_run`, then dividing by `avg`, cancels the `k` out exactly: `(t_i × avg/T_A) / avg = t_i/T_A`, i.e. every entry's `%total` ends up identical to what it already was before "combining." Two runs with wildly different absolute durations but the same 90%-of-their-own-total hotspot would both show 90% — indistinguishable, even though one may represent 24× more absolute time. The fix that actually changes the outcome (and is simpler to implement): use **`T_A + T_B`** — the plain sum of the two runs' totals — as one shared denominator for every entry in both tables, with **no per-entry rescaling at all**. This makes `%total` reflect genuine relative magnitude across runs.

**Problem 2 — `T_A + T_B` alone double-counts GPU time.** `rocprof-sys`'s CPU-side `wall_clock` total is *inclusive* — `main()`'s total covers everything nested under it, including time blocked inside `hipStreamSynchronize`/`hipDeviceSynchronize`/a synchronous `hipMemcpy`, etc. That blocked-waiting-for-the-GPU time is the same physical interval `rocprofv3`'s kernel `TotalDurationNs` already counts from the device side. Summing the two raw totals counts that overlap twice. The fix, using machinery [postprocess/rocprof_sys_hotspots.py](../../Documents/Porting_Adventure/Friendly_Rocprof/postprocess/rocprof_sys_hotspots.py) already has: that tool already buckets `hip*`/`hsa*`/`roctx*`/`kfd*`/etc. entries away from its "CPU compute" bucket into a separate "GPU API / launch overhead" bucket (`GPU_API_PREFIXES` in that file). Subtract that bucket's summed time from the CPU run's raw total *before* adding it to the GPU run's total: `combined_total = (T_A_raw − sum(GPU_API_bucket)) + T_B`. This removes the double-counted synchronization-wait time. **Accepted approximation** (confirmed fine with the user, who noted non-blocking HIP calls are out of scope for a beginner tool): this also strips out genuinely-tiny non-blocking overhead like `hipLaunchKernel`'s few µs, since there's no reliable way to tell blocking from non-blocking HIP/HSA calls by name alone — negligible next to the sync-wait time it correctly removes.

**Explicit non-requirement (stated by the user):** the extractor takes both output directories as independent inputs and does **not** cross-check that they came from the same executable/test case/run — that's the user's responsibility (GIGO). Don't add any such validation.

## Design

### 1. `postprocess/rocprof_combined_hotspots.py` — standalone-invocable extractor that imports its two siblings

Confirmed ground truth from the existing code (read directly, not from memory):
- `rocprof_sys_hotspots.aggregate(output_dir)` → `(cpu_entries, gpu_entries, scanned_files, total_runtime)`; each entry is `{"label", "count", "sum", "pct_self", "pct_total"}`. The `gpu_entries` list here *is* the "GPU API / launch overhead" bucket this plan subtracts.
- `rocprofv3_hotspots.aggregate(output_dir)` → `(entries, scanned_files, total_ns)`; each entry is `{"label", "count", "sum" (seconds), "avg_us", "pct_total"}`.
- Both modules' `select_entries(entries, total, top=None, threshold=None, show_all=False)` are generic over any list of dicts with `"sum"`/`"pct_total"` keys and share identical selection semantics (just a differently-named second parameter) — the combined tool calls one of them directly (e.g. `cpu_tool.select_entries`) on its merged list rather than writing a third copy.
- Both modules' `gather_run_info(output_dir, scanned_files)` → `{"executable", "run_datetime", "total_runtime", "num_ranks"}`, already safe to call independently per side.

Since this script lives in `postprocess/` next to its two siblings and is always invoked directly (`python3 postprocess/rocprof_combined_hotspots.py ...`, including from the launcher below), Python automatically puts the script's own directory at the front of `sys.path` — so it can just do `import rocprof_sys_hotspots as cpu_tool` / `import rocprofv3_hotspots as gpu_tool` with no path manipulation, no relative-import machinery.

Usage: `rocprof_combined_hotspots.py <rocprof-sys-output-dir> <rocprofv3-output-dir> [-o hotspots.txt] [-n N | --threshold PCT | --all]` — two required positional directories, order matters (CPU dir first, GPU dir second), documented in `--help`.

Logic (`build_combined_view(rocprof_sys_dir, rocprofv3_dir)`):
1. `cpu_entries, cpu_gpu_api_entries, cpu_scanned, cpu_total_raw = cpu_tool.aggregate(rocprof_sys_dir)`; `gpu_entries, gpu_scanned, gpu_total_ns = gpu_tool.aggregate(rocprofv3_dir)`.
2. Fail fast (clear, side-specific message) if either `cpu_scanned` or `gpu_scanned` is empty — no cross-validation between the two sides, but each side still needs *some* data to combine.
3. `gpu_api_overhead_sec = sum(e["sum"] for e in cpu_gpu_api_entries)`; `cpu_pure_total_sec = max(0.0, cpu_total_raw - gpu_api_overhead_sec)` (the `max(0.0, ...)` is a defensive clamp only, not expected to trigger); `gpu_total_sec = gpu_total_ns / 1e9`; `combined_total_sec = cpu_pure_total_sec + gpu_total_sec`.
4. Build one merged list: every `cpu_entries` item tagged `"domain": "CPU"`, every `gpu_entries` item tagged `"domain": "GPU"` — **the CPU run's own GPU-API/overhead bucket is deliberately excluded from the merged list** (it's now represented, once, by the real GPU kernel data instead). Each merged item's `"pct_total"` is recomputed against `combined_total_sec` (not copied from either source dict, which are relative to each side's own total).
5. Return the merged list plus a small info dict carrying all the intermediate numbers (`cpu_total_raw`, `gpu_api_overhead_sec`, `cpu_pure_total_sec`, `gpu_total_sec`, `combined_total_sec`) — the report prints these transparently so a curious user can audit the arithmetic themselves.

`write_report(rocprof_sys_dir, rocprofv3_dir, dest_path, top=None, threshold=None, show_all=False)` produces **four tables**, in this order — the fused view for a quick coherent answer, then the full per-domain breakdown (each using that domain's *own* standalone `%total` basis, i.e. "as if generated by the previous two tools") so nothing is hidden and anyone auditing the fused numbers can see exactly what fed into them:

1. **Combined hotspots (fused CPU+GPU ranking)** — the merged list from `build_combined_view`, `%total` against `combined_total_sec`. New `format_table_fused(entries)` in this module (columns `#, total(s), %total, dom, calls, name`, `dom` = `CPU`/`GPU`) — the only genuinely new formatting code this module needs.
2. **CPU compute hotspots** — `cpu_entries` exactly as tool 1 buckets them, selected via `cpu_tool.select_entries(cpu_entries, cpu_total_raw, top, threshold, show_all)` and rendered via `cpu_tool.format_table(...)` directly (no reimplementation — same shape tool 1 already produces, `%total` against that run's own raw total).
3. **GPU kernel hotspots** — `gpu_entries` exactly as tool 2 produces them, selected via `gpu_tool.select_entries(gpu_entries, gpu_total_ns, top, threshold, show_all)` and rendered via `gpu_tool.format_table(...)` directly, `%total` against that run's own total.
4. **GPU API / launch overhead** — `cpu_gpu_api_entries` (tool 1's second bucket, the one subtracted out of the fused total in step 3 above), selected via `cpu_tool.select_entries(cpu_gpu_api_entries, cpu_total_raw, top, threshold, show_all)` and rendered via `cpu_tool.format_table(...)` — this is exactly tool 1's own "GPU API / launch overhead" section, included so the reader can see precisely what got subtracted.

Each section header states in words what its `%total` is relative to (the fused pool vs. that run's own total) — the three different denominators must never be visually ambiguous. Header block (generated timestamp; both source directories labeled "(CPU run)"/"(GPU run)"; both sides' `gather_run_info` best-effort fields, labeled per side, never merged/validated against each other; the five transparency numbers: `cpu_total_raw`, `gpu_api_overhead_sec`, `cpu_pure_total_sec`, `gpu_total_sec`, `combined_total_sec`; files scanned from both sides) precedes all four tables. A short footer note on the fused table explains in one or two sentences what was subtracted and why, pointing at table 4 as the detail.

### 2. `scripts/rocprof_combined_profile.sh` — launcher that reuses the two existing launchers

Rather than re-implementing dependency checks / env vars / flag construction a third time, this script just calls the two existing launchers sequentially with `--no-summary` (so they don't each try to run their own single-domain summary), then runs the combined extractor once:

```bash
CPU_DIR="$OUTPUT_DIR/rocprof-sys"
GPU_DIR="$OUTPUT_DIR/rocprofv3"
"$SCRIPT_DIR/rocprof_sys_profile.sh"  --no-summary -o "$CPU_DIR" -- "$@"
"$SCRIPT_DIR/rocprofv3_profile.sh"    --no-summary -o "$GPU_DIR" -- "$@"
# then, rank-0-gated as in both existing launchers:
python3 "$EXTRACTOR" "$CPU_DIR" "$GPU_DIR" "${SELECTION_ARGS[@]}"
```

Flags: `-o/--output-dir DIR` (default `rocprof-combined-hotspots-output`; split into the two subdirectories above), the same `--top`/`--threshold`/`--all`/`--no-summary`/`--dry-run`/`-h` set as the other two launchers (same "last one wins" shell-level behavior, not argparse-mutually-exclusive, consistent with the existing scripts). `--dry-run` forwards `--dry-run --no-summary` down to *both* sub-launchers (reusing their own tested dry-run output rather than reimplementing it) and then just prints the would-be combined-extractor command line.

Because this is `bash -e` calling two sub-scripts sequentially with no `set +e` around either call, a failure in the CPU launcher aborts before the GPU launcher ever runs, and the script's own exit code is whichever sub-launcher failed — deliberate: unlike the single-tool launchers, there's no single "the app ran once" exit code to forward here (the app runs twice, once under each profiler), so falling through to whichever step failed is the honest behavior, not something to special-case.

Same rank-0 detection chain (`OMPI_COMM_WORLD_RANK`/`PMI_RANK`/`SLURM_PROCID`) and python3-missing-warns-but-doesn't-fail behavior as the existing two launchers, applied once here (not per sub-launcher, since both were called with `--no-summary`).

### Files

- `postprocess/rocprof_combined_hotspots.py` (new)
- `scripts/rocprof_combined_profile.sh` (new)
- `postprocess/tests/test_rocprof_combined_hotspots.py` (new) — **reuses existing fixtures**, no new fixture files needed: pairs `postprocess/tests/fixtures/mpi_2rank` (CPU) with `postprocess/tests/fixtures/rocprofv3_mpi_2rank` (GPU) as the primary case (the CPU fixture already has a non-zero `hipMemcpy` entry in its GPU-API bucket, so the subtraction actually has something to subtract), and `single_rank`/`rocprofv3_single_rank` as a second pairing. Explicitly tests: the fused table's `%total` differs from each side's own standalone `%total` (proving real recombination, not the rejected no-op); the subtraction arithmetic (`combined_total_sec` equals the documented formula, computed from the fixtures' known raw numbers via actual Python expressions in the assertions, not hand-typed decimals); all **four tables** are present in the written report, in order, each with its own correctly-labeled `%total` basis, and table 2/3/4's numbers match calling `cpu_tool`/`gpu_tool`'s own `select_entries`/`format_table` directly on the same fixtures (i.e. identical to what tools 1/2 would print standalone); a deliberately-mismatched pairing (e.g. `single_rank` + `rocprofv3_mpi_2rank`) still combines without error, proving no cross-validation happens; each side missing/empty raises a clear, side-specific `SystemExit`.
- `README.md` — add the third tool alongside the other two.
- `docs/plans/03-combined-hotspots.md` — this approved plan, written verbatim per `CLAUDE.md`'s workflow.
- `docs/DEVELOPMENT_HISTORY.md` (+ regenerated `.docx`) — updated before the commit, per `CLAUDE.md`, including the double-counting rationale from this conversation.

## Verification

- `bash -n scripts/rocprof_combined_profile.sh` (+ `shellcheck` if available, same caveat as the other two if not).
- `python3 -m unittest postprocess/tests/test_rocprof_combined_hotspots.py -v` (and the full existing suite, to confirm nothing in the two sibling modules was disturbed by this read-only import).
- Manual verification with stubbed `rocprof-sys-sample`/`rocprofv3` on `PATH` (same technique as tools 1 and 2): `--dry-run` shows both sub-launchers' own dry-run output plus the combined-extractor command; a full run with both stubs copying real fixture data through produces a sane merged `hotspots.txt`; removing the CPU stub confirms the script aborts before ever invoking the GPU stub; simulated `OMPI_COMM_WORLD_RANK=1` confirms the combined summary step (not each sub-launcher's, since those got `--no-summary`) is skipped.
- Run the extractor directly against an existing fixture pairing and inspect the generated `hotspots.txt`, same as was done for tools 1 and 2.
