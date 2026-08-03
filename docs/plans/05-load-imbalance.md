# Load-imbalance tables for tools 1, 2, 3

## Context

The user ran `profile_CPU_hotspots.sh` (tool 1) on the real HPC system and confirmed it already captures `MPI_*` function calls even though the launcher never sets `ROCPROFSYS_USE_MPIP` explicitly — that env var defaults to `true` in rocprof-sys itself, so this confirms the target system's rocprof-sys build has full MPI support compiled in. Combined with an earlier, independently-confirmed fact — rocprof-sys's raw label format is `|MM|NN>>>label` (MM = rank, NN = thread), and every `wall_clock-<PID>.txt` file is already one-file-per-process — this means **all the data needed for a per-rank load-imbalance analysis already exists in tools 1-3's normal sampling output today**. No rebuild, no instrumentation (tool 4's mechanism) is needed at all; this is a pure post-processing addition.

The user wants: a new table, in tool 1 (CPU functions), tool 2 (GPU kernels), and tool 3 (both — reusing tool 1 and 2's new capabilities directly), placed **after** each report's existing hotspots table(s). Per function/kernel, sorted by `std_dev`, with columns `avg_time`, `std_dev`, `min_time`, `max_time`. Per the user's explicit choice, the table's contents are an **independent ranking by std_dev** (using the existing `--top`/`--threshold`/`--all` flags, applied to std_dev instead of to total time) — not simply the same functions already shown in the hotspots table re-sorted, since a function that's small in total time but wildly imbalanced across ranks would otherwise never be visible.

## Design decisions worth flagging explicitly

- **Rank = file, not the raw `MM` digit.** Grouping by which output file a row came from (already the same assumption `guess_num_ranks()` uses via `PID_SUFFIX_RE`) is simpler and more robust than re-parsing the `MM` prefix out of each raw label, and requires zero changes to `clean_label()`/the existing text-table parsing. One scanned file = one rank's contribution.
- **A rank missing a function is scored as 0.0 for that rank, not skipped.** If a function only runs on some ranks (e.g. rank-dependent branching), that's real, extreme imbalance and should show up — silently excluding ranks that never called it would hide exactly the thing this table exists to surface.
- **`--threshold PCT` is reinterpreted for this table only**, since "% of total runtime" (its meaning for the hotspots tables) has no natural equivalent for a std_dev ranking: here it means *coefficient of variation* — only show functions where `std_dev / avg >= PCT%`. Same flag, same "a percentage cutoff" mental model, different metric underneath. `--top`/`--all` keep their existing meaning (top N by std_dev, or no truncation).
- **Fewer than 2 ranks/files scanned → skip the table entirely**, printing one line explaining why, instead of a table comparing one rank to itself.
- No new CLI flags anywhere — the existing `-n/--top`/`--threshold`/`--all` already forwarded by all three launchers apply automatically to the new table too.

## Implementation

### `postprocess/extract_CPU_hotspots.py`
- `import statistics` (stdlib).
- New `aggregate_per_rank(output_dir)`: same file-scanning loop as `aggregate()` (same `glob`, same `parse_table_file`, same `is_gpu_entry` exclusion of the GPU-API bucket), but instead of merging every file into one global total, returns `(per_file_totals, scanned_files)` where `per_file_totals` is a list of `{label: sum}` dicts, one per scanned file/rank.
- New `compute_load_imbalance(per_file_totals, top=None, threshold=None, show_all=False)`: builds the full set of labels seen in any rank, computes `avg = statistics.mean(values)`, `std_dev = statistics.pstdev(values)` (population std-dev — we have the *entire* set of ranks, not a sample), `min`/`max` over `values = [ft.get(label, 0.0) for ft in per_file_totals]`; applies the top/CV%-threshold/all selection described above; returns `(selected_sorted_by_std_dev_desc, description)`, mirroring `select_entries()`'s existing return shape so `write_report()` can reuse the same `f"... -- showing {desc}"` header pattern.
- New `format_table_load_imbalance(entries)`: `#, avg(s), std_dev, min(s), max(s), function` columns, same visual style as `format_table()`.
- `write_report()`: after the existing two tables, call `aggregate_per_rank(output_dir)`; if `len(scanned_files) < 2`, append the skip note; else call `compute_load_imbalance(...)` with the same `top`/`threshold`/`show_all` already passed into `write_report()`, then `format_table_load_imbalance(...)`.

### `postprocess/extract_GPU_hotspots.py`
- Same three functions, duplicated rather than imported (matching this module's existing stand-alone-by-design relationship to `extract_CPU_hotspots.py`). `aggregate_per_rank()` reuses `parse_kernel_stats_csv()`; no extra per-file summing needed since rocprofv3 already aggregates duplicate kernel names within one file — each file directly gives one `{kernel_name: total_ns/1e9}` dict.
- `write_report()`: same placement, after its existing single hotspots table.

### `postprocess/extract_hotspots.py` (tool 3)
- `write_report()`: add table 5 (CPU load imbalance — `cpu_tool.aggregate_per_rank(rocprof_sys_dir)` → `cpu_tool.compute_load_imbalance(...)` → `cpu_tool.format_table_load_imbalance(...)`) and table 6 (GPU load imbalance — same via `gpu_tool`, against `rocprofv3_dir`), placed after the existing four tables, in that order. Exactly the "reuse tool 1 and 2's new capabilities" the user described — no new statistics logic in this module, just wiring.

### Launcher scripts
No new flags. Update each of `scripts/profile_CPU_hotspots.sh`, `scripts/profile_GPU_hotspots.sh`, `scripts/profile_hotspots.sh`'s `-h`/`--help` text: one sentence noting the new load-imbalance table, and a note on `--threshold`'s dual meaning (% of runtime for the hotspots table(s), coefficient-of-variation % for the load-imbalance table).

### Tests
No new fixtures needed — `mpi_2rank`/`rocprofv3_mpi_2rank` (2 ranks, already has per-function values that differ slightly between ranks) exercise the real computation; `single_rank`/`rocprofv3_single_rank` exercise the "<2 ranks → skip" path. New test coverage in `test_extract_CPU_hotspots.py`, `test_extract_GPU_hotspots.py`, `test_extract_hotspots.py`: `aggregate_per_rank` returns one dict per scanned file; `compute_load_imbalance`'s avg/std_dev/min/max match independently-computed `statistics.mean`/`pstdev` on the same known fixture values (not hand-typed decimals); a label constructed to appear on only one of several synthetic ranks scores 0.0 (not omitted) for the others; the CV%-threshold selection and top-N-by-std_dev selection both behave correctly; single-rank fixtures produce the skip note, not a table; the combined report's tables 5 and 6 appear after table 4 in order, and match calling `cpu_tool`/`gpu_tool`'s new functions directly on the same fixtures (same "matches standalone tool" pattern already used for tables 2-4).

## Verification

- `python3 -m unittest` — full suite (93 existing + new tests).
- `bash -n` on all three launchers (help-text-only changes there, but re-check regardless).
- Manual: regenerate a real `hotspots.txt` from the `mpi_2rank`/`rocprofv3_mpi_2rank` fixtures through each of the three `write_report()`s and read the new table(s) — confirm placement/order and that the numbers look sane; re-run against `single_rank`/`rocprofv3_single_rank` and confirm the skip note appears instead of an empty/degenerate table.
