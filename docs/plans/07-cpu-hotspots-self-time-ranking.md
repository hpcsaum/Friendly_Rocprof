# Rank CPU hotspots by real self-time instead of inclusive time

## Context

The user ran tool 1 (`profile_CPU_hotspots.sh`) against `Heat_Convection_Solver` and found its
"CPU compute hotspots" table dominated by functions that do nothing themselves except call the
next thing down (`main`, `start_thread`, `__libc_start_call_main`, `rocprofsys_main`,
`__libc_start_main`, and legitimate-but-uninteresting user wrapper functions like `solver_run`,
`solver_step`, `convection_step`), plus the same MPI operation appearing 3-4 times under
different internal names (`mpi_allreduce_f08ts_`, `MPIR_Allreduce_cdesc`, `PMPI_Allreduce`,
`MPIR_CRAY_Allreduce`, all within noise of each other).

Root cause, confirmed by reading the raw scanned files in
`Heat_Convection_Solver/rocprof-sys-hotspots-output/2026-08-03_09.24/`:
`profile_CPU_hotspots.sh` explicitly sets `ROCPROFSYS_FLAT_PROFILE=1`. That setting -- per
`docs/how-to/understanding-rocprof-sys-output.rst` in AMD's `ROCm/rocprofiler-systems` repo
(checked against the `rocm-7.0.2` tag specifically) -- "removes all call stack hierarchy": every
row's DEPTH is forced to 0 and every occurrence of a symbol anywhere in the call stack is merged
into one row, so `% SELF` (which distinguishes a function's own work from time spent in what it
calls) comes out as `100.0` for literally every row -- there is no way to tell a real hotspot from
a pass-through wrapper in that data.

Confirmed via the same docs: **`ROCPROFSYS_FLAT_PROFILE` defaults to `false`** -- this project's
own launcher is what turns flat mode on, not rocprof-sys itself. With it off, the exact same
`wall_clock-<pid>.txt`/`sampling_wall_clock-<pid>.txt` files get real per-node DEPTH and a
genuinely differentiated `% SELF` (the docs' own worked example shows values like
18.2/0.0/0.1/100.0/1.0 down a real `main -> ... -> conj_grad` call chain) -- no other setting is
needed. Overhead is only qualitatively documented ("flat has less overhead than hierarchical", no
numbers); the memory-blowup caution in the docs is specifically about the separate
`ROCPROFSYS_TIMELINE_PROFILE` setting, not plain hierarchical mode, which stays a bounded,
aggregated call tree (capped by `ROCPROFSYS_MAX_DEPTH`, default 65535).

**Why this is the better fix than a name-based block-list.** With real self-time available,
ranking the CPU table by self-time instead of inclusive time fixes the *entire* pollution problem,
not just the system-function slice: `main`/`start_thread`/`solver_run`/`solver_step`/
`convection_step` all have near-zero self-time (they just call the next thing), so they fall out
of a self-time ranking on their own, without a language-specific or maintained name list. The
MPI-duplicate-entries problem is expected to mostly self-resolve too: of a wrapper chain like
`mpi_allreduce_f08ts_ -> MPIR_Allreduce_cdesc -> PMPI_Allreduce -> (real blocking wait)`, only
the node that actually does the blocking/communication work will show meaningful self-time; the
others drop toward zero and fall out of the top N on their own. This can't be fully verified from
this CPU-only, no-ROCm dev machine -- flagged below as a residual risk to sanity-check on real
HPC output, with a name-based MPI-canonicalization fallback available later if duplicates persist.

**Not touched, and why:** `extract_GPU_hotspots.py` (tool 2's rocprofv3 kernel-stats table) has no
call-tree/self-vs-inclusive concept -- kernels are already atomic leaf events, so it's left alone.
While investigating the "GPU API / launch overhead" bucket's arithmetic (used in tool 3's
combined-pool double-counting fix), I found what looks like a **pre-existing, separate bug**:
`gpu_api_overhead_sec = sum(e["sum"] for e in cpu_gpu_api_entries)` sums *inclusive* time across
every GPU-API-classified row, but several of those rows are themselves nested inside each other
(e.g. in the sample data, `hip::hipStreamCreate` / `hip::ihipStreamCreate` / `hip::Stream::Stream`
all show the *identical* total, i.e. they're one call chain, not three independent calls) -- so
that sum likely over-counts the real host-side GPU-blocking time today, independent of anything in
this plan. Out of scope here (not what was reported, and fixing it needs its own investigation);
flagged via `spawn_task` once this plan lands so it doesn't get lost.

## Design

### 1. Capture side: `scripts/profile_CPU_hotspots.sh`

Change `export ROCPROFSYS_FLAT_PROFILE=1` to `export ROCPROFSYS_FLAT_PROFILE=0` (explicit, same
style as the existing `ROCPROFSYS_TRACE=0`). Update the `--dry-run` env-var preview line and the
help text's brief mention of overhead: note that hierarchical mode (the new default) has somewhat
higher overhead than flat mode per AMD's docs, but no fixed setting is being removed -- users who
need flat mode back for an overhead-sensitive run can still set `ROCPROFSYS_FLAT_PROFILE=1`
themselves in their environment before invoking the script (not adding a new CLI flag for this
unless it turns out to be needed).

### 2. `postprocess/extract_CPU_hotspots.py` -- track and rank by self-time

**`parse_table_file`**: each row already carries `sum` (inclusive) and the raw `% SELF` field;
add `"self_sum": float(total) * float(pct_self) / 100.0` to the row dict (self-time in seconds
for that one call-tree node), and stop keeping the raw `pct_self` on the row itself (it's only
needed to compute `self_sum`; the aggregated, weighted self-% is derived later from
`self_sum/sum`).

**`aggregate(output_dir)`**: when building `totals[label]`, accumulate `self_sum` the same way
`sum`/`count` are already accumulated (`entry["self_sum"] += row["self_sum"]`) -- correct without
double-counting because self-time at different tree nodes never overlaps, even when the same
symbol name appears at multiple call sites (each occurrence is aggregated additively, exactly
like `sum`/`count` already are). Each emitted `item` gets `self_sum` and a recomputed
`pct_self = self_sum/sum*100` (now a real weighted value, not whatever the last file happened to
report). Drop the `pct_total` field from `aggregate()`'s output entirely -- move that
computation into `select_entries` (next), since which total it's a percentage *of* now depends on
which metric is being ranked.

**`aggregate_per_rank(output_dir)`**: same idea -- each file's `file_totals[label]` becomes a
`self_sum` accumulation instead of `sum`, so the CPU load-imbalance table ranks/reports the same
metric as the main hotspots table (keeps the two tables in one report internally consistent).

**`select_entries(entries, total_runtime, top=None, threshold=None, show_all=False, rank_by="self")`**:
add `rank_by` (`"self"` default, `"inclusive"` when `--unfiltered`). Sort key becomes
`e["self_sum"]` or `e["sum"]` accordingly; compute and attach `pct_total` on each candidate entry
from *that* metric before the threshold filter / truncation, so `--threshold PCT` means "at or
above PCT% of runtime by whichever metric is active" -- consistent, single meaning.

**`format_table(entries)`**: add a `self(s)` column (the new primary, sorted metric) ahead of the
existing `total(s)` (kept as useful secondary context -- "how much of this deep call chain's time
this function's own subtree accounts for" is still informative), giving:
`#  self(s)  %total  total(s)  calls  %self  function`. This also declutters the "GPU API /
launch overhead" table for free (it's built from the same `aggregate()`/`format_table()` path),
independently confirming/fixing the nested-HIP-wrapper redundancy I noticed there, without needing
the separate fix flagged above for `gpu_api_overhead_sec`'s own arithmetic.

**`write_report(..., unfiltered=False)`**: pass `rank_by="inclusive" if unfiltered else "self"`
into both `select_entries` calls and `aggregate_per_rank`'s consumer; add one short preamble note
explaining the metric (own-work self-time by default; pass `--unfiltered` for the old
inclusive/cumulative-time view).

### 3. Downstream propagation (same "fix once, inherited everywhere" pattern as this project's
earlier nested-directory fix)

- `extract_hotspots.py` (tool 3): add `--unfiltered`; thread into `cpu_tool.select_entries(...)`
  calls for both the standalone CPU table and (see below) the fused table, and into
  `cpu_tool.aggregate_per_rank`'s imbalance path. In `build_combined_view`, change the fused-table
  construction to use `e["self_sum"]` (not `e["sum"]`) for CPU-domain entries -- keeps the fused
  CPU+GPU ranking on a consistent "own work" basis for both domains (GPU kernel entries are
  already self-equivalent). Leave `gpu_api_overhead_sec`'s computation exactly as-is (still needs
  inclusive time for the double-counting subtraction -- that's the separately-flagged bug, not
  something to fix as a side effect here).
- `select_hotspot_functions.py` (tool 4): add `--unfiltered`; thread into
  `labels_from_output_dir()` -> `cpu_tool.select_entries(..., rank_by=...)`. This makes tool 4
  prefer real, self-contained hotspot functions as instrumentation targets by default -- exactly
  what you want to instrument, better than today's inclusive-time-biased picks.
  **Also fix `labels_from_report()`**: it parses an existing report's table with
  `line.split(maxsplit=5)` assuming 6 whitespace-separated fields ending in the function name --
  the new `self(s)` column makes that 7 fields, so this must become `maxsplit=6` /
  `parts[6]` / `len(parts) < 7`, or every parsed label will be wrong.
- `scripts/profile_hotspots.sh` / `scripts/instrument_hotspots.sh`: add `--unfiltered`, forwarded
  to the extractor/selector calls exactly like `--top`/`--threshold`/`--all` already are (build a
  small `UNFILTERED_ARGS=()` array, same pattern).
- `scripts/profile_GPU_hotspots.sh`: no change.

## Tests

- New fixture(s) with genuine hierarchical data (real DEPTH values, varying `% SELF` per row,
  modeled on the docs' own worked example) alongside a `main`-like wrapper (near-0% self) and a
  couple of MPI-wrapper-chain rows (one dominant self-time node, several near-zero ones) -- to
  actually exercise the new code path, since every *existing* fixture is flat (`% SELF` always
  100), where `self_sum` reduces to `sum` and today's tests should keep passing unchanged
  (spot-check this rather than assuming it).
- Unit tests: `aggregate()`'s `self_sum` accumulation across multiple occurrences of one label at
  different call sites/files; `select_entries(rank_by="self")` vs `rank_by="inclusive")` ordering
  differs on the new fixture; `pct_total` reflects whichever metric was ranked.
- `select_hotspot_functions.py`: a `labels_from_report()` case against a report generated with the
  new column layout, confirming the correct function names are extracted (this is the case most
  likely to silently break -- verify it explicitly, don't just trust the `maxsplit` arithmetic).
- Extend `test_extract_hotspots.py` with one case confirming the fused table's CPU-side numbers
  use self-time.

## Verification

- `python3 -m unittest discover postprocess/tests` (full suite).
- `bash -n` on the three modified scripts.
- Manual: run `extract_CPU_hotspots.py` against the real
  `Heat_Convection_Solver/rocprof-sys-hotspots-output/2026-08-03_09.24` directory (read-only,
  report written to a scratch path) as a sanity check of today's flat data still working
  unchanged (self_sum == sum there), since real hierarchical output can't be captured on this
  dev machine -- call out explicitly in the report/PR that the actual decluttering effect on real
  hierarchical HPC output is unverified locally and should be checked on the next real run.
- `--dry-run` each modified script with `--unfiltered` to confirm forwarding reaches the right
  underlying python call.

## Docs

Update `README.md`'s tool 1 (and note in tool 3/4) sections on the metric change and
`--unfiltered`. Write this plan verbatim to `docs/plans/07-cpu-hotspots-self-time-ranking.md`.
Update `docs/DEVELOPMENT_HISTORY.md`/`.docx` before committing, including: the flat-profile
capture change and its unverified-on-real-hardware caveat, and the separately-flagged
`gpu_api_overhead_sec` nested-double-count bug as a noted follow-up (not fixed here). All of this
lands on the `calltree` branch, not `master`, per the user's request given this changes captured
data, not just report formatting.

---

> **Post-implementation note (2026-08-03):** the plan above didn't spell out that
> `profile_CPU_hotspots.sh` (tool 1) itself also needs a `--unfiltered` CLI flag forwarded to its
> own `extract_CPU_hotspots.py` call (Part 3's bullet list named tools 3/4 and the other two
> scripts explicitly, but not tool 1's own launcher). Added during implementation once manual
> `--dry-run` verification caught the gap -- tool 1 now parses and forwards `--unfiltered` the
> same way as the other three scripts. See `docs/DEVELOPMENT_HISTORY.md` for the corresponding
> entry. Existing test fixtures (`single_rank`, `mpi_2rank`) turned out to already have realistic
> hierarchical `% SELF` variation (not flat), so no new fixture was needed to exercise the
> self-time code path -- existing tests were extended in place instead of adding one.
