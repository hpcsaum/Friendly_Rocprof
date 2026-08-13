# Reclassify runtime-spawned thread roots via call-tree ancestry (calltree-ready)

## Context

Real HPC data (`Heat_Convection_Solver/profile_hotspots-output-2026-08-04_08.58.53`) showed
`start_thread` at ~200% of the CPU run's own total in tables 1/2 -- the single biggest
remaining distortion after the rank-counting and GPU-API-overhead fixes. Investigation of
the raw `wall_clock-0.txt` file found the exact mechanism:

```
| |0>>> hipRuntimeGetVersion ...
| |0>>>   |_pthread_create                  (2 calls)
| |1>>>     |_start_thread                   948.42s, 100% self   <- lives the whole run
| |2>>>     |_start_thread                     0.00065s, 100% self
...
| |0>>> hipStreamCreate ...
| |0>>>   |_pthread_create                  (1 call)
| |3>>>     |_start_thread                   942.21s, 100% self   <- lives the whole run
```

`main` (thread 0, ~948.4s) is the only real application thread. Threads 1 and 3 are
background threads the **HIP runtime itself** spawns (as a side effect of
`hipRuntimeGetVersion`/`hipStreamCreate` -- almost certainly async-completion/event-polling
threads). rocprof-sys's `pthread_create` interception faithfully instruments their entry
point, but since nothing inside them is explicitly instrumented, their entire lifetime (nearly
the whole run) gets dumped into the generic label `start_thread` at 100% self. Two such
full-run-length ghost threads per rank, on top of `main`'s own span, is exactly why it sums to
~200% of one rank's own wall-clock span.

These same two threads are *also* what the sampling data (already fixed) attributes to
`rocr::core::BusyWaitSignal::WaitAcquire`/`rocr::os::ThreadTrampoline`/
`rocr::core::Runtime::AsyncEventsLoop` -- `start_thread` is a second, coarser accounting of
time that's already visible elsewhere, just under a label that can't be told apart from a
genuine application worker thread by name alone (unlike `rocr::`, `start_thread` is a generic
libc/pthread entry symbol -- any app spawning its own real compute thread would use the exact
same label). Blanket-excluding `start_thread` by name would violate this project's own
no-target-specific-assumptions principle (`CLAUDE.md`).

**Decision (per user)**: fix this structurally, by ancestry, not by label -- a thread-root
row (a row whose thread differs from its parent's thread) is reclassified as GPU-API overhead
only if walking up its own call-tree ancestors finds a GPU-API-classified call (e.g.
`hipRuntimeGetVersion`, `hipStreamCreate`). A thread spawned directly by real application code
keeps its current classification untouched. **This also needs to be built as a reusable
building block for a future real call-tree view**, not a one-off hack -- `parse_table_file()`
currently reads `DEPTH` but discards it, and `aggregate()` immediately flattens everything into
one global per-label dict, so there is no parent/child structure anywhere in the code today.
The ancestry-reconstruction piece this fix needs *is* the first real step toward that future
feature (reconstructing the tree from `DEPTH` + row order), so it's being built as a standalone,
independently-testable utility rather than inlined as a private helper.

Confirmed this only needs to touch `wall_clock`'s data: sampling files (`sampling_wall_clock`)
show `start_thread` too, but there with `% SELF = 0.0` (sampling resolves the *actual* function
each thread was running, already correctly landing under `rocr::`/`hsa::`/etc. labels via the
existing label-based classification) -- so it already contributes ~0 self-time and needs no
special handling. Applying the same ancestry logic uniformly to every scanned file (not
special-casing by filename) is simplest and harmless for sampling files (no parent found there
since sampling's per-thread view starts at depth 0 with no spawn context, so it just falls back
to today's label-based classification).

## Design

All changes are in `postprocess/extract_CPU_hotspots.py`.

### 1. `parse_table_file()`: keep what's already being parsed but currently discarded

Add two keys to each returned row: `"depth": int(depth)` (the `DEPTH` column, parsed today
only to validate column count) and `"thread_id": str` (the last `|`-delimited segment of the
raw prefix before `>>>`, extracted via a new small helper `thread_id_from_raw_label(raw_label)`
-- e.g. `"|1>>>foo"` -> `"1"`, `"00|00>>>foo"` -> `"00"`). No existing field changes meaning;
this is purely additive, so no existing caller/test relying on `label`/`count`/`sum`/`self_sum`
is affected.

### 2. New function: `attach_ancestry(rows)`

A standalone, general-purpose utility -- reconstructs parent/child structure from a list of
rows in original file order (which is already a call-tree pre-order walk) using a depth-stack:

```python
def attach_ancestry(rows):
    """Reconstruct each row's parent in the call tree from DEPTH + file order (a
    depth-stack walk: a row's parent is the most recent prior row at depth-1).
    Mutates rows in place, adding "parent" (a reference to the parent row dict,
    or None at the root) and "is_thread_root" (True when this row's thread_id
    differs from its parent's -- i.e. this row is where a NEW OS thread's own
    subtree begins within this file's listing, immediately after whatever
    call spawned it).

    General-purpose: this is the reusable first step for a future real call-tree
    view (walk "parent" links to render nesting/indentation), not specific to
    any one classification decision -- classify_gpu() below is just its first
    consumer.
    """
    stack = []
    for row in rows:
        while stack and stack[-1]["depth"] >= row["depth"]:
            stack.pop()
        parent = stack[-1] if stack else None
        row["parent"] = parent
        row["is_thread_root"] = parent is not None and parent["thread_id"] != row["thread_id"]
        stack.append(row)
    return rows
```

Verified against the real data's exact shape: `pthread_create` (thread 0, depth 1) is the
stack top when `start_thread` (thread 1, depth 2) is reached -> parent found, thread differs ->
`is_thread_root=True`. `main`/`hipRuntimeGetVersion` (thread 0, depth 0, first rows) get
`parent=None` -> `is_thread_root=False`, correctly not flagged.

### 3. New function: `classify_gpu(row, path)`

Replaces the current inline `is_gpu_entry(row["label"], path)` call inside `scan_ranks()`'s
per-file loop:

```python
def classify_gpu(row, path):
    if is_gpu_entry(row["label"], path):
        return True
    if row.get("is_thread_root"):
        ancestor = row["parent"]
        while ancestor is not None:
            if is_gpu_entry(ancestor["label"], path):
                return True
            ancestor = ancestor["parent"]
    return False
```

Own-label classification (existing behavior, e.g. `rocr::`/`hip::`) always wins outright. Only
when a row is a **thread root** does ancestry get consulted at all -- a normal (non-thread-root)
row's classification is completely unchanged from today. This is the key safety property: a
real application's own compute functions are never reclassified by what their *thread's*
ancestry looks like, only a thread-root node itself can be.

### 4. Wire into `scan_ranks()`

Right after `rows = parse_table_file(path)`, call `attach_ancestry(rows)`. Change the per-row
tagging line from `row = dict(row, gpu=is_gpu_entry(row["label"], path))` to
`row = dict(row, gpu=classify_gpu(row, path))`. After classification, only
`label`/`count`/`sum`/`self_sum`/`gpu` are needed downstream (the `wall_clock`/`sampling`
buckets, and everything after) -- strip `depth`/`thread_id`/`parent`/`is_thread_root` when
building the emitted row dict, so `aggregate()`/`aggregate_per_rank()`'s existing data contract
(and every existing test asserting on row dict shape) is unaffected; the ancestry bookkeeping
is scan_ranks()'s own internal implementation detail.

## Tests

`postprocess/tests/test_extract_CPU_hotspots.py`:
- `AttachAncestryTests`: pure unit tests against small synthetic row lists (no fixture files
  needed) -- verify parent linking across a depth-stack sequence, `is_thread_root` correctly
  True only when thread_id changes relative to the found parent, and a depth-0 root gets
  `parent=None`/`is_thread_root=False`.
- `ClassifyGpuTests`: unit tests on synthetic rows -- a thread-root row whose ancestor chain
  includes a `hip`-prefixed label classifies as GPU even though its own label
  (`start_thread`) doesn't match `GPU_API_PREFIXES`; a thread-root row whose ancestor chain has
  no GPU-classified entry stays CPU; a non-thread-root row's classification is untouched by
  ancestry regardless of its ancestors.
- New fixture `postprocess/tests/fixtures/gpu_spawned_thread/wall_clock-4001.txt` reproducing
  both the positive and negative case in one file, mirroring the real data's exact shape:
  - Thread 0: `main` (depth 0), `hipRuntimeGetVersion` (depth 0), `compute_stencil` (depth 0),
    `|_pthread_create` (depth 1, under `hipRuntimeGetVersion`) and a second
    `|_pthread_create` (depth 1, under `compute_stencil`).
  - Thread 1 (spawned from the `hipRuntimeGetVersion`-side `pthread_create`): `start_thread`
    (depth 2), large sum, 100% self -- **must** end up GPU-classified.
  - Thread 2 (spawned from the `compute_stencil`-side `pthread_create`, i.e. real application
    code): `start_thread` (depth 2), large sum, 100% self -- **must** stay CPU-classified. This
    is the critical negative case proving the fix doesn't blanket-hide real application threads.
  Integration test via `aggregate()` on this fixture: thread 1's `start_thread` contribution
  lands in `gpu_entries`, thread 2's lands in `cpu_entries`, even though both rows share the
  exact same label.
- Full regression run of the existing suite (163 tests) to confirm no existing fixture's
  classification changes (none currently contain a `pthread_create`/multi-thread shape, so this
  should be a pure addition).

## Verification

- `python3 -m unittest discover postprocess/tests` -- full suite including new tests.
- Re-run `postprocess/extract_hotspots.py` against the real
  `Heat_Convection_Solver/profile_hotspots-output-2026-08-04_08.58.53` fixture (write to a new
  file, keep the prior verification files on disk) and confirm: `start_thread` no longer appears
  in tables 1/2's top entries (or, if it still has some residual self-time from files without a
  reconstructible spawn ancestor, that residual is proportionate, not ~200%); it now appears in
  table 4 instead; the combined-pool arithmetic's `CPU run raw total` changes accordingly (it's
  the same `root_sum`-based total_runtime computation from the earlier fix, unaffected by this
  reclassification -- only which *bucket* start_thread's self_sum lands in changes, not the
  root_sum denominator itself).
- Update `docs/DEVELOPMENT_HISTORY.md`/`.docx` with a narrative section covering: the
  `start_thread`-at-200% investigation, the HIP-runtime-spawned-background-thread root cause,
  why label-based exclusion was rejected (`CLAUDE.md`'s no-target-specific-assumptions
  principle), the ancestry-based structural fix, and its explicit framing as reusable
  groundwork for a future call-tree view feature.
- Write this plan verbatim to `docs/plans/09-calltree-ancestry-thread-classification.md` once
  approved.
- Hold the commit for user approval, same as the two prior fixes this session -- all three
  (rank/sampling-file dedup, GPU-API sync-wait narrowing, this ancestry fix) are still
  uncommitted and share one pending `DEVELOPMENT_HISTORY.md` update.
