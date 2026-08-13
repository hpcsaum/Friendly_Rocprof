# New "calltree" tool: sampling-based, filtered, cross-referenced with wall_clock

> **Update:** the `WALL(s)` wall-clock cross-reference column this plan describes was
> later removed (the user found it mostly empty and not worth the complexity) and both
> `extract_calltree.py` and `extract_calltree_traced.py` were changed to aggregate every
> rank into ONE merged call tree with avg/std_dev/min/max load-balance columns, instead
> of printing one call tree per rank. Kernel attachment was also reworked to match by the
> kernel name's own compiler-embedded owner subroutine first. See
> [DEVELOPMENT_HISTORY.md](../DEVELOPMENT_HISTORY.md) for details.

## Context

The committed `extract_calltree.py` (wall_clock-preferred) turned out to have a real
limitation, found only by inspecting real `Heat_Convection_Solver` data: `wall_clock`'s DEPTH
numbers only reflect *instrumented* call-tree depth, so any non-instrumented frame in between two
instrumented ones is invisible — e.g. GPU kernel launches appeared to hang directly off `main`
(sibling of `MPI_Allreduce`), when the real call chain (confirmed via `sampling_wall_clock`,
which captures every real unwound stack frame regardless of instrumentation) goes through several
real intermediate frames (Cray OpenACC/HIP-offload runtime, real Fortran subroutines). This isn't
a tool bug — it's what the underlying data actually contains — but a `sampling_wall_clock`-based
tree shows the *true* call structure instead.

Two throwaway experiments this session validated the idea and worked out the filtering it needs:
switching the preferred source to `sampling_wall_clock` alone produces an unusably noisy tree
(5027 lines for one rank — GOTCHA/template/dynamic-linker internals dominate, since sampling
unwinds through every real frame including rocprof-sys's own instrumentation machinery). Adding
targeted filtering cut that to 227 lines of genuine, readable application structure. The user
wants this built as the new `calltree` tool (superseding the current one as the default), with
two more capabilities: a wall_clock timing cross-reference column (since sampling's own
timing is statistically approximate, unlike wall_clock's GOTCHA-exact numbers), and the existing
GPU-kernel-attachment feature carried over (expected to work *better* now, since real per-subroutine
launch call sites are visible instead of everything colliding near `main`).

**Decisions from this planning round:**
1. **Keep both tools** — the current one is useful on its own (fast, exact where instrumented,
   no noise-filtering judgment calls needed). Rename it `extract_calltree_traced.py`
   ("traced" = built on rocprof-sys's exact/GOTCHA-instrumented data). The name `calltree`
   (`extract_calltree.py`) is reassigned to the new sampling-based tool.
2. **Wall-clock cross-reference**: one new column, `WALL(s)` — `wall_clock`'s own **self**-time
   for that exact label (not inclusive; matches this codebase's own "self-time by default"
   convention), blank (`-`) when that label has no `wall_clock` entry for that rank. Built from
   `cpu_tool.scan_ranks()`'s raw per-rank merged rows directly (not `aggregate_per_rank()`, which
   excludes GPU-classified rows internally — the cross-reference should still work for a
   GPU-API-ish label if `--show-gpu-api` reveals it).
3. **Four independent filter tiers**, each hideable/revealable separately, confirmed against the
   real data this session:
   - **GPU/offload-runtime noise** (PRUNE — whole subtree hidden): the existing tool's
     `hip`/`hsa`/`roctx`/`kfd`/`rocdecode`/`rocjpeg`/`rocr` prefixes and `.kd`-suffix check,
     broadened from `startswith` to substring/`in` matching (case-insensitive) to also catch
     namespace-qualified C++ symbols the prefix check misses (`rocprofiler::hip::...`), plus new
     substrings observed in real data: `cray_acc`, `hiphardwaredevice`, `present_table`.
     Flag: `--show-gpu-api` (existing name/semantics, now broader).
   - **rocprof-sys/GOTCHA wrapper frames** (SPLICE — the node itself is removed, its children
     reparented to its own parent, since real code sits *inside* these wrappers, not beside them;
     pruning them like GPU noise would delete the whole program, as the first, buggy pass of the
     experiment did): `tim::`, `gotcha`, `rocprofsys`, `__libc_start`, `lookup_hashtable`,
     `lookup.constprop`, `lib_bindings`, `library_gots`. New flag: `--show-rocprofsys-internals`
     (splicing is skipped entirely when set — wrapper nodes render normally instead).
   - **MPI library internals** (COLLAPSE — the first real MPI frame hit while descending is shown,
     its own further internals are not): `MPI_`/`PMPI_`/`MPIR_`/`MPID_`/`MPIDI_` prefixes, or a
     Fortran MPI-binding shim suffix (`_f08_`/`_f08ts_`). New flag: `--show-mpi-internals`.
   - **Compiler-runtime helpers** (PRUNE), confirmed as real residual noise in this session's own
     data: `_f90_`, `__allocate`, `_dealloc`, `posix_memalign`, `_mid_memalign`, `_int_memalign`,
     `_int_malloc`, `_int_free`, `sysmalloc`, `__default_morecore`, `sbrk`, `_fwf`,
     `_xfer_iolist`. Documented explicitly as the least universal tier (observed on Cray's Fortran
     runtime specifically, not necessarily present with other compilers). New flag:
     `--show-compiler-runtime`.
   - Plus one convenience flag, `--show-all-internals`, equivalent to passing all four above at
     once (per the user's "can I have both" answer to per-tier vs. single-flag).
4. **Kernel-launch anchor detection broadened** the same way as GPU-noise detection (substring,
   not prefix, to catch `hip::hipModuleLaunchKernel(...)`-style qualified symbols), plus one new
   recognized entry point observed in real data: `__cray_start_acc_kernel` (Cray's
   compiler-generated OpenACC/offload launch entry point) — documented as compiler-specific,
   expandable later for other compilers' equivalents (e.g. LLVM OpenMP target offload) once
   observed. With real per-subroutine call sites now visible, the existing proportional-split
   logic (built for the old tool's rarer multi-anchor case) is expected to engage far more often
   and far more usefully — this is its ideal use case, not an edge case.

## Approach

### Shared module: new `postprocess/calltree_common.py`

The kernel-anchor-placement math (`find_kernel_anchors`, `attach_kernel_summaries`,
`make_kernel_node`, the launch-label matcher) and small rendering primitives (`get_children`,
`count_all_descendants`, `format_aligned_rows`, `resolve_run_dirs`) move here, imported by
**both** tools. This is a deliberate, documented departure from this codebase's usual
"standalone tools duplicate small constants" convention (`GPU_API_PREFIXES`, `SYNC_WAIT_LABELS`
duplication precedent is for a few constants, not ~150 lines of load-bearing math) — the
proportional-split kernel-attribution logic should have exactly one implementation shared by both
tools, not two that can silently drift apart. `render_node`/`build_children_map`/
`nearest_visible_ancestor` stay **per-tool** (genuinely different visibility rules between them),
parameterized by a small `visibility` object/dict of per-tier show-flags rather than a single
`show_gpu_api` bool, so each tool's own filter tiers plug in cleanly.

**Refactor** `extract_calltree_traced.py` (the rename) to import the same shared primitives
from `calltree_common.py` too, rather than leaving its current inline copies to drift from the
new tool's — matches the "one implementation" rationale above.

### Rename: `postprocess/extract_calltree.py` → `postprocess/extract_calltree_traced.py`

- Update its module docstring/`HELP_BLURB` to clarify it's the fast/exact-where-instrumented
  variant, cross-reference the new tool by name, and note its default output becomes
  `calltree_traced.txt` (was `calltree.txt` — freed up for the new tool).
- Rename `postprocess/tests/test_extract_calltree.py` →
  `postprocess/tests/test_extract_calltree_traced.py` (update its `MODULE_PATH`/spec references
  accordingly); existing fixtures (`calltree_kernel_anchor`, etc.) are unaffected.
- No launcher-script auto-generation for this renamed tool specifically — it remains available
  standalone (`python3 postprocess/extract_calltree_traced.py <output-dir>`), matching this
  project's "extractor also works standalone" pattern for every other tool. The existing
  `CALLTREE_EXTRACTOR` variable in the three launcher scripts keeps its filename
  (`extract_calltree.py`), so it automatically now runs the **new** tool once created — no
  script logic changes needed for that part, only for forwarding the new `--show-*` flags (see
  below).
- `docs/plans/13-calltree-tool.md` stays as the historical record of this tool as originally
  built; add a short header note (matching this repo's divergence-note convention) pointing to
  the rename and this new plan doc.

### New file: `postprocess/extract_calltree.py` (sampling-based)

**`load_rank_trees(cpu_dir, show_rocprofsys_internals)`**: prefers `sampling_wall_clock-<pid>.txt`
per rank, falls back to `wall_clock-<pid>.txt` only for a rank with no sampling file at all (same
whole-file-fallback principle as the traced tool, just the preference order flipped). Per row,
sets `row["gpu"]` (existing `classify_gpu()`/`.kd`/broadened-GPU-noise check, from
`calltree_common.py`), `row["compiler_runtime"]` (new tier), `row["mpi_territory"]` (new tier).
Then, unless `show_rocprofsys_internals`, runs the splice pass (reassign every row's `parent` to
skip past any chain of wrapper ancestors, then drop wrapper rows from the list) before computing
`roots`.

**Wall-clock cross-reference**: `wall_clock_self_by_rank(cpu_dir)` — one `cpu_tool.scan_ranks()`
call, building `{rank_key: {label: self_sum}}` from its raw merged rows directly (see decision 2
above for why not `aggregate_per_rank()`). `render_node` gains a 4th column, looking up
`wall_clock_by_rank.get(rank_key, {}).get(node["label"])`, rendered as `-` when absent (marker
rows keep only 3 blank-worthy columns as today, no `WALL(s)` value either).

**`build_children_map(rows, show_mpi_internals)`**: same shape as the traced tool's, plus: a
node with `mpi_territory=True` contributes no children unless `show_mpi_internals`.

**`render_node`/`render_forest`**: same tree-connector approach as the traced tool, gains the
`WALL(s)` column and checks both `row["gpu"]` (vs `show_gpu_api`) and `row["compiler_runtime"]`
(vs `show_compiler_runtime`) for pruning.

**CLI**:
```
extract_calltree.py OUTPUT_DIR [-o OUTPUT] [--max-depth N]
  [--show-gpu-api] [--show-rocprofsys-internals] [--show-mpi-internals]
  [--show-compiler-runtime] [--show-all-internals]
```
`--show-all-internals` sets all four other show-flags; `-o` default `<OUTPUT_DIR>/calltree.txt`
(reclaimed from the renamed tool).

**GPU kernel integration**: same 3-case logic (single anchor / proportional split / no-anchor
fallback) via the shared `calltree_common.find_kernel_anchors`/`attach_kernel_summaries`, now
fed a broadened, `calltree_common`-owned launch-label matcher (substring + `__cray_start_acc_kernel`).

### Launcher script integration

`profile_CPU_hotspots.sh`, `profile_hotspots.sh`, `instrument_hotspots.sh` keep their existing
`CALLTREE_EXTRACTOR`/`CALLTREE_ARGS` wiring (from `docs/plans/13-calltree-tool.md`) unchanged in
structure — it automatically now runs the new tool once the rename+recreation lands. Add the four
new `--show-*` flags (plus `--show-all-internals`) as additional pass-through options collected
into the same `CALLTREE_ARGS` array, mirroring how `--max-depth`/`--show-gpu-api` were added last
time.

### Docs

- `README.md`'s "Call tree" section becomes two subsections: the new `calltree` tool (primary,
  described in full) and `calltree_traced` (short, "faster/simpler alternative, exact where
  instrumented but shallower — see below for why").
- New `docs/plans/14-sampling-calltree-tool.md` (this plan, saved verbatim after approval) is the
  record for this tool; `docs/plans/13-calltree-tool.md` gets the short header note described
  above.

### Tests

New `postprocess/tests/test_extract_calltree.py` (fresh file, new tool) plus
`postprocess/tests/test_calltree_common.py` for the shared module's kernel-anchor logic (migrated
from the traced tool's existing kernel-integration tests, which move to exercise
`calltree_common` directly instead of duplicating the assertions per-tool). Coverage needed,
using new fixtures built the same hand-crafted-with-known-values way as before:
- Each of the four filter tiers: hidden by default, visible with its own flag, and with
  `--show-all-internals`.
- Splice behavior specifically: a wrapper node's children correctly reparent to its grandparent
  (not dropped), including the "whole ancestor chain is wrapper frames" case (a row ending up
  with `parent=None`, becoming a new root — this is exactly what fixed the first broken
  experiment pass, worth a dedicated regression test).
- MPI collapse: a real MPI internal chain 3+ levels deep collapses to showing only the first MPI
  frame, both with and without `--show-mpi-internals`.
- Wall-clock cross-reference: a label present in both sources shows a real `WALL(s)` value; a
  label only in `sampling_wall_clock` (no `wall_clock` counterpart) shows `-`.
- Kernel anchor broadening: a namespace-qualified launch symbol (`hip::hipModuleLaunchKernel(...)`)
  and a bare `__cray_start_acc_kernel` both correctly resolve as anchors.
- `sampling_wall_clock`-missing-for-one-rank fallback to `wall_clock` for that rank only.

`extract_calltree_traced.py`'s existing 23 tests get renamed/re-pathed to the new test file name,
re-verified green after its `calltree_common.py` refactor (behavior must not change).

## Verification

- `python3 -m unittest discover -s postprocess/tests` — full suite green, including the renamed
  traced-tool tests (behavior-unchanged assertion) and the new tool's + shared module's tests.
- Manually regenerate `calltree.txt` (new tool) against the same real
  `Heat_Convection_Solver/profile_hotspots-output-2026-08-10_12.47.32` directory used for both
  experiments this session, with default flags — confirm it matches experiment #2's already-
  user-reviewed 227-line output (modulo the new `WALL(s)` column and any anchor-placement changes
  from the broadened launch-label matching) — deliver the file for direct review, same as every
  prior real-data validation this session, rather than just reporting the diff as passing.
- Also regenerate `calltree_traced.txt` (renamed tool) against the same directory, confirm its
  content is byte-for-byte identical to the pre-rename tool's last delivered output (proves the
  `calltree_common.py` refactor didn't change its behavior).
- `bash -n` on the three modified launcher scripts; stubbed-binary `--dry-run` check (as done for
  the previous calltree round) to confirm the new flags are forwarded correctly.
