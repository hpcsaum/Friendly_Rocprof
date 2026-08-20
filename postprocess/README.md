# `postprocess/` architecture

This is the developer-facing companion to the root [README.md](../README.md), which covers what
each tool does and how to run it. This document covers how they're built: the pipeline philosophy,
the `stage1`-`stage6` architecture, and where to hook in when adding a new tool.

## Philosophy

Every tool in `tools/` does the same fundamental job: take the raw per-rank text/CSV files
`rocprof-sys`/`rocprofv3` write, and turn them into one short, readable report. That job breaks
down into a fixed sequence of transformations — parse, link ancestry, classify noise, aggregate,
rank and render, assemble into a report — and every one of those transformations is genuinely
shared across tools, not just similar between them. `postprocess/` is organized so that sequence
is explicit: one directory per stage, each owning exactly one transformation and knowing nothing
about the tools built on top of it. A new report tool is almost entirely assembly — reusing stage1
parsing, stage3 noise classification, stage4 aggregation, and stage5/stage6 rendering — rather than
new parsing or rendering code, by construction.

## Layout

```
postprocess/
  _stage_paths.py   -- sys.path bootstrap, see below
  stage1/           -- parsing + run-directory resolution
  stage2/           -- ancestry linking
  stage3/           -- noise classification + tree surgery
  stage4/           -- aggregation
  stage5/           -- ranking, selection, and rendering
  stage6/           -- report assembly, run metadata, noise-config, shared CLI helpers
  tools/            -- the 9 CLI entry points
  tests/            -- mirrors the layout above, plus a shared tests/fixtures/
```

### `stage1/` — parsing and run-directory resolution

- `stage1_rocprofsys_sample.py` — parses `rocprof-sys`'s pipe-delimited timemory text tables
  (`wall_clock-<pid>.txt`, `sampling_wall_clock-<pid>.txt`, etc.) into row dicts (`label`, `count`,
  `depth`, `sum`, `self_sum`, ...).
- `stage1_rocprofv3.py` — parses `rocprofv3`'s `*_kernel_stats.csv` output.
- `stage1_run_dirs.py` — resolves a "run" directory (which may nest a `rocprof-sys/` and/or
  `rocprofv3/` subdirectory, or be one of those directories directly) into the `(cpu_dir,
  gpu_dir_or_None)` pair every CPU+GPU-pairing tool needs — `resolve_run_dirs()` for the
  single-directory-argument case, `resolve_two_dirs()` for the optional-second-argument case.

### `stage2/` — ancestry linking

`stage2_rocprofsys_sample.py`'s `attach_ancestry()` turns a flat list of parsed rows (each carrying only
its own `depth`) into a real tree: each row gets a `parent` back-reference to the row that called
it, `is_thread_root` for a rank's own top-level frames, and the frame-count-based logic needed to
tell two DEPTH-numbering conventions rocprof-sys's own output uses apart.

### `stage3/` — noise classification

`stage3_rocprofsys_sample.py` turns ancestry-linked rows into a per-row set of noise *tags* — a tag is a
fact about a row ("this row's label matches `wrapper_noise`"), deliberately kept separate from what
a tool does about it (drop it, hide its children, splice it out and reparent, collapse it). The
same tag can get a different treatment in a different tool without re-deriving the classification.
Tag definitions live in `default_noise_patterns.json` (stage6), not Python constants, so adding a
recognized noise pattern never requires a code change — see stage6 and "Customizing noise
filtering" below for the exact schema. This module also owns the generic, tag-driven tree-surgery
primitives every tool's own filtering builds on: `remove_tagged_subtrees()` (drop a whole subtree),
`splice_by_tag()` (remove a row, reparenting its children, optionally folding its self-time into
the new parent), `make_collapses_children()`/`make_is_pruned()` (predicates for the stage5
renderers below).

### `stage4/` — aggregation

Two parallel merge strategies, because a ranked table and a call tree need fundamentally different
shapes from the same underlying rows:

- `stage4_rocprofsys_sample_flat.py` — merges rows **by label** across every rank into one flat pool
  (`aggregate()`), or keeps each rank's own per-label totals separate (`aggregate_per_rank()`, for
  load-imbalance tables). Destroys tree position on purpose — a hotspot table doesn't care where in
  the tree a function was called from, only its totals.
- `stage4_rocprofsys_sample_tree.py` — merges N per-rank trees into one **by tree position**
  (`merge_rank_trees()`), preserving structure; also owns per-rank loading (`load_rank_trees()`),
  GPU-kernel-data pairing (`pair_gpu_per_rank()`) and GPU-kernel-to-CPU-launch-site attachment
  (`attach_gpu_kernels()`, see below), and `caller_chains_for_label()`, the inverse walk (target
  function → every distinct root-to-it ancestor chain) `extract_hotspot_callers.py` uses.
- `stage4_rocprofv3.py` — the equivalent by-kernel-name aggregation for `rocprofv3`'s own GPU data.
- `stage4_rank_merge_math.py` — the shared avg/std_dev/min/max-across-ranks math both merge
  strategies' load-imbalance/load-balance columns use.

### `stage5/` — ranking, selection, and rendering

`stage5_table_render.py` is the generic backend every ranked-table tool shares: `select_entries()`
(rank/threshold-filter/truncate a list of dict entries by whatever field a caller names, with no
opinion on what the entries represent), `render_table()` (turn entries into aligned text given a
column spec), plus the wrapping/legend-text helpers (`wrap_trailing_label()`, `pct_total_note()`,
`ranking_note()`) that keep a table's prose and its actual shape from drifting apart.
`stage5_tree_render.py` is the equivalent for call trees: `render_forest()`/`render_node()` (draw
`tree`-style connectors), `format_aligned_rows()` (right-align columns under a tree), and
`render_gpu_kernel_fallback()` (the rendering half of GPU-kernel attachment — the actual
attachment is `stage4_rocprofsys_sample_tree.attach_gpu_kernels()`; this only renders whatever
that couldn't place anywhere into the fallback table). Each `stage5_*_table.py` file is just a
column spec (what a specific table looks like); each `stage5_calltree*_view.py`/
`stage5_wallclock_calltree_view.py` file is a specific tool's own prune/collapse predicates plus
whatever postprocessing step it needs, composed over stage4's loading/merging/attachment
functions and stage5's rendering functions.

### The stage4 → stage5 entry contract

Every `stage5_*_table.py`/`stage5_tree_render.py` function is deliberately generic — it only ever
reads a handful of dict keys, never how they were computed — which is what lets a report kind be
reused across data sources instead of rewritten per source (see the trace-CSV tool family design,
`docs/plans/3.1-trace-postprocessing-family.md`). That reuse only actually happens when a stage4
backend targets one of a small number of canonical shapes; a new backend that invents its own
field names for the same kind of report loses the reuse even though the shape is conceptually the
same. The shapes in use today:

- **flat entry** (hotspots tables): `{label, count, sum, self_sum, pct_self, pct_total}`, plus an
  optional `domain` for a fused multi-source table. Produced by
  `stage4_rocprofsys_sample_flat.aggregate()`/`stage4_rocprofv3.aggregate()`; consumed by
  `stage5_cpu_hotspots_table.py`/`stage5_gpu_hotspots_table.py`/`stage5_fused_hotspots_table.py`.
- **tree node** (calltree): `{label, parent, children, per_rank, tags, structural_drop_tags,
  static_children}`. Produced by `stage4_rocprofsys_sample_tree.merge_rank_trees()`; consumed by
  `stage5_tree_render.py`'s rendering functions.
- **per-rank label→value** (load imbalance): plain `{label: value}`, one dict per rank. Produced
  by `stage4_*.aggregate_per_rank()`; consumed by
  `stage5_load_imbalance_table.compute_load_imbalance()` — the simplest shape, already fully
  generic with no per-domain variation at all.
- **per-rank timing summary** (POP metrics): `{rank_key, total_time, comm_time, useful_compute,
  cpu_only_time, gpu_busy_time}`, one dict per rank. Produced by
  `stage5_pop_metrics_table.gather_timing_summary_per_rank()`; consumed by
  `compute_metrics_from_per_rank()` in the same file.

A new backend for one of these *existing* report kinds should emit one of these shapes and gets
that report's stage5 view for free. A genuinely new report kind that doesn't fit any of them gets
its own new shape and its own stage5 view file — a normal outcome, not something to force into an
existing shape.

### `stage6/` — report assembly

- `stage6_report_builder.py` — the structural pieces every report shares: `standard_header()` (the
  tool-name/description/run-metadata block), `render_report()` (assembles a header, numbered
  sections, and a footer into one report string), `write_report_file()`.
- `stage6_run_metadata.py` — best-effort guessing of executable name/run datetime/total
  runtime/rank count from whatever metadata file (if any) a run happened to produce; a field that
  can't be found is left blank, never an error.
- `stage6_noise_config.py` — resolves the process-wide noise-tag definitions every stage3-consuming
  tool shares for its run: bundled `default_noise_patterns.json`, optionally customized by a
  user's `--extra-noise-config`/`$FRIENDLY_ROCPROF_NOISE_CONFIG` file. A module-level singleton,
  not threaded as a parameter — every real invocation of these tools is a single, one-shot CLI
  process with exactly one active noise configuration for its whole run.
- `stage6_cli_common.py` — the argparse/validation boilerplate most tools share: directory
  existence checks (`require_directory()`/`require_directories()`), `-o`/`--output` resolution
  (`resolve_dest()`), the `-n/--top`/`--threshold`/`--all` selection group
  (`add_selection_args()`), `--max-depth` (`add_max_depth_arg()`), and the `--show-*` noise-tier
  flags (`add_noise_tier_args()`).

### `tools/` — the CLI entry points

Each of the 9 files here is a compose-and-print script: parse arguments (mostly via
`stage6_cli_common`), pull data through stage1→stage4, rank/render it through stage5, assemble it
through stage6, write the file. `extract_calltree.py`/`extract_wallclock_calltree.py`/
`extract_hotspots.py`/`extract_hotspot_callers.py` also import each other directly
(`extract_CPU_hotspots`/`extract_GPU_hotspots` as `cpu_tool`/`gpu_tool`) to reuse their
`gather_run_info()` rather than duplicating metadata-guessing logic.

## Cross-stage imports and `_stage_paths.py`

A report tool typically touches most of the six stages at once (a calltree tool alone spans
stage1, stage2, stage3, stage4, and stage5), and several stages import from each other too
(stage4/stage5 import from stage1-3, stage3 imports from stage6). Every module keeps its plain,
flat import style regardless — `from stage1_rocprofsys_sample import parse_table_file`, not a
package-qualified path — so `postprocess/_stage_paths.py` puts every `stageN/` and `tools/`
directory on `sys.path` once; import it (after putting `postprocess/`'s own path on `sys.path` —
see any file in `tools/` for the one-line pattern) before importing anything from another stage.
See that module's own docstring for the exact mechanism.

## Customizing noise classification

`default_noise_patterns.json` (in `stage6/`) maps each tag name to a pattern definition:
`"prefixes"`/`"substrings"`/`"suffixes"` (case-insensitive label matches) and/or
`"filename_substrings"` (matched against the source file's basename), plus a couple of structural
rules (`"ancestor_for_thread_roots"`, `"first_real_descendant_skip_tag"`, `"sibling_group_source_tag"`
— see `stage3_rocprofsys_sample.py`'s own module docstring for exactly what each one does). A user's
`--extra-noise-config`/`$FRIENDLY_ROCPROF_NOISE_CONFIG` file layers a diff on top, resolved by
`stage6_noise_config.configure()`:

```json
{
  "add": {"other": ["my_library_prefix_"]},
  "remove": {"gpu_api": ["some_substring_that_over-matches"]},
  "disable": ["compiler_runtime_noise"]
}
```

`add`/`remove` only ever touch a tag's `"substrings"` list; `disable` drops a tag entirely and is
resolved first, so an `add`/`remove` naming a disabled tag is a no-op rather than an error. The
reserved `other` tag starts empty (no bundled patterns) and exists specifically for
`--extra-noise-config` to add to.

## Kernel-to-CPU attachment

`stage4_rocprofsys_sample_tree.py`'s `attach_kernel_summaries()` places real GPU kernel data (from
`rocprofv3`) onto the CPU call-tree node that actually launched it, two-tier:

1. **Name match**: Cray's OpenACC/HIP-offload kernel naming embeds the enclosing Fortran
   subroutine's name directly in the kernel name (`kernel_owner_label()` strips the
   compiler-generated suffix). When a CPU tree node with that exact label exists, the kernel
   attaches there directly — precise, since the compiler put that name there because that's
   literally the subroutine the kernel came from.
2. **Structural fallback**: a kernel that can't be matched by name (no Cray-style naming, or its
   owner subroutine wasn't sampled as its own distinct frame) attaches to the nearest ancestor of a
   known kernel-launch call (`find_kernel_anchors()`) instead — proportionally split by launch-call
   count when more than one candidate site exists. Neither approach is per-dispatch-exact: this
   toolchain's text/JSON output has no per-call timestamps to correlate a specific dispatch against
   a specific launch call.

A kernel matched (by either tier) becomes a synthetic node with a real `parent` link back to its
attachment point, so `caller_chains_for_label()` can walk it like any other row — see
`make_kernel_node()`'s and `attach_kernel_summaries()`'s own docstrings for exactly how.

## Building a new tool

A new `tools/extract_X.py` typically needs, in order:

1. **Resolve input directories** — `stage1_run_dirs.resolve_run_dirs()`/`resolve_two_dirs()`.
2. **Get data**:
   - Flat ranked table → `stage4_rocprofsys_sample_flat.aggregate()`/`aggregate_per_rank()` (CPU) or
     `stage4_rocprofv3.aggregate()` (GPU).
   - Call tree → `stage4_rocprofsys_sample_tree.load_rank_trees()` + `merge_rank_trees()`/
     `flatten_tree()`.
3. **Rank/render**:
   - Table → `stage5_table_render.select_entries()` + `render_table()` against a `stage5_*_table.py`
     column spec (or write a new one).
   - Tree → `stage5_tree_render.render_calltree_text()`/`format_aligned_rows()`.
4. **Assemble the report** — `stage6_report_builder.standard_header()`, `render_report()`,
   `write_report_file()`.
5. **Wire up `main()`** using `stage6_cli_common`'s `add_selection_args()`/`add_max_depth_arg()`/
   `add_noise_tier_args()`/`require_directory()`/`require_directories()`/`resolve_dest()`, and
   `stage6_noise_config.add_cli_argument()`/`configure_from_args()` if the tool reads
   noise-classified CPU data.

None of this is enforced by an interface or base class — it's a convention every existing tool
already follows, reusing the same functions rather than reimplementing them.

## Tests

`tests/` mirrors the `stageN/`/`tools/` layout, with one shared `tests/fixtures/` directory (not
split per stage — many fixtures, like a small MPI run, are exercised by tests across several
stages/tools at once). Two module-loading styles are in use, both preceded by the same
`_stage_paths` bootstrap:

- Most stage-module tests do a plain `from stageN_x import y` — fine for a module with no
  `__main__` guard or CLI surface of its own.
- Tool-level tests (and a few stage tests that specifically want a fresh, isolated module
  instance) use `importlib.util.spec_from_file_location()` instead, registering the loaded module
  under `sys.modules` — necessary since tools are meant to be run as scripts, not imported.

`stage6_noise_config`'s `_TAG_DEFS` is the one real piece of cross-module mutable state in this
codebase (a deliberate process-wide singleton, see its own module docstring); a test that loads an
isolated copy of it via `spec_from_file_location` must restore the previous `sys.modules` entry
afterward (see `tests/stage6/test_stage6_noise_config.py`), or it will silently detach every other
module's already-captured reference to the shared instance for the rest of the test process.

## Further reading

[docs/pop_metrics_reference.md](../docs/pop_metrics_reference.md) covers the POP-metrics-specific
math `extract_pop_metrics.py` implements — which official metrics are computed and why some
aren't, plus the project-specific GPU extensions.
