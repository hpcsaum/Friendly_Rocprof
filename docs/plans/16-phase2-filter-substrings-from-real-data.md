# Phase 2 (partial): expand filter substrings from real test_apps HPC data

## Context

Phase 1 (previous session) built `test_apps/` (c_app, fortran_app, cpp_app) and got them running on
real HPC hardware. Two real build bugs were found and fixed on that branch (OpenMP-offload link
flags, `--hipstdpar`-at-link conflict); GNU/OpenMPI testing is postponed due to a site toolchain
issue, out of scope here. The user has now captured 6 real profiling output directories under
`test_apps/results/` — every language (C/C++/Fortran) × both working compilers (amd, cray), all on
Cray MPICH, all on one MI300A node (`gfx942`) — and asked to use them to do Phase 2: expand the
noise-filtering substrings across the postprocessing tools from what's actually observed, per the
two-phase plan written in `docs/plans/15-test-apps-suite.md`. This is "partial" Phase 2: it covers
what these 6 directories actually exercise (AMD/LLVM and Cray-CCE OpenMP-target-offload, MPICH);
Open MPI and GNU-compiler-runtime noise remain for a later pass once that data exists.

Every finding below was confirmed by direct `grep` against the real files in `test_apps/results/`
(not inferred) — quoted verbatim so the "documented-schema, not invented" fixture convention can be
followed exactly when writing the new tests.

## Real findings driving each change

1. **`__tgt_target_kernel` is an unfiltered OpenMP-target-offload kernel-launch entry point**,
   confirmed present in 5 of 6 real `calltree.txt` files (`C_amd`, `C_cray`, `CPP_amd`, `CPP_cray`,
   `fortran_amd`) directly beneath application code (`launch_omp_kernel`) — e.g. in
   `profile_hotspots_C_cray/calltree.txt`:
   ```
   └── launch_omp_kernel
       └── __tgt_target_kernel
           └── [GPU kernels -- rocprofv3]
   ```
   This is the same role `__cray_start_acc_kernel` already plays for OpenACC in
   `KERNEL_LAUNCH_LABEL_SUBSTRINGS` — both AMD/LLVM's and Cray CCE's OpenMP-target-offload paths
   link the same `libomptarget` entry point in this environment, so one substring covers both.

2. **Beneath `__tgt_target_kernel`, on AMD/LLVM builds only** (`C_amd`, `CPP_amd`, `fortran_amd`),
   a chain of unfiltered LLVM AMDGPU-offload-plugin internals shows up in the rendered call tree —
   pure GPU-runtime plumbing, no application code, the OpenMP-offload equivalent of the already-
   filtered `rocprofiler::`/HIP internals:
   ```
   PluginManager::getDevice(unsigned int)
   └── DeviceTy::loadBinary(__tgt_device_image*)
       └── llvm::omp::target::plugin::GenericPluginTy::load_binary(...)
           └── llvm::omp::target::plugin::GenericDeviceTy::loadBinary(...)
               └── llvm::omp::target::plugin::AMDGPUDeviceTy::loadBinaryImpl(...)
                   └── llvm::omp::target::plugin::AMDGPUDeviceTy::launchDMInitKernel(...)
                       └── llvm::omp::target::plugin::AsyncInfoWrapperTy::finalize(llvm::Error&)
   ```

3. **On the Cray-compiler runs, `hotspots.txt`'s "CPU compute hotspots" (table 2) and "CPU load
   imbalance" (table 5) are polluted with AMD's GPU-kernel-JIT-compilation internals** — the
   clang/LLVM frontend + comgr pipeline that compiles GPU machine code the first time a kernel
   launches — misclassified as real CPU application compute time. Verbatim, from real
   `hotspots.txt` files:
   ```
   profile_hotspots_C_cray/hotspots.txt:      CPU    21  clang::CodeGen::mergeDefaultFunctionD...
   profile_hotspots_fortran_cray/hotspots.txt: CPU     1  amd_comgr_iterate_map_metadata
   profile_hotspots_C_cray/hotspots.txt (table 5): int llvm::array_pod_sort_...(...)
   ```
   The last one is return-type-prefixed (`int llvm::...`), which a `startswith`-only check (how
   `extract_CPU_hotspots.py`'s `GPU_API_PREFIXES` matches today) cannot catch — substring matching
   is required, the same reasoning `extract_calltree.py` already applied for its own broadened list.
   Confirmed via `grep` that `clang::`/`llvm::`/`amd_comgr` do **not** currently leak into any
   `calltree.txt` (only `hotspots.txt`), so `calltree_common`'s kernel-anchor logic is unaffected.

4. **The two independently-maintained MPI-prefix lists disagree**, and one has a latent
   case-sensitivity bug (this is why you chose "reconcile now" even though it isn't purely
   data-driven): `extract_calltree.py`'s `MPI_PREFIXES = ("mpi_", "pmpi_", "mpir_", "mpid_",
   "mpidi_")` is compared against a *lowercased* label, so it matches regardless of source casing.
   `extract_pop_metrics.py`'s `MPI_PREFIXES = ("MPI_", "PMPI_", "MPIR_", "MPID_")` is compared
   against the *raw* label with no lowercasing — it happens to work today only because real MPICH
   symbols (`MPI_Init`, `PMPI_Allreduce`, `MPIR_Waitall`, confirmed via `grep` across all 6
   `calltree.txt`) are already uppercase-prefixed, and it's also missing `mpidi_`/`MPIDI_` entirely.
   Any differently-cased MPI symbol would silently be undercounted as communication time.

5. **Open MPI's internal-symbol prefixes (`ompi_`, `opal_`, `orte_`) are well-documented Open MPI
   conventions**, even though no real Open MPI capture exists yet — per your feedback, worth adding
   proactively as a "most probable" list to refine later, rather than waiting on data that's
   currently postponed. The one real false-positive risk found while exploring (`ompi_group_t`
   inside rocprof-sys/timemory's own generic GOTCHA-wrapper template signature,
   `tim::component::gotcha<101ul, int, ompi_group_t**>(...)`) only appears **mid-string**, never at
   the start of a label — so matching these three prefixes with the same `startswith()` mechanism
   `MPI_PREFIXES` already uses (not a bare substring/`in` check) naturally avoids that landmine
   without needing any extra guard logic.

## Concrete changes

### `postprocess/calltree_common.py`
- `KERNEL_LAUNCH_LABEL_SUBSTRINGS`: add `"__tgt_target_kernel"`. Update the preceding comment — no
  longer say other compilers' offload equivalents "aren't included yet"; note it's now confirmed
  covered for AMD/LLVM's and Cray CCE's OpenMP-target-offload paths, per Finding 1.

### `postprocess/extract_calltree.py`
- `GPU_NOISE_SUBSTRINGS`: add `"__tgt_target_kernel"`, `"pluginmanager::"`, `"devicety::"`,
  `"llvm::omp::target::plugin::"` (Finding 2), plus `"clang::"`, `"llvm::"`, `"amd_comgr"` for
  consistency with the same GPU-kernel-JIT-compilation noise category confirmed via `hotspots.txt`
  (Finding 3), even though it didn't happen to surface in these particular 6 `calltree.txt` renders.
- Update the preceding comment to cite these real observations (mirroring the existing style).

### `postprocess/extract_CPU_hotspots.py`
- New constant `GPU_COMPILE_NOISE_SUBSTRINGS = ("clang::", "llvm::", "amd_comgr")`, matched via
  substring (`in`, case-insensitive) rather than `startswith` like `GPU_API_PREFIXES` — required
  because real labels include return-type-prefixed forms a prefix check can't catch (Finding 3).
- `is_gpu_entry()`: also return `True` when any `GPU_COMPILE_NOISE_SUBSTRINGS` entry is a substring
  of the lowercased label, additive to the existing `startswith(GPU_API_PREFIXES)` check.
- This fixes the same misclassification in `extract_hotspots.py` (tables 2/4) and
  `extract_pop_metrics.py` (compute-vs-communication split) automatically, since both consume
  `cpu_tool`'s classification rather than reimplementing it — no changes needed in those two files
  for this part.

### `postprocess/extract_calltree.py` (MPI reconciliation, continued)
- `MPI_PREFIXES`: also add `"ompi_"`, `"opal_"`, `"orte_"` (Finding 5) — matched via the same
  `lname.startswith(MPI_PREFIXES)` mechanism `is_mpi_territory()` already uses, so the confirmed
  `ompi_group_t`-in-a-GOTCHA-template false positive (which only occurs mid-string) is not matched.
  Comment these three as "most probable Open MPI internal prefixes, not yet confirmed against real
  Open MPI data — refine once Open MPI test_apps captures exist," distinct from the MPICH-prefix
  comment above them.

### `postprocess/extract_pop_metrics.py`
- `MPI_PREFIXES`: change to `("mpi_", "pmpi_", "mpir_", "mpid_", "mpidi_", "ompi_", "opal_",
  "orte_")` — same lowercase set as the reconciled `extract_calltree.py` list (Finding 4 + 5).
- Its call site (`label.startswith(MPI_PREFIXES)`) becomes `label.lower().startswith(MPI_PREFIXES)`.
- Update the caveat text (~line 411) that prints `MPI_PREFIXES` to reflect the new lowercase values,
  the case-insensitive matching, and that it now also covers Open MPI (probable, unconfirmed).

## Explicitly deferred / not changed this pass

- **Generic, collision-prone GPU kernel name `"kernel"`** from `cpp_app`'s `--hipstdpar` backend
  (confirmed in `CPP_amd`'s `kernel_stats.csv`/`kernel_trace.csv`, ~95% of that run's GPU time) —
  a kernel-identification/anchor-attribution concern, not a noise-filter-substring concern. Per
  your decision, left for a separate follow-up. (`CPP_cray` shows no such dispatch at all because
  Cray's compiler has no stdpar-offload path — `kernel_stdpar.cpp` runs CPU-only there, exactly as
  documented in that file already; this is expected, not a gap in the `cray` build.)
- **`__internal_tgt_target_teams`** — confirmed present in Cray's real symbol table
  (`functions-<rank>.json`) but never actually appears in any of the 6 rendered `calltree.txt` or
  `hotspots.txt` (not on an exercised hot path in this data). Not added, per this project's
  "observed in real sampled/exercised data" bar — add later if a run actually surfaces it.
- **`_cce$noloop$form`** and other compiler-specific kernel-name-mangling suffixes — a
  kernel-name-matching/normalization concern, out of scope for noise filtering.
- **Data-quality-only observations, no code change needed**: `fortran_amd` simultaneously loaded
  both production Cray MPICH and a separate "unsupported" test MPICH build
  (`/opt/hlrs/testing/unsupported/mpich/4.3.0b1-7.14/`); MPI-internal call-stack depth varies
  between runs due to sampling variance, not a toolchain difference; `fortran_cray`'s hotspots run
  is dominated by rocprof-sys's own GOTCHA symbol-interposition overhead rather than simulation
  work, likely a measurement-quality artifact for that specific run.

## Tests

Add hand-crafted fixture-based unit tests (existing `make_row()`/documented-schema convention,
using the exact real symbol strings quoted above, not invented ones):
- `tests/test_calltree_common.py` (`IsKernelLaunchTests`): `"__tgt_target_kernel"` → True.
- `tests/test_extract_calltree.py` (`GpuNoiseTierTests`): `"__tgt_target_kernel"`,
  `"PluginManager::getDevice(unsigned int)"`,
  `"llvm::omp::target::plugin::GenericPluginTy::load_binary(...)"`,
  `"amd_comgr_iterate_map_metadata"` → all classified as GPU noise.
- `tests/test_extract_CPU_hotspots.py` (`IsGpuEntryTests`):
  `"clang::CodeGen::mergeDefaultFunctionDefinition(...)"` → True;
  `"int llvm::array_pod_sort_by_key(...)"` → True (specifically exercises substring-vs-startswith);
  `"amd_comgr_iterate_map_metadata"` → True; a plain CPU-application label (e.g. `"run_simulation"`)
  → still False, to guard against a false-positive regression.
- `tests/test_extract_pop_metrics.py`: a case verifying an `mpidi_`-prefixed and a differently-cased
  MPI label are now both correctly counted as communication time.

## Docs

Update `docs/DEVELOPMENT_HISTORY.md` (+ regenerate `.docx` via the documented pandoc command,
per `CLAUDE.md`) with a new entry: "Phase 2 (partial): filter-substring expansion from real
test_apps HPC data" — summarizing the 4 findings above, the exact constants touched, and the
explicitly-deferred items. Per project convention, do not claim this is "verified" beyond
unit-test coverage against hand-crafted fixtures matching real data — it hasn't been re-run
against live HPC hardware inside this change.

## Verification

- `python3 -m unittest discover` (existing suite + new cases above) must pass first.
- Only the 4 python files above are touched, plus their respective test files — no new files.
- **Real-data validation, per your instruction**: unlike the Phase 1 apps themselves, the
  postprocessing tools are pure Python with no ROCm/HPC dependency, so they *can* run on this dev
  machine directly against the 6 already-captured directories under `test_apps/results/`. After the
  code changes and unit tests pass, re-run `extract_calltree.py` **and `extract_calltree_traced.py`**
  (and `extract_hotspots.py`/`extract_pop_metrics.py` where applicable) against each of the 6
  directories, writing output back to the same filenames already there so the new, filtered
  versions replace the old ones in place — keeping the new versions, as you asked, so you can
  inspect them.
  `extract_calltree_traced.py` matters specifically because it only filters via
  `cpu_tool.classify_gpu()` (the `extract_CPU_hotspots.py` fix, Finding 3) plus its own `.kd`-suffix
  check — it does **not** get `extract_calltree.py`'s broadened `GPU_NOISE_SUBSTRINGS`
  (`__tgt_target_kernel`, `pluginmanager::`, `devicety::`, `llvm::omp::target::plugin::`), since
  that list is local to `extract_calltree.py` and isn't shared. So it's genuinely open, not yet
  checked, whether these OpenMP-offload-plugin frames also leak into the traced tool's output — running
  it against the real 6 directories will show whether `extract_calltree_traced.py` needs the same
  substrings added, or whether its wall_clock-based data source just doesn't surface them the way
  the sampling-based `extract_calltree.py` does. Treat this as an open question this pass resolves
  from real output, not a predetermined answer.
- Manually diff/inspect each regenerated `calltree.txt`/`hotspots.txt` against what was reviewed in
  this planning session, confirming: (a) Findings 1–3's noise (`__tgt_target_kernel`,
  `PluginManager::`/`DeviceTy::`/`llvm::omp::target::plugin::`, `clang::`/`llvm::`/`amd_comgr`) no
  longer appears in the default view; (b) no genuine application frames got hidden as a side effect;
  (c) the reconciled/extended MPI prefixes don't change anything unexpected on these MPICH-only
  runs. Iterate on the substring lists if the real output reveals a problem or a new pattern worth
  covering, before considering this pass done.
- This is still not "verified on live HPC hardware" in the strongest sense — no new `rocprofv3`/
  `rocprof-sys` capture is taken, only the already-captured 6 directories are reprocessed — worth
  stating precisely in `DEVELOPMENT_HISTORY.md` rather than overclaiming, per this project's
  convention.
