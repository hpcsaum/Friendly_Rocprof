# Extracting theoretical GPU occupancy from `.ll` — exploration

Status: **exploration only — no plan, no code written.** Stage 1 (`parsing/stage1/`), the real
LLVM-IR frontend this exploration reads its examples from, is being implemented in parallel on this
branch by someone else; nothing here changes it. All findings below come from actually compiling
real HIP/OpenMP/CPU code with the ROCm 7.2.4 toolchain installed on this machine (`/opt/rocm-7.2.4`,
`hipcc`/`amdclang`/`llc`/`opt` all present) and inspecting the real `.ll`/`.s` output — not from
reasoning about LLVM in the abstract. No physical GPU is present or needed for any of this: register
allocation, resource-usage accounting, and the occupancy number itself are all static, compile-time
computations the AMDGPU backend does unconditionally, whether or not a device is attached.

## Verdict, up front

**Occupancy has two independent hard limiters — LDS (shared memory) and register footprint
(VGPRs/SGPRs) — weighed against fixed per-architecture hardware ceilings, plus the launch
configuration (threads/block). Of these, exactly one is cheaply and exactly readable from Stage 1's
`.ll`, one is sometimes readable, and the one that matters most in practice is not readable at all.**

- **LDS bytes/workgroup: exact, free, already confirmed.** A `__shared__`/team-shared array becomes
  an `addrspace(3)` global with a compile-time-constant size, directly in the frontend IR. Verified:
  a `258 x double` tile shows up as `2064` bytes in both the raw `.ll` and, independently, in the real
  backend's own `LDSByteSize` accounting — an exact match, no backend needed.
- **Explicit launch-bound intent: exact when the user wrote one, absent otherwise.** A
  `__launch_bounds__(256, 4)` annotation shows up verbatim as `"amdgpu-waves-per-eu"="4"` and
  `"amdgpu-flat-work-group-size"="1,256"` function attributes in the `.ll` — a real, if narrow,
  window into what the *developer asked for*. Absent any such annotation (the common case), neither
  attribute appears at all — nothing to read.
- **Register footprint (VGPRs/SGPRs) — the limiter that actually decides occupancy for most real
  kernels — does not exist anywhere in frontend LLVM IR, confirmed by grepping four real `.ll` files
  end to end.** It is produced only by the AMDGPU backend's real instruction selection + register
  allocation, and:
  - it is **highly sensitive to the actual optimization level**, confirmed with the same kernel's
    source compiled three different ways (16 vs. 32 vs. 44 VGPRs — see below);
  - for a kernel that calls into an external device-side runtime (confirmed with a real OpenMP
    `target` region), even a full backend compile of that single module **can't fully resolve it** —
    LLVM emits an unresolved symbolic formula, not a number, deferring resolution to the linker.

So: a full, standalone "theoretical occupancy" number is not buildable from Stage 1's `.ll` alone.
What *is* buildable, cheaply, is a narrower annotation — LDS reservation, explicit launch-bound
intent, and (see below) the target architecture string, all free from the same `.ll` — sitting next
to, not replacing, the real occupancy number ROCm's own profiler (`rocprof-compute`) already reports
once the kernel actually runs. This mirrors [`calltree_from_ll_idea.md`](calltree_from_ll_idea.md)'s
own shape almost exactly: a cheap, exact structural fact (there: the call graph; here: LDS size +
launch-bound intent + arch tag) is real and worth having, while the harder quantity everyone actually
wants (there: provable independence; here: the real occupancy number) needs machinery this project's
static frontend doesn't have and, in this case, genuinely *can't* have without a full backend compile
— and even then, only sometimes.

## What's cheaply extractable — confirmed with real tests

All three tests below compile a small real stencil kernel — one HIP `__global__` kernel using
`__shared__` memory, one OpenMP `#pragma omp target teams distribute parallel for`, and one plain
CPU loop — at Stage 1's own documented contract (`-O0 -gline-tables-only -Xclang
-disable-O0-optnone`, `--offload-arch=gfx90a`), matching exactly what `docs/plans/5.1-...md` and
`5.3-...md` specify as Stage 1's real input.

### Target architecture is tagged on every kernel, for free

```
$ grep -o '"target-cpu"="[^"]*"' kernel_O0.ll | sort -u
"target-cpu"="gfx90a"

$ grep -n 'attributes #4' kernel_O0.ll
attributes #4 = { ... "amdgpu-flat-work-group-size"="1,1024" ... "target-cpu"="gfx90a"
"target-features"="+16-bit-insts,...,+gfx90a-insts,...,+wavefrontsize64" ... }
```

Every kernel's own LLVM function attributes carry `target-cpu` (the exact `gfxNNN` string) and
`target-features` (a flag list including `+wavefrontsize64`/`+wavefrontsize32` when relevant). This
is real, already present, and requires zero backend work — a downstream occupancy-facts consumer
could key a per-architecture hardware-constants lookup table (max VGPRs/SGPRs per SIMD, max LDS per
CU, max waves per SIMD, wavefront size) directly off a string already sitting in the IR, with no
separate `--offload-arch` argument needed from the user. This project's own docs already reference
`gfx908`/`gfx90a`/`gfx940`/`gfx941`/`gfx942`/`gfx950` as real targets, so a small hand-maintained
table (values from AMD's own ISA/CDNA documentation, not derived from `.ll`) is the natural shape for
the hardware-constants half of this idea.

### LDS (shared memory) reservation is exact, and cross-checked against the real backend

The test kernel declares `__shared__ double tile[BLOCK + 2]` with `BLOCK = 256`:

```
$ grep -n 'addrspace(3)' kernel_O0.ll
15:@_ZZ14stencil_kernelPKdPdiE4tile = internal addrspace(3) global [258 x double] undef, align 16
```

`258 × 8 bytes = 2064 bytes` — a compile-time-constant size sitting in a global's type, no analysis
required beyond reading the array type. Compiling the *same* source through the real, full production
pipeline (`hipcc` default optimization, real backend codegen) confirms the backend agrees exactly:

```
; LDSByteSize: 2064 bytes/workgroup (compile time only)
```

This is the one real, exact, load-bearing number this idea can hand over with full confidence — LDS
capacity per CU is a fixed hardware constant (64 KiB on most current CDNA/RDNA parts), so "does this
kernel's LDS request alone rule out high occupancy" is answerable straight from the frontend `.ll`,
no backend compile needed. Static (fully unrolled/constant-indexed) LDS sizing only — a
runtime-sized `extern __shared__` array (size passed as a kernel-launch parameter) would need the
same host-side launch-argument tracing `docs/calltree_from_ll_idea.md`'s Use Case 3 already flags as
open, not attempted here.

### An explicit launch-bound request is directly readable — but only when the developer wrote one

Re-testing the same kernel with `__launch_bounds__(256, 4)` added:

```
$ grep -o '"amdgpu-flat-work-group-size"="[^"]*"\|"amdgpu-waves-per-eu"="[^"]*"' kernel_lb_O0.ll | sort -u
"amdgpu-flat-work-group-size"="1,256"
"amdgpu-waves-per-eu"="4"
```

`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`'s second argument maps directly
and losslessly onto `amdgpu-waves-per-eu` — the developer's own explicit occupancy *target*, sitting
right there in the `.ll`. Confirmed the attribute is genuinely conditional, not always-present: the
same kernel compiled *without* `__launch_bounds__` has **zero** occurrences of
`"amdgpu-waves-per-eu"` anywhere in the file. So this is a real but narrow signal — present only for
the subset of kernels whose author already thought about occupancy explicitly, silent otherwise. It
is also only the *request*, not proof the compiler achieved it; whether the real kernel actually fits
inside that budget is still a backend-only question (see below).

### The default work-group-size ceiling differs by language/runtime — also free, also confirmed

Absent an explicit annotation, Clang still stamps a default `amdgpu-flat-work-group-size` range, and
it differs by source language:

```
HIP  (kernel_O0.ll):  "amdgpu-flat-work-group-size"="1,1024"
OMP  (omp_O0.ll):     "amdgpu-flat-work-group-size"="1,256"
```

A HIP `__global__` kernel with no launch-bounds annotation defaults to an upper bound of 1024
threads/block; an OpenMP `target teams distribute parallel for` region compiled through
`amdclang -fopenmp --offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa` defaults to 256. This is
a real, cheap, per-language fact worth recording if this idea is ever built — the *ceiling* the
compiler assumed, not the actual runtime block size (which is a host-side launch-call argument,
outside `.ll`'s device-side text entirely — see Decisions below).

## The real blocker: register footprint is backend-only, and swings hard with optimization level

Register allocation (VGPRs/SGPRs) is what actually decides occupancy for the large majority of real
kernels in practice (LDS is usually not the binding constraint — see the counters reference quoted
below). It is produced by the AMDGPU backend's instruction selection and register allocation, which
run on top of LLVM IR — they are not part of LLVM IR itself. **Grepping all four `.ll` files
generated for this exploration for `NumVgprs`/`NumSgprs`/`Occupancy`/any register-count string
returns nothing** — the concept simply isn't representable at the frontend-IR level Stage 1 reads.

Compiling one real kernel's identical source three different ways confirms this isn't just an
absence — the *actual number*, once you do go get it, moves a lot depending on exactly which
optimization pipeline actually ran:

| Compile path | NumVgprs | TotalNumSgprs | ScratchSize (spill) | Reported Occupancy |
|---|---|---|---|---|
| **Real production build** (`hipcc`, full default `-O3`-class pipeline) | **16** | 16 | 0 | 8 |
| Stage 1's raw contract `.ll` (`-O0 -gline-tables-only -disable-O0-optnone`) fed directly through `llc -mcpu=gfx90a` | 32 | 56 | 136 bytes | 8 |
| The *same* `.ll` after Stage 1's own planned normalization passes (`sroa,mem2reg,instcombine,simplifycfg` — see `5.1-theoretical-roofline-tool-design.md`) then `llc` | **44** | 46 | 32 bytes | 8 |

(Full real `; Kernel info:` comment blocks for all three, unedited, are in this exploration's
scratch directory and reproducible with the commands shown throughout this document.)

Two real conclusions follow directly from this table:

1. **Feeding Stage 1's own `.ll` through `llc` ourselves would answer the wrong question.** The
   number produced (32, or 44 after normalization) describes a deliberately unoptimized/partially
   normalized code shape Stage 1 keeps *specifically because* it's easy to walk for op/byte counting
   — not the shape that will actually run on the GPU. Neither number is a usable proxy for the real
   16-VGPR/0-spill footprint the production compile actually produces.
2. **Stage 1's own planned normalization pass list does not reliably converge toward the real
   number, and can move the wrong direction.** Adding `sroa`/`mem2reg`/`instcombine`/`simplifycfg` —
   the exact pass list `5.1-theoretical-roofline-tool-design.md` already plans to run for its own,
   unrelated reasons — pushed VGPR usage *up* (32 → 44) rather than down, because naive `mem2reg`
   promotion here keeps more scalar temporaries alive as SSA values across the kernel body than the
   real `-O3` pipeline's fuller optimization set (GVN, real scheduling, etc.) ends up needing. A
   register-footprint estimate built on top of Stage 1's existing normalization would be actively
   misleading, not just imprecise, and there is no cheap partial fix available inside `.ll`-only
   tooling — closing this gap for real means running (or approximating) the actual production
   optimization pipeline, which is a fundamentally different, heavier thing than what Stage 1 does
   today.

Coincidentally, all three builds above land on `Occupancy: 8` — LLVM's own arithmetic, not something
this exploration computed. That's not evidence the register-count discrepancy is harmless in general;
it's evidence that for *this specific kernel*, none of the three register counts (16, 32, or 44) is
large enough to become the binding constraint before the hardware's own wave-slot ceiling kicks in
first. A kernel closer to the register-bound regime would very plausibly show three different
Occupancy numbers here, not the same one — not tested, since it would need a second, deliberately
register-hungry fixture kernel this exploration didn't build.

### A harder case: some kernels' register footprint isn't even known until link time

The OpenMP test kernel calls into the OpenMP device runtime library (barrier/reduction support code
not defined in the same translation unit). Feeding its device IR through the identical `llc
-mcpu=gfx90a` command used above produces this instead of numbers:

```
; NumVgprs: __omp_offloading_10302_480350_stencil_omp_l4.num_vgpr
; TotalNumVgprs: totalnumvgprs(...num_agpr, ...num_vgpr)
; Occupancy: occupancy(8, 8, 512, 8, 8, max(...numbered_sgpr+extrasgprs(...), 1, 0), max(totalnumvgprs(...), 1, 0))
```

LLVM emits a literal **unresolved algebraic expression** — `occupancy(...)` — rather than a number,
because the callee's own register usage isn't known within this one module; the expression is
resolved later by the linker (`lld`'s AMDGPU-specific expression evaluator), once every referenced
symbol is available. This is a real, separate blocker beyond "the IR doesn't have it yet": even a
full single-module backend compile of exactly the artifact Stage 1 would need is sometimes
**structurally incapable** of producing a number at all, for reasons that have nothing to do with
optimization level. Any occupancy tool for OpenMP-offloaded or otherwise non-self-contained device
code needs either a full link step in its pipeline, or an explicit "unknown — depends on a runtime
callee" fallback; this is not a corner case worth ignoring, since OpenMP offload is one of the three
input shapes this project explicitly commits to supporting (see `CLAUDE.md`).

## Per-source-type findings, as tested

**HIP kernels** (`__global__`, compiled `--offload-arch=gfx90a --cuda-device-only`): the most
tractable of the three. Real `amdgpu_kernel` calling convention, real `addrspace(3)` LDS globals when
`__shared__` is used, real `target-cpu`/`target-features`/optional launch-bound attributes, all
confirmed above. Register footprint is exactly the general blocker described above — no more, no
less tractable than any other AMDGPU kernel once you're past IR.

**OpenMP `target` offload** (`#pragma omp target teams distribute parallel for`, compiled
`-fopenmp --offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa --offload-device-only`): compiles
to the identical `amdgpu_kernel` shape as HIP (same calling convention, same `target-cpu` attribute,
same LDS mechanism when a reduction or team-shared construct needs it — not exercised by this
particular fixture, which uses no shared/reduction clause and correctly shows zero `addrspace(3)`
globals). The one HIP-vs-OMP difference actually found is the default work-group-size ceiling
(1024 vs. 256, above). The one OMP-specific extra difficulty found is the link-time-only register
resolution above, which does not occur for the HIP fixture (no external device-side callees in that
kernel) — a real, language-dependent difference in how hard this problem is, not a symmetric one.

**A standard, not-yet-offloaded CPU loop** (plain C, compiled for `x86_64-unknown-linux-gnu`, no
offload flags at all): confirmed the IR carries **none** of the above — no `addrspace` annotations at
all, no `amdgpu_kernel` calling convention, no `target-cpu=gfxNNN`, nothing wavefront-shaped:

```
$ grep -n '^target triple\|addrspace\|amdgpu\|define' cpu_O0.ll
target triple = "x86_64-unknown-linux-gnu"
define dso_local void @stencil_cpu(ptr noalias noundef %0, ptr noalias noundef %1, i32 noundef %2) ...
```

"Occupancy" isn't merely hard to compute here — it isn't a defined concept for this IR at all; a
wavefront/CU/register-file model doesn't exist for x86_64 in the first place. The only honest thing
this idea could offer a not-yet-ported CPU loop candidate is a *qualitative* complexity signal (how
many distinct live values does the loop body carry at its deepest point, does it already touch an
array large enough that a naive port's `__shared__` tiling would eat into the same 64 KiB LDS budget
demonstrated above) — never a number, and never something derived by compiling the CPU IR itself,
since compiling *for a GPU target* is a precondition for any of the real facts above to exist at all.
This is squarely the same territory `docs/plans/5.1-theoretical-roofline-tool-design.md`'s own
op/byte-count roofline analysis already covers for CPU loops (arithmetic intensity, trip counts) —
this idea adds nothing new on top of that for the CPU case specifically.

## Where the real (measured) version of this already lives

`docs/rocprof_compute_counters_reference.md`, already in this repo, documents that `rocprof-compute`
reports both a *measured* occupancy and the literal, hardware-counter-derived reason it's capped,
once a kernel actually runs on real silicon:

> **Wavefront Occupancy** ... The time-averaged number of wavefronts resident on the accelerator over
> the lifetime of the kernel. ... This is also presented as a percent of the peak theoretical
> occupancy achievable on the specific accelerator.

> **Insufficient SIMD Waveslots / VGPRs / SGPRs**, **Insufficient CU LDS**, **Insufficient CU
> Barriers** — Workgroup Manager (SPI) — The direct answer to "why is occupancy capped" ... register
> or LDS pressure from the kernel's own resource usage.

This is exactly the same shape of finding `calltree_from_ll_idea.md`'s Use Case 1 already reached for
its own, different static fact: the empirical/dynamic side of this project's toolchain already
computes the real thing, from real register allocation and real hardware counters, strictly more
reliably than any static `.ll`-only estimate ever could — because it has the one input (compiled,
linked, actually-scheduled ISA) that this whole exploration confirms `.ll` fundamentally lacks. The
genuinely new, non-redundant role for a static `.ll`-derived fact here is the same *annotation*
pattern already proposed there: sit a cheap static fact (LDS bytes/workgroup, an explicit
`__launch_bounds__` target) next to `rocprof-compute`'s real measured numbers, so a mismatch (e.g.
"the developer asked for 4 waves/EU via `__launch_bounds__`; measured achieved occupancy corresponds
to 2") is visible and explainable — not to predict the real number in place of measuring it.

## Checking the idea against the real Stage 1 implementation

Stage 1's data model (`parsing/stage2/stage2_ir.py`) already has three relevant pieces of schema —
all currently placeholders, none populated by any real extraction logic yet:

- **`TargetKind`** (`CPU_LOOP` / `GPU_DEVICE_KERNEL` / `GPU_OFFLOAD_REGION`) already exists on every
  `Kernel` and already distinguishes exactly the three cases this exploration tested — a real,
  already-built discriminator for "does occupancy even apply here," free to use as-is.
- **`MemorySpace.SHARED_LOCAL`** already exists as an enum value on `Access`/`Symbol`, but nothing in
  `stage1_ir_walker.py` ever detects an `addrspace(3)` global or tags anything with it — confirmed by
  grep: zero hits for `addrspace`, `SHARED_LOCAL`, or `__shared__` anywhere in the walker. The LDS
  byte-size fact demonstrated above as cheap and exact is, today, not actually read by any code —
  real, free, and simply not wired up yet.
- **`LaunchConfig`** (`grid_dim`, `block_dim`, `gpu_vendor`) already exists as a field on `Kernel`,
  but `LaunchConfig(...)` is only ever constructed in `stage1_ir_json.py`'s (de)serialization
  helpers — never in the actual walker. No code path today populates a real kernel's grid/block
  dimensions from an actual `hipLaunchKernelGGL`/`<<<...>>>` host-side call site; the field is schema
  reserved for a future stage, per `GPUVendor`'s own docstring ("Unused by any 5.2 logic — reserved
  so 5.4/5.6 don't need a Kernel schema change"). This is exactly the same open item
  `calltree_from_ll_idea.md`'s Use Case 3 already flagged (matching a call site's launch-config
  arguments back to the callee) — still unresolved, confirmed still true here.
- **No field anywhere carries the `target-cpu`/`target-features` string** demonstrated above as
  free and exact — genuinely new extraction surface, not a gap in an existing field.

## Decisions that would need to be made before this becomes a real plan

- **Scope**: a narrow "known ceiling facts" annotation (LDS bytes/workgroup + explicit launch-bound
  intent + arch tag, all confirmed cheap and exact above) vs. attempting a full occupancy number —
  the latter requires solving the register-footprint problem this exploration found no `.ll`-only
  answer to.
- **Whether a real occupancy number is even in-scope for a tool that only ever sees `.ll`.** Every
  path tried here that produces a real number needs the actual compiled+linked backend output
  (production-optimized ISA, not Stage 1's `-O0` contract `.ll`, and — for non-self-contained kernels
  — a full link step) as an *additional* input alongside the `.ll`, a fundamentally different
  artifact than what Stage 1 currently accepts. Whether this project wants a tool with that
  dependency is a real, unmade choice, not a detail.
- **If the compiled/linked artifact is accepted as a second required input**: whether to parse the
  human-readable `; Kernel info:` `.s` comment block (confirmed present and simple to grep, but not a
  documented/stable ABI — an LLVM-internal debug-print convention, in principle subject to change
  without notice across LLVM versions) vs. some more structured source (a `.note`/ELF metadata
  section read via `roc-obj`, not investigated in this pass).
- **Whether to wire up the existing but unpopulated `MemorySpace.SHARED_LOCAL`/`LaunchConfig` schema
  now that both are confirmed to have a real, extractable source in the `.ll`** — a smaller, narrower
  piece of this idea that stands on its own regardless of what happens with the harder
  register-footprint question.
- **Whether/how to correlate host-side launch config (actual block size) with a device kernel's
  `.ll`**, needed for anything beyond the compiler's own default/declared ceiling — the same
  call-site-argument-matching machinery `calltree_from_ll_idea.md` already left open, not
  re-litigated here.
- **Whether an "annotate `rocprof-compute`'s real measurements with static facts" integration (the
  only clearly non-redundant use found here) is worth building at all**, given it requires both a
  real profiling run to have already happened and the compiled ISA to be available — a narrower
  precondition than either input alone.
- **How much, if anything, to say about the not-yet-ported CPU case.** This exploration found no
  real, number-producing answer for it beyond what the roofline tool's own op/byte-count analysis
  already provides; whether a qualitative "complexity/LDS-pressure-if-tiled" hint is worth building on
  top of that is unproven, not evaluated further here.

## Addendum: a sound, coarse "will this be a problem" register screen — refined per follow-up, tested

The section above concluded a full occupancy *number* isn't buildable from `.ll` alone. A follow-up
narrowed the actual ask: not a real value, but a cheap **"this kernel is going to be a problem"**
screen — count per-thread scalar/array read/write live values, assume one loop iteration = one
thread (the same loop-to-kernel mapping the roofline tool's own `Kernel` model already uses), and
check the resulting minimum register need against the architecture's real budget. This turns out to
be **a real, buildable, and testable idea** — not the same thing as predicting the compiler's actual
register allocation (still backend-only, per above), but a **sound one-directional bound**: a
lower-bound estimate of live per-thread values, run through the *same* arithmetic the AMDGPU backend
itself uses to turn a register count into an occupancy number, produces an **upper bound on real
achievable occupancy**. If that bound is already low, the real number is guaranteed to be at least
that bad. Built a real prototype (`vgpr_lowerbound.py`, `llvmlite`-based, not part of the real
codebase) and tested it against three real kernels compiled with the same ROCm 7.2.4 toolchain used
throughout this document. It works, with one significant, clearly-bounded failure mode found and
confirmed by testing, not assumed.

### Why an underestimate of register need gives a *sound* claim, and an overestimate doesn't

If `R_est` is a genuine **lower bound** on the real per-thread VGPR count (`R_est ≤ R_real`, always),
then running it through the backend's own occupancy arithmetic — `waves = min(hw_max_waves,
TOTAL_VGPR // roundup(R, granule))` — necessarily gives `occ_est ≥ occ_real` (a smaller register
count can only permit *more* concurrent waves, never fewer). So: **`occ_est` is a sound upper bound
on the real occupancy — "at best this many waves, could be worse."** That is exactly the right shape
for a screening claim: if `occ_est` is already low, the kernel is *guaranteed* to be at least that
constrained in reality — a trustworthy "this will be a problem" flag. The converse does **not**
hold: a healthy-looking `occ_est` is *not* proof the kernel is fine, since `R_est` could still be
undercounting the true need (confirmed below, concretely). **This is a one-sided tool: it can prove
trouble, it can never prove safety** — the same "flag candidates for a human, don't try to prove
absence" shape this project's own `calltree_from_ll_idea.md` already settled on for its own harder
question (independence, there; occupancy, here).

### Method, as actually built and run

1. **Divergence tainting**: seed = any call to `llvm.amdgcn.workitem.id.{x,y,z}` (what HIP's
   `threadIdx.{x,y,z}` — and OMP's per-thread index machinery — lower to); forward-propagate through
   every instruction's operands to a fixed point. A value never touched by a divergent input is
   uniform (broadcast across the wavefront, SGPR-eligible) and correctly excluded — this is what
   keeps the resulting count a VGPR-specific *lower* bound rather than an overcount that would break
   the soundness direction above. (`blockIdx`/`blockDim`-derived values are correctly left uniform —
   confirmed no false taint from `llvm.amdgcn.workgroup.id.*`, which legitimately is broadcast.)
2. **A simplified liveness proxy**: not full CFG-aware dataflow liveness — each divergent SSA value's
   "live span" is [its def position, its last lexical use position] in one flattened, per-block-order
   instruction list; the peak point is the flattened index where the sum of live values' slot-widths
   (2 for `double`/64-bit pointers, 1 for `i32`/`float`/LDS or private pointers — read straight off
   the module's own `target datalayout` string, another free, exact fact) is largest. **A real,
   named gap**: this doesn't properly model cross-iteration liveness for a genuine (non-unrolled)
   loop-carried value the way a textbook backward-dataflow-to-fixed-point liveness analysis would —
   plausibly still an undercount in that case, not tested separately from the unrolling gap below.
3. **Requires Stage 1's own planned normalization passes, PLUS one more: `inline`.** Confirmed by
   direct test: `sroa,mem2reg,instcombine,simplifycfg` alone (Stage 1's currently-planned list) does
   **not** eliminate the small device-runtime wrapper function (`__ockl_get_local_id`, a real,
   observed intermediary) that `threadIdx.x` actually routes through before `llvm.amdgcn.workitem.id.x`
   itself appears — divergence tainting on that pass list alone found **zero** divergent values (a
   silent, total failure, not a partial one). Adding `inline` to the front of the same list collapses
   the wrapper into a direct intrinsic call and taints correctly. This is a real, new, previously
   unidentified requirement on top of what `5.1-theoretical-roofline-tool-design.md` currently plans.
4. **A real `llvmlite` footgun found and fixed while building this**: `str()` on an operand
   `ValueRef` that references a *function* (a call's callee) dumps that **function's entire body
   text**, not just its name — confirmed directly (`str(callee_operand)` for a call to the wrapper
   above printed the wrapper's whole `define ... }` text). A naive substring check for
   `"workitem.id"` against that text is a **false-positive trap**: it happened to "work" purely
   because the wrapper's own body happens to mention the intrinsic by coincidence, not because the
   check was sound — confirmed by deliberately re-testing the same substring approach on IR where the
   wrapper was *not* inlined and getting a spuriously plausible-looking (but meaningless) answer.
   Fixed by matching `op.name` (reliable for a named global/function reference) instead of `str(op)`.
   Recorded here because it's exactly the kind of silent, plausible-but-wrong failure this document's
   own standard of evidence exists to catch — a first version of this prototype produced a
   coincidentally-reasonable-looking number for the wrong reason, and only re-testing against a
   second, deliberately-varied input surfaced that it was luck, not correctness.

### Empirical validation: three real kernels, ROCm 7.2.4, ground truth from the real compiled backend

| Kernel | Estimator's peak live VGPR-slots | Estimator's occupancy upper bound | **Real compiled** NumVgprs | **Real compiled** Occupancy | Verdict |
|---|---|---|---|---|---|
| Original stencil kernel (this document's earlier fixture) | 11 | 8 | 16 | 8 | Sound, tight, no false alarm |
| `register_heavy_kernel` — 40 independent accumulators, expressed as a `#pragma unroll`ed loop over a local `double acc[40]` array | 11 | 8 | **106** | **4** | **False negative** on Stage 1's currently-planned pass list — real kernel is genuinely register-bound; estimator misses it (resolved below by adding an unroll step) |
| `register_heavy_unrolled_kernel` — the *identical* computation, written as 40 separate named source-level scalars instead of an array+loop | **89** | **5** | 106 | 4 | Sound, close, correctly flags real trouble |

The middle row is the important, honestly-reported failure: the *same arithmetic*, expressed via a
compiler-unrolled local array instead of source-level-distinct scalars, produces an estimate 10×
too low and a completely missed occupancy problem. The reason, confirmed by inspection: `#pragma
unroll` is a hint the real production `-O3` pipeline's own loop-unroll pass acts on (turning the
array into 40 independent scalar registers via unroll-then-SROA) — Stage 1's `-O0`-based contract
never runs a real unroller (unrolling is a middle-end optimization, disabled at `-O0`), so
`sroa`/`mem2reg` alone can't split an array indexed by a genuine runtime loop variable into scalars;
the array stays as one opaque memory object our SSA-value-based liveness analysis is blind to. The
third row confirms this precisely: given the *identical* register-pressure pattern already expressed
as distinct named scalars (no array, no loop the analysis would need to unroll itself), the same
estimator gets within ~16% of the real register count and correctly flags the real occupancy problem
(estimated ceiling 5 vs. real 4 — an upper bound, as the soundness argument above requires).

**Practical consequence, stated plainly**: this technique is a real, sound "will this be a problem"
screen for kernels whose register-hungry values are already exposed as distinct scalars at the
source/IR level (independent accumulators, manually-staged intermediate values, register-blocked
loop bodies written out explicitly) — a genuinely common HPC pattern, and arguably the more
*interesting* case for a porting-candidate screen, since a hand-unrolled/register-blocked CPU loop is
exactly the shape most likely to become register-bound after a naive port.

### Closing the array/loop blind spot: mimicking the real unroll step, tested

A natural follow-up question: rather than accepting the middle row above as a hard limitation, could
the same analysis pipeline just run its own loop-unroll pass first, to replicate what the real
production compiler does to that array before the divergence/liveness analysis ever sees it? Tested
directly: added `loop-unroll` to the pass list, followed by a **second** `sroa,instcombine,
simplifycfg` pass (needed so the now-constant-indexed array accesses the unroller exposes actually
get split into independent scalars, not just unrolled-but-still-array-shaped code) —

```
opt -passes="inline,sroa,mem2reg,instcombine,simplifycfg,loop-unroll,sroa,instcombine,simplifycfg" \
    heavy_O0.ll -S -o heavy_normalized_unrolled.ll
```

Confirmed the loop genuinely disappeared (the kernel body dropped from a real loop structure to 2
straight-line blocks, and its one `alloca [40 x double]` — the `acc` array — vanished entirely,
fully promoted to scalars) with **no special unroll-threshold flag needed**: `NACC=40` is a
compile-time constant (a `#define`, not a runtime value), so `loop-unroll`'s own default heuristics
recognized a small, fully-known trip count and fully unrolled it without prompting. Re-running the
estimator on this pipeline's output against the **original `#pragma unroll`-only source** (not the
hand-unrolled variant) reproduced the hand-unrolled result **exactly**: peak = 89 live VGPR-slots,
occupancy upper bound = 5 — matching the real `Occupancy: 4` just as closely as the hand-unrolled
case did, from source code that never spelled out 40 separate variables. **The blind spot closes for
real, confirmed, not just argued** — the earlier "false negative" row was a property of Stage 1's
*currently-planned* pass list lacking an unroll step, not an inherent limit of the technique itself.

**The real remaining condition, matching the phrasing that prompted this test — "assuming it gets
realistic parameters to unroll"**: this specific success is the *easy* case, where the trip count
(`NACC`) is already a compile-time constant sitting in the IR, so "how many times to unroll" has one
unambiguous, parameter-free answer (all of them) that both LLVM's real production pipeline and this
exploration's own `opt` pipeline arrive at independently. A loop whose trip count is a genuine
**runtime** value (a kernel argument, not a `#define`) doesn't have that unambiguous answer — the
real compiler would use a cost-model-chosen **partial** unroll factor for an unknown trip count
(commonly a small power of two, not "all of it"), and getting this analysis to pick the *same* factor
means either requiring a concrete value for that runtime parameter as an input (the same `--param`
binding mechanism `5.1-theoretical-roofline-tool-design.md`'s own Stage 3 already uses for its
symbolic trip-count evaluation — a real, existing precedent, not new infrastructure) or accepting a
best-guess unroll factor that may not match what the real backend would actually choose. Not tested
in this pass; the compile-time-constant case above is the one directly confirmed.

### The gfx90a hardware constants used above are now confirmed against two independent real data points, not guessed

Reverse-engineered `TOTAL_VGPR=512`, `granule=8`, `hw_max_waves=8` from the real `; Kernel info:`
comment blocks earlier in this document, then checked the same formula against *both* real kernels
tested here, independently:
- Stencil kernel: `NumVgprs=16` → `VGPRBlocks=1` → `(1+1)×8=16`/wave → `⌊512/16⌋=32`, capped at the
  hardware's own 8-wave/SIMD ceiling → **8**. Matches the real reported `Occupancy: 8`.
- Register-heavy kernel: `NumVgprs=106` → `VGPRBlocks=13` → `(13+1)×8=112`/wave → `⌊512/112⌋=4`.
  Matches the real reported `Occupancy: 4` **exactly**, no capping needed — this is the first real
  data point in this whole exploration where the register budget, not the hardware wave-slot
  ceiling, is what actually binds.

Two-for-two is a real, if still small, confirmation — worth cross-checking against AMD's own
published CDNA2/gfx90a ISA specification before trusting these specific numbers for any architecture
other than gfx90a, since they were derived by matching real compiler output rather than read directly
from AMD's documentation.

### What this changes about the earlier verdict

The original "Verdict, up front" said register footprint is invisible in `.ll` and left it there.
That's still true for the *actual* value — but **a sound, one-directional bound on it is not only
possible, it's now empirically demonstrated**, including for the case that initially looked like a
hard limitation: a loop over a small local array with a **compile-time-constant** trip count is fully
recoverable, confirmed, by adding a real `loop-unroll` pass (plus a second normalization pass to
mop up afterward) ahead of the divergence/liveness analysis — no special tuning needed for that case.
Combined with the LDS-bytes-per-workgroup fact from the main body of this document (exact, and
completely unaffected by any of the caveats above), a real "is this loop going to be a problem if
ported/launched as a kernel" screen looks like: take the per-thread divergent live value count (run
through an unroll step whenever the loop's trip count is statically known, with the runtime-trip-count
case below honestly surfaced as unresolved rather than silently guessed at), run it through the real
per-architecture occupancy arithmetic together with the exact LDS reservation size, and report
whichever of the two comes back lower — a genuinely new, non-redundant static capability this
exploration did not think was available before actually building and testing a prototype.

### Scope not yet tested, stated plainly rather than left implicit

- **Runtime (non-compile-time-constant) trip counts remain open.** The unroll step confirmed above
  only had one unambiguous answer because `NACC` was a `#define`, not a kernel argument — a loop whose
  bound is a genuine runtime value needs either a concrete value supplied for it (the same `--param`
  binding mechanism Stage 3 of the roofline design already uses for its own symbolic trip-count
  evaluation) or a best-guess partial-unroll factor that may not match the real backend's own
  cost-model choice. Not tested; flagged here rather than assumed solved by the constant-trip-count
  result above.
- **Data divergence only, not control divergence.** The prototype taints *values*, not *branches* —
  a real branch-divergence effect (extra exec-mask bookkeeping, or values kept live across a
  divergent `if` purely because of which lanes are active) isn't modeled. Not expected to change the
  main conclusion (the technique is still a sound lower bound, since ignoring a real cost source only
  makes the estimate more optimistic, which is the safe direction here) but not verified against a
  branch-heavy kernel.
- **SGPR side not built, only VGPR.** The same taint analysis's *complement* (uniform, non-divergent
  live values) would give an analogous SGPR lower bound by the identical argument — not implemented
  or tested here; every real kernel measured in this document had a comfortably small `TotalNumSgprs`
  (16 in both cases), so it never would have been the binding constraint regardless.
- **One kernel, one architecture's constants, three register-pressure shapes.** Real, but narrow,
  empirical coverage — not a claim this generalizes to every kernel shape or every `gfxNNN` target
  without further testing.

## Follow-up: resolving the open decisions and specifying the exact model

Written during a later, broader exploration (`docs/gpu_kernels_from_ll_idea.md`, plan 5.5 scoping)
that revisited this document's own open items with what that exploration separately learned —
the recognized thread-index call vocabulary needed for coalescing (its Q3), the OpenMP `collapse()`
div/mod index-recovery finding, and the `BranchPolicy.DIVERGENT_SUM` branch-cost work. This section
resolves each open item from above against that context, adds one new real (negative) empirical test,
and specifies the resulting model concretely enough to plan against.

### The original "Decisions" list, resolved

- **Scope (narrow ceiling-facts vs. full occupancy number): both, kept explicitly separate.** Ship
  the cheap, exact ceiling facts (LDS bytes/workgroup, explicit launch-bound intent, arch tag,
  default work-group-size ceiling) unconditionally — they have zero dependency on anything harder.
  Ship the VGPR-based occupancy estimate as a **separate, explicitly-labeled one-directional bound**,
  never merged into or presented alongside the exact facts as if it had the same certainty.
- **Whether a real occupancy number is in-scope for a `.ll`-only tool: no, settled.** This
  document's own findings already answer it — every path to a real number needs the actual
  compiled+linked backend artifact, a fundamentally different input than Stage 1 accepts. A real
  number is `rocprof-compute`'s job, not this tool's.
- **`; Kernel info:` comment parsing vs. ELF/`.note` metadata: moot, not merely deferred.** This
  decision assumed a second, compiled/linked input would be accepted specifically to get a real
  number — but the sound-lower-bound technique below needs **only** Stage 1's own `.ll`, run through
  an extended normalization pipeline, to produce its estimate. There is no second input to parse in
  the actual scoped model, so neither format question needs answering at all. (The separate
  "annotate `rocprof-compute`'s real measurements" integration, resolved next, reads
  `rocprof-compute`'s own structured report output, not a hand-parsed `.s` comment block either.)
- **Wire up `MemorySpace.SHARED_LOCAL`/`LaunchConfig`: yes, resolved, no longer open.**
  `gpu_kernels_from_ll_idea.md`'s Q1/Q2 sections give the concrete extraction recipe for both
  (`addrspace(3)` → `SHARED_LOCAL`; the `hipLaunchKernelGGL`→`dim3`-constructor-call pattern →
  `LaunchConfig`, for the common same-translation-unit case) — no longer a decision, a spec.
- **Correlating host-side launch config with device `.ll`: resolved for the common case, open in
  general.** Same source as above — direct, local `dim3` construction is now a solved extraction
  case; a launch site whose grid/block arguments come from a separate function or translation unit
  is still `calltree_from_ll_idea.md`'s open cross-function call-argument-tracing problem, consistent
  with every other place in this project's exploration series that hits the same boundary.
- **Whether the "annotate real measurements" integration is worth building: yes, more strongly than
  before.** This is no longer a single, isolated idea — `gpu_kernels_from_ll_idea.md`'s branch-cost
  section found a **second**, independent instance of the identical pattern (a `KEEP_SEPARATE`
  report row sitting next to `rocprof-compute`'s real **VALU Active Threads** counter, to resolve
  which of a divergent branch's possible costs actually applied). Two independent static facts now
  converge on the same integration shape — worth building as a general "static annotation next to a
  real measurement" mechanism in Stage 6, not a one-off.
- **How much to say about the not-yet-ported CPU case: unchanged.** Nothing in the later exploration
  bears on this; the original conclusion (no GPU-specific occupancy concept applies pre-port; the
  roofline tool's existing op/byte analysis is all there is) stands as-is.

### The addendum's own "Scope not yet tested," resolved

- **Runtime (non-compile-time-constant) trip counts: resolved by reduction, not left open.** Given a
  concrete value via the *same* `--param` binding mechanism Stage 3 already uses (and which
  `gpu_kernels_from_ll_idea.md`'s host↔device transfer-table design also reuses for USM placement
  hints — a now-recurring, deliberately-reused input shape across this whole project, not a new
  mechanism each time), a runtime trip count with a supplied binding reduces **exactly** to the
  already-solved compile-time-constant case: full unroll, one unambiguous answer, nothing new to
  design. Without a supplied binding, the honest answer stays "unknown," never a silent guess.
- **Control (branch) divergence: tested directly this session — real, negative result, worth
  reporting precisely rather than assumed.** Hypothesized that a divergent branch's exec-mask
  save/restore bookkeeping would show up as extra `TotalNumSgprs`, connecting this gap to the "SGPR
  side not built" gap below. Tested against a real, carefully-matched pair compiled through the full
  production pipeline (`hipcc`, default optimization, gfx90a) — two kernels, identical instruction
  shapes, differing *only* in whether the compared value is thread-derived (divergent) or a kernel
  argument (uniform):
  ```
  divergent (i % 2 == 0):   TotalNumSgprs: 11   NumVgprs: 7
  uniform   (mode % 2 == 0): TotalNumSgprs: 11   NumVgprs: 6
  ```
  **`TotalNumSgprs` is identical; the only delta is one extra VGPR in the divergent case.** The
  hypothesized SGPR connection does not hold, at least for a single, non-nested divergent region this
  small — a real, honest negative result, not a claim that control divergence never costs registers,
  just that it doesn't show up as *SGPR* cost the way textbook exec-mask-save/restore intuition
  suggests, for this shape. **The "SGPR side not built" and "control divergence not modeled" gaps
  should be treated as independent, not unified**, until a larger or nested-divergent-branch fixture
  says otherwise (not tested here) — a real, narrower version of the original open item, not a closed
  one. The 1-VGPR delta itself is unexplained by this test (plausibly a different lowering detail —
  e.g. how the compare result or a masked store's operand gets allocated — not investigated further).
- **SGPR side still not built — unaffected by the above, still a real, standalone extension.** The
  same taint analysis's complement (values *never* reached by the divergence taint — uniform,
  SGPR-eligible) gives an analogous SGPR lower bound by the identical soundness argument, independent
  of whatever control divergence turns out to cost. Still not implemented or tested.
- **Narrow empirical coverage — meaningfully widened, but not by re-running the estimator itself.**
  `gpu_kernels_from_ll_idea.md`'s own fixtures (a real 2-D HIP kernel, a real `collapse(2)` OpenMP
  kernel, a team-shared-memory OpenMP kernel, the divergent/uniform branch pair above) are new, real,
  and available — but the estimator's own liveness-proxy code was not re-run against any of them in
  this pass. Concretely flagged as the most valuable next validation step, below.

### The exact model, specified

**Inputs**: one Stage-1-contract `.ll` (`-O0 -gline-tables-only -Xclang -disable-O0-optnone`,
`--offload-arch=<gfxNNN>`); a small, hand-maintained per-architecture hardware-constants table
(`TOTAL_VGPR`, VGPR granule, max waves/SIMD — `TOTAL_VGPR=512`/`granule=8`/`hw_max_waves=8` cross-
checked for gfx90a specifically above, **not yet re-verified for any other `gfxNNN`** — a concrete,
cheap next step given fixtures for gfx942/gfx950 already exist from this session's own compiler
testing) keyed off the free `target-cpu` string; optional `--param` bindings for any runtime trip
count feeding a loop this estimator needs to unroll.

**Pipeline** (extends, doesn't replace, Stage 1's existing CPU normalization list):
1. Load + verify the module (needs `llvm.initialize_all_targets()`/`initialize_all_asmprinters()`,
   the already-documented Stage 1 gap `gpu_kernels_from_ll_idea.md` also relies on).
2. Normalize: Stage 1's existing `sroa, mem2reg, instcombine, simplifycfg`. **`inline` is *not* part
   of this list after all — a real correction, tested directly.** Ran
   `opt -passes="inline,sroa,mem2reg,instcombine,simplifycfg"` against the real `collapse(2)` OpenMP
   device `.ll`: `__kmpc_get_hardware_thread_id_in_block()` (OpenMP-target's thread-index primitive)
   **stayed an external declaration, with no body to inline, even after `inline` ran** — OpenMP's
   device-runtime wrapper isn't bundled into this module the way HIP's `__ockl_get_local_id` is; it's
   only resolved at the final device-link step, which Stage 1 never sees. So `inline` cannot expose a
   raw `llvm.amdgcn.workitem.id.x` for OpenMP at all — by-name recognition of the wrapper *call itself*
   (`__kmpc_get_hardware_thread_id_in_block`, alongside `llvm.amdgcn.workgroup.id.*` staying correctly
   uniform) is the only option there, not an alternative to `inline`. And since the same by-name
   recognition works equally well for HIP's `__ockl_get_local_id`/`__ockl_get_group_id`/
   `__ockl_get_local_size` calls directly — no need to inline them either, just treat the call result
   as tainted the same way — **by-name recognition should be the primary, always-working mechanism for
   both languages, not `inline`**, which turns out to be neither sufficient (OpenMP) nor necessary
   (HIP) for this specific purpose. `inline` may still matter for other reasons (stdpar's STL-accessor
   resolution, `gpu_kernels_from_ll_idea.md`'s hypothesis-6 section) but isn't part of *this* pipeline
   anymore. Then, only when a loop's trip count is a compile-time constant (or a runtime one with a
   concrete `--param` binding supplied): `loop-unroll`, followed by a second `sroa, instcombine,
   simplifycfg` pass, exactly as confirmed above.
3. Divergence-taint: seed = every result of a recognized thread-index call, by name
   (`llvm.amdgcn.workitem.id.{x,y,z}`; HIP's `__ockl_get_local_id`/`__ockl_get_group_id`/
   `__ockl_get_local_size`; OpenMP's `__kmpc_get_hardware_thread_id_in_block`) — forward-propagate
   through operands to a fixed point; correctly excludes `llvm.amdgcn.workgroup.id.*`/
   `__kmpc_get_hardware_num_blocks` (uniform: same value for every lane in one workgroup). **For a
   `collapse()`-shaped OpenMP loop specifically: confirmed by direct manual trace through the real
   normalized IR — no special-case code needed.** Tainting the flat id (`%13 = add %12(untainted),
   %9(tainted thread-id)`) and forward-propagating through the loop's own PHI and the `sdiv`/`mul`/
   `sub` chain recovering `row`/`col` (`%18/%21 = sdiv %.0(tainted), N(untainted)` → `row` tainted;
   `%23 = sub %.0(tainted), %22(tainted)` → `col` tainted) correctly reaches the final array index
   (`%26 = row*N + col`, tainted) with ordinary operand-following alone — **the earlier hedge about
   needing a special unroll/recognition step for this shape was overcautious; plain forward taint
   propagation already gets it right.** A related bonus, resolving Q3's own hedge about whether sympy
   would simplify the floor/mod round-trip back to the flat id for coalescing purposes: `idx = row*N +
   col` where `row = floor(flat/N)`, `col = flat − floor(flat/N)·N` reduces to `flat` by pure structural
   cancellation (`A + (B − A) = B`) — this needs no floor/mod-specific identity knowledge from sympy at
   all, only that both `sdiv` occurrences resolve to the *same* sub-expression (guaranteed, since both
   have identical operand chains and the resolver is a deterministic function of that chain) — a
   materially easier bar than originally worried about.
4. Peak-live-value liveness proxy: unchanged from above (flattened per-block-order instruction list,
   live span = [def, last use], peak = the flattened index where summed live tainted slot-widths is
   largest, slot width read from `target datalayout`).
5. `occ_est = min(hw_max_waves, floor(TOTAL_VGPR / roundup(peak_live_slots, granule)))` — a sound
   upper bound on real occupancy, never an exact value.
6. Combine with the LDS-bytes-derived ceiling (Q1's exact `addrspace(3)` reading) — report **whichever
   is lower** as the combined static "known ceiling" bound.
7. **Output shape is a small, explicitly-labeled table, not one number** — mirroring the transfer-table
   design's own separation of certainty levels:
   - *Exact facts* (always populated): LDS bytes/workgroup, explicit launch-bound intent (if
     present), arch tag, default work-group-size ceiling.
   - *A one-directional bound* (clearly labeled, never presented as measured): the combined VGPR/LDS
     occupancy ceiling from steps 3–6, with an explicit note of which known gaps could still make
     real occupancy *lower* than this bound (an unbound runtime trip count; the not-yet-built SGPR
     side). **One gap is now a concrete counting rule, not just a caveat**: re-tested control
     divergence against a second, cleaner, *nested*-branch fixture (properly argument-count-matched
     this time) and confirmed the earlier negative result generalizes — `TotalNumSgprs` identical
     (11 vs. 11) between a nested-divergent and a matched nested-uniform kernel — but also found a
     real, replicated, *positive* signal the estimator should act on: **VGPRs go up by roughly one per
     divergent branch *condition* evaluated** (+1 VGPR for one divergent condition; +2 for two nested
     ones), confirmed identically in both the flat and nested tests. The liveness proxy should count
     each divergent branch's own condition value (the tainted `i1`/comparison result feeding `br`) as
     an additional live tainted slot in its own right, not only the values computed *inside* the arms
     — a concrete, actionable refinement, not a vague "control divergence isn't modeled" hedge.
   - *Not reported at all*: a real, exact occupancy number — reserved for `rocprof-compute`'s own
     measurement, annotated by the facts above, per the now-twice-independently-motivated integration
     pattern.

**A confirmed, still-valid cross-cutting synergy, restated precisely**: the divergence-taint
classifier in step 3 is the *same* mechanism `gpu_kernels_from_ll_idea.md`'s `BranchPolicy.DIVERGENT_SUM`
work needs at each branch point — one shared implementation, two independent consumers (an occupancy
estimate here; a branch-cost policy decision there) — genuinely confirmed, unaffected by this
session's SGPR negative result, which only concerns whether divergence *also* costs extra registers,
not whether the classifier itself is shared.

### Concrete next validation steps — status after actually running them

1. **Done — confirmed by manual trace, not yet by the estimator's own code.** The `collapse(2)`
   taint-propagation question is resolved above: real, direct trace through the actual normalized IR
   confirms no special-case is needed. What's *still* not done is running the estimator's own
   liveness-peak/slot-counting code (never built as reusable software in either exploration — the
   original addendum's `vgpr_lowerbound.py` prototype was scratch work from a prior session, not
   preserved) to get an actual `occ_est` number for this fixture and compare it against the real
   compiled `Occupancy` — the manual trace confirms the *mechanism* works, not a specific number yet.
2. **Done, partially — a real, honestly-scoped result, not a full confirmation.** Compiled a
   register-heavy fixture (40 independent accumulators, matching the original addendum's own fixture
   shape) through the real production pipeline for gfx90a, gfx942, and gfx950: **identical NumVgprs
   (84) and identical Occupancy (5) across all three.** This is consistent with gfx90a's
   `TOTAL_VGPR=512`/`granule=8` also holding for gfx942/gfx950, and re-confirms those same constants
   against a *third*, independent gfx90a data point (this fixture) beyond the two the original
   addendum used. It does **not**, on its own, rule out a different, proportionally-scaled constant
   pair (e.g. double both `TOTAL_VGPR` and `granule`) producing the identical observed Occupancy —
   a genuinely different fixture, chosen to break that specific degeneracy, would be needed to fully
   pin down gfx942/gfx950's own constants independently; not attempted here, and AMD's own published
   CDNA3/CDNA4 ISA specs would be a more direct source than reverse-engineering a second data point.
3. **Done — re-tested, and refined into an actionable rule rather than a settled negative.** See the
   VGPR-counting-rule note above: the SGPR/control-divergence independence holds up under a second,
   properly-matched, *nested* test, but the accompanying VGPR-cost finding turns this from "control
   divergence isn't modeled" into a concrete, small, specific thing the estimator should count.
