# Introducing GPU kernels into the static roofline tool — exploration (plan 5.5 scoping)

Status: **exploration only — no plan, no code written.** This is a feasibility/scoping sketch for
plan 5.5 (the phase 5.1's own roadmap calls "GPU support"), following the same evidentiary standard
as [`occupancy_from_ll_idea.md`](occupancy_from_ll_idea.md) and
[`calltree_from_ll_idea.md`](calltree_from_ll_idea.md): claims below are either (a) confirmed by
actually compiling real HIP/OpenMP-target-offload C/C++ source with the ROCm 7.2.4 toolchain
installed on this machine (`hipcc`/`amdclang`/`opt` — real, working, driver-free; no GPU needed for
any of this, since kernel `.ll` generation and register/LDS accounting are compile-time-only) and
inspecting the real `.ll` output, or (b) explicitly flagged as "not tested" / "a real, scoped gap in
today's code" rather than assumed. Real HIP/OpenMP fixtures used throughout are in this exploration's
own scratch directory, reproducible with the commands shown.

This exploration also had a real advantage the other two didn't: a background survey of the actual,
current `parsing/` codebase (`stage1_ir_walker.py`, `stage1_frontend_llvm_ir.py`, `stage2_ir.py`) and
of what plans 5.1/5.3/5.4 already decided about GPU support. That survey found something worth
stating up front, since it corrects an assumption baked into 5.1's own roadmap: **5.1's roadmap
described "5.4" as the milestone where GPU support would land, expecting Stage 1 to need no changes
for it ("Same Stage 1 frontend... extended with `reuse_gpu.py`/`backend_gpu.py`")**. 5.4 as actually
built (`docs/plans/5.4-function-level-kernels-and-aggregation.md`) is confirmed, by direct grep, to
contain **zero** GPU-related code or design — it's a real, substantial CPU-only milestone (function-
level kernel decomposition, `SuccessorKind` sequencing, `SymbolOrigin`) that simply wasn't the GPU
phase after all. So 5.5 is not "just" Stage 3/4 GPU adapters on top of an already-GPU-aware Stage 1 —
**Stage 1 today has zero GPU recognition of any kind** (confirmed: no `addrspace`, no `amdgpu_kernel`
calling-convention check, no `llvm.amdgcn.*`/`llvm.nvvm.*` intrinsic recognition, no OpenMP-target-
offload runtime-call recognition anywhere in `stage1_ir_walker.py`; `stage1_frontend_llvm_ir.py`
doesn't even call `llvm.initialize_all_targets()`, so it cannot currently load an `amdgcn-amd-amdhsa`
`.ll` module at all). 5.5 needs to absorb that Stage 1 work too, not just Stage 3/4.

## Verdict, up front

**All three of the exploration's own opening questions (shared memory, host/device transfers,
coalescing) come back "yes, and cheaply" — and, more importantly, all three turn out to be readable
using machinery `stage1_ir_walker.py` already has for CPU loops, extended rather than replaced.** The
same is true for five of the seven modeling hypotheses given: they map onto the existing `Kernel`
schema's already-reserved-but-unpopulated fields (`TargetKind.GPU_DEVICE_KERNEL`/`GPU_OFFLOAD_REGION`,
`ParallelKind.GPU_THREAD`/`GPU_BLOCK`/`GPU_GRID`, `MemorySpace.SHARED_LOCAL`, `LaunchConfig`,
`GPUVendor`) almost exactly as 5.1 originally sketched them, and onto the sympy-expression-based
index/trip-count machinery already built for CPU loops. The two genuinely hard parts —
occupancy/register pressure (hypothesis 7) and the CUDA/USM edges of vendor coverage — are already
covered by prior exploration (`occupancy_from_ll_idea.md`'s tested, sound one-directional VGPR lower
bound) or are confirmed, honest gaps (Unified Shared Memory's actual transfer volume is not present
anywhere in `.ll`, full stop — not "hard to extract," genuinely absent).

## The three `.ll` questions, answered with real compiled tests

### 1. Explicit shared memory: exact, cheap, and identical for HIP and OpenMP

Confirms and extends `occupancy_from_ll_idea.md`'s HIP finding — **and newly confirms the same
mechanism for OpenMP's team-shared allocator**, not tested there. A HIP kernel's
`__shared__ double tile[258]` and an OpenMP `#pragma omp target teams` region's
`#pragma omp allocate(tile) allocator(omp_pteam_mem_alloc)` compile to the *identical* IR shape:

```llvm
; HIP:
@_ZZ14stencil_kernelPKdPdiE4tile = internal addrspace(3) global [258 x double] undef, align 16

; OpenMP (omp_pteam_mem_alloc):
@tile = internal addrspace(3) global [258 x double] poison, align 16
```

`addrspace(3)` + a fixed-size array type = LDS bytes/workgroup, exact, free, no backend compile
needed, for either language — `258 × 8 = 2064` bytes readable straight off the type. This is real,
already-reserved schema (`MemorySpace.SHARED_LOCAL`), simply never populated by the walker today
(every `Symbol` the walker builds is hardcoded `MemorySpace.UNKNOWN`).

**One new, real gotcha found that HIP doesn't have**: at `-O0`, OpenMP's runtime-based codegen also
routes ordinary-looking per-thread temporaries through `__kmpc_alloc_shared`/`__kmpc_free_shared`
runtime calls (not the LDS `addrspace(3)` global — a small heap-like allocation, freed at scope
exit), even for values that aren't actually team-shared:

```llvm
%39 = call align 16 ptr @__kmpc_alloc_shared(i64 4)   ; an ordinary i32 local, not "shared" in the
                                                        ; LDS/team sense despite the function's name
```

A GPU walker needs to recognize `__kmpc_alloc_shared`/`__kmpc_free_shared` as transparent
allocation/deallocation bookkeeping (treat the returned pointer like an ordinary local, same as an
`alloca`) rather than either (a) an opaque call whose cost is unknowable, or (b) mistaking the name
for genuine LDS traffic — a false-positive source specific to OpenMP-target, absent from HIP.

### 2. Host↔device data transfers: exact for explicit copies, structurally absent for USM

Three real, statically-readable shapes, plus one real, statically-*unreadable* shape — confirmed by
compiling and inspecting host-side IR for all four:

**HIP explicit (`hipMemcpy`)** — an ordinary call, direction and size both recoverable:
```llvm
%30 = call i32 @hipMemcpy(ptr %27, ptr %28, i64 %29, i32 noundef 1)   ; 1 = hipMemcpyHostToDevice
%55 = call i32 @hipMemcpy(ptr %52, ptr %53, i64 %54, i32 noundef 2)   ; 2 = hipMemcpyDeviceToHost
```
The direction is a small, stable enum constant; the byte count (`%29`/`%54`) is an ordinary
SSA-resolvable scalar (here `sext i32 n to i64; mul i64 _, 8` — exactly the kind of expression
`_resolve_scalar_value()` already resolves for CPU trip counts/free params). **Same call also exposes
`LaunchConfig` for free** — `hipLaunchKernelGGL` lowers to a `hipLaunchKernel` call preceded by two
`dim3` constructor calls (`_ZN4dim3C2Ejjj(&grid_alloca, x, y, z)`, `_ZN4dim3C2Ejjj(&block_alloca, x,
y, z)`) whose arguments are ordinary, already-resolvable values:
```llvm
call void @_ZN4dim3C2Ejjj(ptr %7, i32 %34, i32 1, i32 1)   ; grid  = ((n+255)/256, 1, 1)
call void @_ZN4dim3C2Ejjj(ptr %8, i32 256, i32 1, i32 1)   ; block = (256, 1, 1)
```
This directly closes 5.1's own open item ("`LaunchConfig(...)` is only ever constructed in
`stage1_ir_json.py`'s deserializer, never in a real walk") for the common case where the launch site
and its `dim3` arguments are local to the same translation unit — genuinely new extraction surface,
not previously identified as tractable.

**OpenMP `target` map clauses** — a `__tgt_target_kernel` call plus two constant global arrays whose
shape is part of LLVM's own stable, documented `OMPConstants.h` encoding:
```llvm
@.offload_sizes    = private constant [3 x i64] [i64 4, i64 0, i64 0]     ; bytes per mapped item
@.offload_maptypes = private constant [3 x i64] [i64 800, i64 34, i64 33] ; bit 0x01=TO, 0x02=FROM, ...
%68 = call i32 @__tgt_target_kernel(ptr @5, i64 -1, i32 0, i32 0, ptr @<region_id>, ptr %15)
```
A `0` size means the real size is computed elsewhere (bound to a runtime free parameter, e.g.
`in[0:n]`) — the same free-param fallback the CPU model already uses, not a dead end. **The zero-copy
case is equally readable**: an `is_device_ptr`-annotated region's maptypes (`288, 288` here) simply
lack the `TO`/`FROM` bits entirely — a real, static "no copy happens for this argument" signal, from
the same bitmask.

**HIP Unified Shared Memory (`hipMallocManaged`) is the one confirmed *unreadable* case** — not
"harder," genuinely absent: the allocation call is visible, but there is **no** subsequent
`hipMemcpy`-shaped call anywhere in the host IR for USM-allocated pointers used by a kernel launch.
The runtime migrates pages on first touch, at actual run time; there is nothing in `.ll` — no call,
no size, no direction — describing that traffic. An honest GPU cost model must surface USM-managed
allocations as "transfer volume/timing unknown, not zero" rather than silently omitting the cost or
guessing at it.

**Re-checked across architectures rather than assumed from one data point**: re-ran the same
`hipMallocManaged` fixture through `--offload-arch=gfx90a`, `gfx942`, `gfx942:xnack+`,
`gfx942:xnack-`, and `gfx950`. The host-side IR's `hipMallocManaged`/`hipLaunchKernel`/`hipFree` call
shapes are byte-identical across all five — the only diff anywhere is the `__hip_fatbin_<hash>`/
`__hip_cuid_<hash>` symbol names (a hash of the embedded device binary, non-semantic). The
`gfx942:xnack+` vs. `gfx942:xnack-` device IR differs by exactly one bit in the `target-features`
string (`+xnack`/`-xnack`) and nothing else — no new call, no different memory-instruction shape.
So the "unreadable" verdict holds architecture-independently, and the *compile-time*
`:xnack+`/`:xnack-` target-id suffix doesn't change it.

Worth being explicit about one distinction this raises: a runtime `HSA_XNACK=1`/`0` environment
variable (which governs whether the ROCm runtime actually uses true page-fault-retry demand paging
vs. eager whole-allocation migration for a managed pointer) is a **different knob** from the
compile-time `:xnack+`/`:xnack-` target-id suffix tested above, and is read by the runtime at
queue/context creation — strictly *after* the `.ll` this tool reads has already been produced.
It cannot appear in a compiled artifact at all, structurally, not merely "wasn't found" — there is no
version of this static tool, at any level of sophistication, that could recover it from `.ll`, since
the fact doesn't exist yet at compile time. This is a stronger, cleaner unreadability claim than
"the register-footprint problem" from `occupancy_from_ll_idea.md` (which is merely *deferred* to a
later compilation stage, not fundamentally absent) — worth keeping the two distinct when 5.5 documents
its own honest limitations.

### 3. Coalesced-access classification: yes — using the walker's own existing index-expression machinery

This is the strongest finding of the three, because it needs **no new extraction mechanism** — only
recognizing a small, fixed set of call targets that today fall through to "opaque call."

Confirmed: HIP's `threadIdx.x`/`blockIdx.x`/`blockDim.x` at `-O0` do **not** appear as the raw
`llvm.amdgcn.workitem.id.x`/`workgroup.id.x` intrinsics at the use site — they route through small
wrapper calls (matching `occupancy_from_ll_idea.md`'s earlier finding, confirmed again here):
```llvm
%28 = call i64 @__ockl_get_local_id(i32 noundef 0)    ; threadIdx.x  (arg 0/1/2 = x/y/z)
%32 = call i64 @__ockl_get_group_id(i32 noundef 0)    ; blockIdx.x
%36 = call i64 @__ockl_get_local_size(i32 noundef 0)  ; blockDim.x
```
OpenMP's `distribute parallel for` builds the equivalent global id from a *different* set of
recognized calls, combined with ordinary `mul`/`add` — and, usefully, **without** needing an `inline`
pass first (unlike HIP's `__ockl_*` wrappers, these appear directly at `-O0`):
```llvm
%33 = call i32 @__kmpc_get_hardware_thread_id_in_block()
%34 = call i32 @__kmpc_get_hardware_num_threads_in_block()
%35 = call i32 @llvm.amdgcn.workgroup.id.x()
%36 = mul i32 %35, %34      ; workgroup_id * block_size
%37 = add i32 %36, %33      ; + thread_id  ->  the flattened global id
```
Either way, the *pattern* is the same: a fixed, small vocabulary of by-name-recognized calls
(`__ockl_get_local_id`/`get_group_id`/`get_local_size` for HIP; `__kmpc_get_hardware_thread_id_in_block`/
`num_threads_in_block`/`num_blocks` + `llvm.amdgcn.workgroup.id.*` for OpenMP; the raw
`llvm.amdgcn.workitem.id.*`/`workgroup.id.*` intrinsics directly, wherever no wrapper is present)
feeding ordinary, already-walkable arithmetic — exactly the shape `OPAQUE_CALL_STL`/`OPAQUE_CALL_MPI`
already establish as a precedent (recognize a specific callee name, treat its result specially),
extended with one new `SymbolOrigin` (e.g. `GPU_THREAD_INDEX`) so `_resolve_scalar_value()` treats a
recognized thread-index call's result the way it already treats an induction-variable PHI, instead of
falling through to `CALL_SITE`/`UNRESOLVED`.

Once that's in place, every global-memory (`addrspace(1)`) `Access.index_expr` built by the existing
`_gep_to_access()`/`_resolve_index_operand()` pipeline is — in both fixtures tested — an ordinary
affine sympy expression in the recognized thread-index symbol (`gid`, `gid-1`, `gid+1`), built by
**exactly the same code path already used for a CPU loop's induction variable**, no new machinery
needed. Coalescing classification then falls out as a pure sympy operation with no new extraction
work at all:

- coefficient of the thread-index term = **1** (in elements) → fully coalesced: contiguous lanes hit
  contiguous elements, one burst per wavefront. Wavefront width itself is already a free, exact fact
  (`+wavefrontsize64`/`+wavefrontsize32` in every kernel's own `target-features` attribute, confirmed
  in `occupancy_from_ll_idea.md`) — no separate lookup needed beyond what's already proposed there.
- coefficient = **0** (index doesn't depend on the thread index at all — a `blockIdx`-only broadcast
  read) → uniform/broadcast, a single transaction, *cheaper* than the unit-stride case, a real third
  classification 5.1's own roadmap note already anticipated needing a fixture for.
- coefficient = a constant **> 1** → strided/uncoalesced, with the stride (coefficient × dtype size)
  being exactly the number a downstream memory-transaction-count model needs.
- a genuinely data-dependent index (`Access.depends_on` non-empty — gather/scatter) → correctly falls
  back to "not statically classifiable," the same worst-case-fallback shape
  `IndirectAccessPolicy.ALWAYS_COLD`/`OPTIMISTIC_REUSE` already models for CPU reuse distance — a
  policy toggle, not a new concept.

**The 2-D case was checked, and confirmed affine — for HIP's native two-axis form.** Built and
compiled a real 2-D matrix-add (`row = blockIdx.y*blockDim.y+threadIdx.y`, `idx = row*N+col`) at
Stage 1's exact contract. Confirmed: the *same* call vocabulary as the 1-D case, just parameterized by
a literal dim-argument (`__ockl_get_group_id(i32 1)`/`get_local_size(i32 1)`/`get_local_id(i32 1)` for
`y`, vs. `(i32 0)` for `x` — the dim argument is itself a plain compile-time-constant literal at each
call site, trivially readable, no further resolution needed). `row` and `col` are each independently
affine in their own recognized primitive, combined by ordinary `mul`/`add` into
`row*N + col` — structurally identical to the CPU transpose fixture's existing `i*n+j` (`N` a runtime
free parameter, exactly like that fixture's `n`). No new gap found here.

**A real, different, and more consequential gap: OpenMP's `collapse(2)` clause does not use two
hardware axes at all.** Compiled a real `#pragma omp target teams distribute parallel for
collapse(2)` matrix-add and inspected the device IR: only **one** flattened thread-index primitive
appears (the same single-axis `__kmpc_get_hardware_thread_id_in_block`/`llvm.amdgcn.workgroup.id.x`
vocabulary as the plain (uncollapsed) 1-D case) — the original loop nest's `row`/`col` indices are
recovered from that one flat id via genuine integer division and remainder against the **collapsed
loop's own runtime-valued trip count** (`N`, itself a kernel argument, not a compile-time constant):
```llvm
%68 = sdiv i64 %62, %67      ; row = flat / N
%85 = mul nsw i64 %79, %84   ; (flat / N) * N
%86 = sub nsw i64 %72, %85   ; col = flat - (flat/N)*N   -- no srem/urem emitted at -O0; LLVM
                             ; manually expands "%" into this div/mul/sub idiom instead
```
`idx = row*N + col` is then built from these two recovered values by ordinary `mul`/`add` — and is
**mathematically identical to the original flat id** whenever `row = flat // N, col = flat % N` (this
access really is perfectly coalesced) — but the IR only shows that through a floor-division/modulo
round-trip, not a bare affine expression in the flat id. The derivative-based coalescing check
described above only gives the correct answer if that round-trip gets algebraically simplified back
to the flat id first — sympy *can* represent and potentially simplify `floor(x/n)*n + Mod(x,n) → x`,
but whether it reliably does so automatically, for a symbolic (runtime) divisor `n`, starting from an
expression assembled instruction-by-instruction rather than handed a clean pre-simplified formula, was
**not tested here** — a real, new, honestly open gap, and arguably more consequential for
coalescing-classification accuracy than the runtime-trip-count-unroll gap
`occupancy_from_ll_idea.md` already flagged for its own, different purpose. Practically, this means a
GPU walker likely needs to recognize the `collapse(N)` div/mul/sub decomposition as its own structured
idiom and reconstruct the original per-dimension loop variables directly — mirroring the existing
precedent of `_recover_omp_for_loop_dim()` reading structured bounds out of a recognized
`__kmpc_for_static_init_4` call rather than resolving raw arithmetic — rather than trusting generic
arithmetic resolution plus a hoped-for sympy simplification. **This is OpenMP-specific**: HIP has no
`collapse()` equivalent (a genuine multi-D HIP kernel just uses `blockIdx.y`/`threadIdx.y` directly, as
confirmed affine above), so this is one more real, language-dependent asymmetry to add to the list
already found for shared-memory bookkeeping (Q1) and thread-index wrapping (Q3, main text) — OpenMP
needs recognizing *more*, and structurally different, idioms than HIP does, not just a parallel
by-name vocabulary.

**This also connects directly to hypothesis 7 (occupancy/register pressure), not just to
coalescing.** Per the user's own real-world experience profiling this class of code: the
`row`/`col`-recomputation arithmetic above is a known, concrete contributor to why an OpenMP-target
kernel typically needs more registers than the equivalent hand-written HIP kernel for the same
computation — every collapsed dimension re-derives its own `div`/`mul`/`sub`-recovered index as a
*genuinely live, divergent* (thread-varying) value, on top of whatever the kernel's own logic needs,
where HIP's native multi-axis form gets each dimension directly from its own independent hardware
register (`workitem.id.y`, no arithmetic reconstruction required). This is a real, concrete extra
source of per-thread VGPR pressure that `occupancy_from_ll_idea.md`'s divergence-tainted liveness
estimator (the sound one-directional occupancy upper bound described under hypothesis 7) would need
to actually count — the `sdiv`/`mul`/`sub` chain recovering `row`/`col` from the flat id is itself
built entirely from divergent values (the flat id is divergent), so a correct implementation of that
estimator already taints and counts these intermediate values by construction, provided it walks
through this idiom rather than special-casing it away — but this was not specifically
verified against a `collapse()` fixture in `occupancy_from_ll_idea.md`'s own testing (its three test
kernels were all single-axis, non-collapsed). Worth flagging as a concrete, motivated fixture gap for
whoever validates that estimator against real OpenMP-target code: a `collapse()`d kernel is a real,
now-confirmed-common case where OpenMP's own index-recovery overhead, not just the kernel's "real"
computation, is a first-order contributor to register pressure and thus occupancy — exactly the kind
of case that estimator should be re-validated against before being trusted for OpenMP-target kernels
specifically (its existing 2-for-2 validation was HIP- and non-collapsed-OpenMP-shaped only).

And, unchanged from before: `_resolve_scalar_value()` itself was read, not modified or run against a
`GPU_THREAD_INDEX`-tagged call in this pass — the change described above (for both the 2-D and
`collapse()` cases) is real, scoped, and small, but still unverified by an actual code change.

## `private`/`firstprivate` and `reduction`: how OpenMP's data-sharing clauses show up in `.ll`

Two follow-up questions, both checked directly rather than reasoned about abstractly.

### `private`/`firstprivate`: both are ordinary per-thread storage — but taint must follow the *value*, not the *storage*

Compiled a real fixture with both clauses (`firstprivate(bias)` on a host-declared scalar, plus an
ordinary implicitly-firstprivate scalar parameter `scale`, and `private(tmp)` on a second kernel) and
inspected the device `.ll` directly. Both `bias`/`scale` (firstprivate) and `tmp` (private) become
ordinary `addrspace(5)` (private/per-thread) allocas — structurally identical to any other
loop-local scalar, one independent copy per lane, nothing new. The real distinction that matters for
divergence-tainting isn't visible in *where* the value is stored — it's in what gets written there:

- `bias`/`scale`: their alloca is written **once**, at kernel entry, from an ordinary scalar kernel
  argument (`store i64 %4, ptr %23` — the same argument-passing mechanism as any other by-value
  parameter, confirming firstprivate scalars need no special extraction path at all, just the existing
  kernel-argument handling). That argument is uniform (same value broadcast to every lane at dispatch
  time), so every lane's private copy holds the *same* value — the taint analysis correctly leaves it
  untainted, provided it follows the value stored through the alloca rather than assuming
  "`addrspace(5)` ⇒ per-thread ⇒ divergent."
- `tmp`: confirmed by direct trace — `%51 = mul %50(tainted, loaded from `i`), 2` gets stored into
  `tmp`'s own alloca (`%20`), and the later `%57 = load i32, ptr %20` picks up that taint correctly
  through the ordinary store→load dataflow the walker already resolves for any address-taken local.

**The design conclusion is a precise, useful correction to a plausible-but-wrong shortcut**: a naive
rule ("anything in an `addrspace(5)` alloca is per-thread, therefore divergent") would be *safe* in
the one-directional-bound sense (over-tainting only ever makes the estimate more conservative, never
wrong in the unsafe direction) but needlessly imprecise — it would inflate the peak-live-VGPR-slot
count for every `firstprivate` scalar a kernel happens to have, even though such values are exactly
the SGPR-eligible, register-cheap case in reality. The correct rule — already how this document's
taint mechanism is specified everywhere else — is to taint *values* via their real dataflow (through
loads/stores, exactly like the walker's existing alloca-resolution fallback for any address-taken CPU
local), never *locations*. No new mechanism needed here, only discipline in applying the existing one.

### `reduction`: the per-thread phase is ordinary; the cross-thread combine is entirely opaque

Compiled `#pragma omp target teams distribute parallel for reduction(+:sum)` and inspected the device
`.ll`. Two structurally distinct phases, confirmed directly:

1. **Per-thread partial accumulation — no new mechanism needed at all.** `sum`'s running partial lives
   in an ordinary private alloca (`%32`), initialized to the reduction's identity value (`0.0`), updated
   each iteration by a plain `fadd`/`store` pair inside the grid-stride loop. This is *exactly* the
   existing CPU `reduce_residual` worked example already in this project's own test fixtures
   (`build_reduce_residual_kernel()`, `local_sum += fabs(local[i])`) — an ordinary loop-carried scalar
   accumulator, already fully handled by the walker's existing address-taken-local resolution path.
2. **Cross-thread/cross-team combine — confirmed entirely opaque, a real, new gap.** After the loop,
   the per-thread final partial is handed to exactly **one** call:
   ```llvm
   call void @__kmpc_xteamr_d_16x64(double %74, ptr %31, ptr %72, ptr %73,
       ptr @__kmpc_rfun_sum_d, ptr @__kmpc_rfun_sum_lds_d, double 0.000000e+00, i64 %49, i32 %48, i32 1)
   ```
   `%72`/`%73` are runtime-managed cross-team scratch buffers (not a `.ll`-visible `addrspace(3)`
   global at all); `__kmpc_rfun_sum_d`/`__kmpc_rfun_sum_lds_d` are the actual combine-op callbacks,
   passed by *pointer* — and confirmed to be bare `declare`s in this module, no body available to
   inspect either. **No LDS global, no `atomicrmw`, no barrier appears anywhere in this `.ll` for the
   reduction's real cross-thread work** — every bit of it lives inside AMD's own OpenMP device-runtime
   library, resolved only at final link time, structurally invisible to a `.ll`-only tool. This is the
   same *shape* of gap `occupancy_from_ll_idea.md` already found for register/occupancy accounting
   when a kernel calls into external runtime code (an unresolved symbolic formula instead of a number)
   — here for memory-traffic/timing cost instead of registers.

**What's still free, even though the real cost isn't**: the callback names themselves are a cheap,
exact, by-name-readable fact — `__kmpc_rfun_<op>_<dtype>` (here: `sum`, `d`=double) tells you exactly
which reduction operator and dtype are involved, no backend needed, matching this whole exploration's
recurring "the name alone is a free fact" pattern. The recognition rule this needs is exactly the
`OPAQUE_CALL_STL`/`OPAQUE_CALL_MPI` precedent: recognize `__kmpc_xteamr_*`/`__kmpc_rfun_*` **by name**
as "a real, semantically-known reduction combine step," distinct from a genuinely-unknown opaque call
— but still honestly report its cost as **not computable from static analysis alone**, not silently
priced as zero or as an ordinary cheap accessor call.

**A real, honest retro-compatibility flag, directly relevant to this project's own ROCm 7.0.2
compatibility requirement**: this exact `__kmpc_xteamr_*` shape is what ROCm 7.2.4's toolchain (the
only version installed on this machine) produces. OpenMP-target reduction lowering is known to have
changed across LLVM versions — older toolchains have historically lowered `reduction()` into a more
directly-visible LDS-tree-plus-atomic sequence in the caller's own IR, a genuinely different shape
from one opaque call. **Not tested against ROCm 7.0.2 here** (not installed on this machine) — whoever
builds this recognition rule should not assume the `__kmpc_xteamr_*` shape is universal across every
ROCm version this project commits to supporting; it may need a second recognition path for whichever
shape an older toolchain actually produces.

**HIP has no equivalent gap, by construction, not by luck.** HIP has no built-in `reduction()`
clause — any HIP reduction is hand-written by the developer using `__shared__` + `__syncthreads()` +
explicit tree-reduce arithmetic, all of which are already fully covered by Q1 (LDS) and ordinary
instruction walking, with no compiler-synthesized opaque runtime call involved at all. The opacity
gap here is specifically an OpenMP-runtime-lowering artifact, not a general "reductions are hard"
problem.

## Atomics: a deliberate scope decision — report the static fact, don't model contention

A real atomic operation's actual slowdown depends on how many threads genuinely collide on the same
address at runtime — a property no static tool can see, and one where most real algorithms only ever
have a handful of colliding threads at once (rarely a real problem in practice). Modeling that
contention-dependent cost is out of scope for this tool; the decision made here is that Stage 1/3
should surface the static, always-true fact — **how many bytes of atomic access a kernel performs,
and where** — and leave judging whether that's actually a problem to the person reading the report,
exactly the same "flag a fact, don't predict runtime behavior" posture this whole exploration keeps
converging on, just decided explicitly here rather than reasoned into as a byproduct of a harder
question. This directly resolves a gap 5.3's own docs already flagged as deferred: "GPU-target
OpenMP/HIP device-side atomics — out of scope for this CPU-target walker; flagged for whoever picks up
the GPU-side plan."

**Confirmed: this is mostly a reporting gap, not an extraction gap — the CPU walker already resolves
atomics correctly, it just can't distinguish them from ordinary accesses afterward.**
`_atomicrmw_to_accesses()` (`stage1_ir_walker.py`, l.1477-1511) already exists, already handles both a
GEP-indexed atomic (`hist[i % 4] += ...`) and a bare-pointer one (`*total += ...`) via the exact same
`_gep_to_access`/`_resolve_base_symbol` machinery every ordinary access uses, and already produces a
correct `Access` pair (one `READ`, one `WRITE`) so the real memory traffic an atomic RMW performs
isn't undercounted. **But neither `Access` carries any marker that it came from an `atomicrmw`** —
today, an atomic byte and an ordinary byte are indistinguishable once counted, so there's no way to
report "X bytes of atomic write" as its own line the way this section asks for. The fix is small and
additive: an `is_atomic: bool = False` field on `Access`, matching the schema's existing pattern for
this kind of tag (`Role`, `SymbolOrigin`) — Stage 3/6 can then sum atomic and non-atomic bytes
separately without touching the extraction logic that already works.

**Confirmed empirically: real GPU atomics are plain, visible instructions for both HIP and OpenMP —
genuinely different from `reduction()`'s opacity, not the same gap under another name.** Compiled a
HIP kernel using `atomicAdd` on both an LDS location and a global one, and an OpenMP kernel using a
*per-statement* `#pragma omp atomic` (not the whole-loop `reduction()` clause from above) — both
produce an ordinary, directly visible `atomicrmw` instruction in the kernel's own IR:
```llvm
; HIP, atomicAdd into __shared__ memory:
%16 = atomicrmw fadd ptr %13, double %15 syncscope("agent") monotonic, align 8

; OpenMP, #pragma omp atomic (a single statement, not the reduction() clause):
%57 = atomicrmw fadd ptr %56, double 1.000000e+00 monotonic, align 8
```
**A real, useful contrast worth stating plainly**: OpenMP's two reduction-shaped constructs get
completely different treatment from the same compiler — the whole-loop `reduction()` clause lowers to
an opaque runtime call (previous section), while a per-statement `#pragma omp atomic` lowers to a
plain, fully-walkable instruction. A developer reaching for either construct to express "combine into
one shared location" gets very different static visibility depending purely on which spelling they
used, not on what the code actually does.

**One address-space subtlety, matching Q1's own resolution pattern rather than needing a new one**:
an `atomicrmw`'s pointer operand doesn't always show its true address space directly in the printed
instruction (HIP's LDS atomic above prints as a bare `ptr %13`, no `addrspace(3)` qualifier visible at
that instruction) — the same `addrspacecast`-to-generic pattern Q1 already found for ordinary
loads/stores applies here too, and the same base-symbol-resolution tracing `_resolve_base_symbol`
already does to classify an ordinary access as `SHARED_LOCAL`/`GLOBAL` would resolve an atomic's true
scope the same way, with no new mechanism needed — just applying `is_atomic` and the existing
`MemorySpace` classification to the same `Access` object. That scope distinction (LDS-contained,
workgroup-sized contention domain vs. global-memory, grid-wide domain) is itself a free, cheap,
worthwhile fact to surface alongside the byte count — not a contention *model*, just a size-of-the-blast-
radius classification a human can use to judge risk faster. The memory-ordering qualifier
(`monotonic`/`acquire`/`release`/`acq_rel`/`seq_cst`) and `syncscope` are equally free and equally
worth reporting (a stronger ordering is real, uncontroversial extra hardware cost) — but, consistent
with the scope decision above, only as reported facts, never as an attempted performance-delta model.

## Working through the seven hypotheses

**1. Per-iteration CPU model → per-thread GPU model (no kernel-internal loop form, 1 thread = 1
iteration).** Maps directly onto the existing schema with no new fields: a GPU kernel becomes one
`Region` whose `LoopDim.var` is the recognized flattened thread-index symbol from Q3 above,
`parallel_kind=ParallelKind.GPU_THREAD` (already defined, never assigned), `lower=0`,
`upper=grid_dim*block_dim` (a `LaunchConfig`-derived sympy expression — a free parameter when the
launch site isn't in the same `.ll`, exactly like an unresolved CPU trip count already is). 5.4's
whole branch/sequence/inner-loop decomposition machinery (hypothesis 3, below) then applies to this
one `Region`'s body completely unchanged.

**2. No CPU/GPU aggregation, ever.** This is a Stage 3/4/6 constraint, not a Stage 1/2 one: Stage 4
needs a distinct `backend_gpu.py` + GPU machine-spec schema (arch string → CU count, wavefront size,
peak FLOPs/wave, HBM bandwidth, LDS capacity/bandwidth — all keyed off the already-free `target-cpu`
string per `occupancy_from_ll_idea.md`), selected per `Kernel.target_kind` and never mixed into the
same roofline denominator as a CPU spec. Stage 6's report composition needs to keep CPU time, GPU
kernel time, and host↔device transfer time (see next point) as three separate, clearly-labeled
totals — never summed into one number.

**3. A GPU kernel is its own function, subdividable by branches/inner loops.** This is 5.4's existing
CFG-decomposition machinery (`SuccessorKind`, branch-arm handling, the function-summary kernel)
applied to a device-kernel function's body — no new decomposition logic needed, only a new *entry
point*: `_is_candidate_function()` needs to also admit `amdgpu_kernel`-calling-convention functions
and `__omp_offloading_<hash>_<hash>_<name>_l<line>`-named functions. Worth noting, **not yet
verified**: both kinds of function carry real DWARF debug info pointing back to the original
`.c`/`.cpp` source file and line (confirmed present in this exploration's own device `.ll` output),
which is exactly what `_is_candidate_function()`'s existing basename-matching filter already keys on
for CPU functions — it's plausible this filter admits GPU-kernel functions with no change at all,
once the module can even load (see the `initialize_all_targets()` gap below). Not tested against the
real function in this pass.

**Structural decomposition (splitting a kernel into branch arms) reuses 5.4 unchanged — but the COST
policy across those arms genuinely can't, and a real, surprising discovery here narrows the actual
gap to something much smaller than expected.** A CPU branch and a GPU branch aren't the same physical
event: a wavefront executing SIMT-in-lockstep can't take two different paths per lane the way a CPU
core can — when lanes within a wavefront disagree on a branch condition, the hardware executes
**every taken arm serially** (masking off the lanes that didn't choose it), so the wavefront pays the
**sum** of every taken arm's cost, not the cost of whichever single arm "the branch" resolves to.
A *uniform* branch (every lane agrees) behaves exactly like an ordinary CPU branch — only the one
taken arm's cost applies, for the whole wavefront.

Confirmed by compiling a real three-branch fixture (a genuinely divergent bounds check
`if (i < n)`, a genuinely divergent even/odd split `if (i % 2 == 0)`, and a genuinely uniform
argument-gated branch `if (mode > 0)`, `i` built from `blockIdx`/`threadIdx` as in Q3): **nothing in
the frontend `.ll` distinguishes a divergent branch from a uniform one at all** — all three compile to
an identical, ordinary `br i1 %cond, label %A, label %B` with plain debug metadata, no special
attribute, no different instruction. This has to be *derived*, and the mechanism to derive it already
exists: the exact same divergence-tainting analysis `occupancy_from_ll_idea.md`'s VGPR
lower-bound estimator already built and validated (seed = any value reachable from a recognized
thread-index primitive, forward-propagate through operand chains to a fixed point) correctly
classifies all three branches in this fixture when applied to each branch's own condition value: `i<n`
and `i%2==0` are both tainted (built from `threadIdx`-derived `i`) → divergent; `mode>0` is untainted
(`mode` is a plain kernel argument, no thread-index dependency anywhere in its def chain) → uniform.
One and the same taint mechanism, now serving a *third* purpose across this exploration (coalescing
in Q3, register pressure in hypothesis 7, and branch-cost classification here).

**The genuinely surprising part, found by reading the real aggregation code rather than assuming its
shape — and worth being precise about, since the more common "worst case" reading (pick whichever
single arm needs more memory operations, a standard WCET-style convention) is a completely reasonable
expectation that this code does NOT implement.** Confirmed directly in the literal *code*, not just
comments, in both places arms get combined: `_combine_op_counts_by_policy()` (`stage3_aggregate.py`,
~l.137-149) returns `counts_list[best_index]` — one arm, picked — for `BEST_CASE`, but for anything
else (which `_require_worst_or_best_case()` restricts to `WORST_CASE`) returns
`_sum_op_counts(counts_list)` — every arm, added together, unconditionally. `_combine_reuse_by_policy()`
(~l.344-365) is the identical shape for bytes: `BEST_CASE` returns `reuse_results[best_index]` whole;
`WORST_CASE` loops over **every** arm's `ReuseResult` and adds each level's bytes into a running total
(`loaded[level] = loaded.get(level, 0) + n`) — no max, no comparison, anywhere in that path. So
`WORST_CASE` is genuinely defined as "every arm's cost, summed," not "whichever arm costs more" — a
real, deliberate asymmetry with `BEST_CASE` (which *does* pick one arm, by bytes), not a
misreading. `_best_arm_index()`'s own docstring gives a plausible reason a single "worse" arm was
never picked the same way: ops and bytes could disagree on which arm looks worse (one arm pricier in
FLOPs, a different one pricier in bytes moved), and summing sidesteps ever needing a single,
possibly-inconsistent ranking for the "worse" direction — `BEST_CASE` already accepts that same
awkwardness in the other direction, arbitrarily breaking ties by bytes alone.

**This creates a real, worth-naming tension, not just an incidental fact**: if `WORST_CASE` is ever
changed toward the more intuitive single-worst-arm reading — a completely legitimate thing to want
purely on CPU-modeling-accuracy grounds, independent of anything GPU-related — that change would
silently make the GPU story below **wrong**, specifically an *underestimate*: a genuinely divergent
wavefront pays for every taken arm's cost, not just whichever single one looks pricier by one metric.
Whatever the original CPU rationale for choosing sum-over-pick actually was, **the sum behavior *as
currently implemented* happens to be exactly the literal, physically correct cost for a genuinely
divergent GPU branch**, while the more intuitive pick-the-worse-arm alternative would NOT be.

**Update, live during this same exploration: a concurrent fix is correcting `WORST_CASE`'s CPU
semantics to the more intuitive pick-the-single-worse-arm reading** (exactly the direction the
preceding paragraph flagged as a legitimate, independent CPU-accuracy concern). That settles the
tension named above, but changes the concrete conclusion: **GPU divergence genuinely does need its
own, distinct policy after all** — not a repurposed `WORST_CASE`, since `WORST_CASE` will no longer
mean "sum every arm" once that fix lands. The actual, narrower gap is now:
1. A per-branch-point divergence classifier (the taint analysis above, newly wired into candidate
   branch points rather than only into register-footprint estimation).
2. **A new `BranchPolicy` value** (e.g. `DIVERGENT_SUM`) carrying exactly the sum-every-arm semantics
   `WORST_CASE` is being corrected away from. **The fix has since landed and all 352 tests pass**
   (confirmed directly, not assumed) — worth one small correction to the paragraph above now that
   the actual diff is known: `add_op_counts`/`add_reuse` (generic dict-merge utilities used all over
   this codebase for ordinary sequential-piece summing) do stay, but the *specific* per-level
   summing loop inside `_combine_reuse_by_policy()` was removed outright, not left dormant — that
   function (and its op-counts sibling, `_combine_op_counts_by_policy()`) now unconditionally
   `return results[best_index]` for both `BEST_CASE` and `WORST_CASE`. So a real `DIVERGENT_SUM`
   would need that per-level summing loop *re-added* as its own small combination function, not
   merely "re-wired" — the concept is simple to restore (it's exactly what was just deleted, visible
   in the fix's own diff), but it doesn't currently exist anywhere in the codebase to dispatch to.
   Confirmed empirically, not just structurally, that the tied-arm fixtures this document's own
   earlier tests relied on (`branch_if_else`, `nested_branch`) can't actually distinguish "picks the
   pricier arm" from "picks arbitrarily" or even "still secretly sums" — a genuine gap in the fix's
   own test coverage until a new test against the one *asymmetric*-cost fixture already in the repo
   (`branch_ops_vs_bytes` — 7×FP_MUL/6656 bytes vs. 3×FP_ADD/16640 bytes) confirmed `WORST_CASE` now
   resolves to the FP_ADD arm (more bytes, fewer ops) rather than the FP_MUL one — the real,
   distinguishing proof, now a permanent regression test
   (`BestAndWorstCaseArmSelectionConsistencyTests.test_worst_case_ops_and_bytes_come_from_the_same_pricier_arm`).
   Arguably the cleaner outcome of the two anyway: a GPU divergent branch and a CPU worst-case
   branch-prediction assumption were never really the same concept sharing one enum value by design
   — they only happened to coincide before the fix — so giving each its own name removes exactly the
   kind of landmine (one policy silently meaning two different physical things depending on target)
   the earlier draft of this section was worried about happening *by accident* later.
3. A way to force the new `DIVERGENT_SUM` policy specifically for a branch point classified as
   divergent, while still honoring whatever policy the user configured for the *rest* of the
   kernel's (uniform) branches — a real architectural gap either way, since `Assumptions.branch_policy`
   today is a single, whole-scope setting (one policy/probability for everything `aggregate_scope()`
   walks), not something resolved independently per branch point. This part of the finding is
   unaffected by which policy ends up carrying the sum semantics: `BEST_CASE`, any probability
   policy, and `KEEP_SEPARATE` all structurally assume "exactly one arm really executes" (confirmed
   in `stage2_assumptions.py`'s probability-sums-to-1 validation), which is categorically **wrong**,
   not just less precise, for a genuinely divergent branch — only the new sum-shaped policy is
   physically meaningful there.
4. `stage3_aggregate_branch.py`'s own existing limitation that `KEEP_SEPARATE`/probability policies
   only support a scope with exactly one branch point (falling back to `WORST_CASE` — now
   pick-the-worse-arm, not sum — with a warning otherwise, l.12-17/152-159) becomes directly relevant
   for GPU kernels with multiple divergent branch points (this exploration's own three-branch fixture
   has two independent `if`s, both boundary-check-shaped, exactly the pattern a ported CPU stencil
   produces): that CPU-side fallback is the wrong target for a divergent branch post-fix, reinforcing
   that GPU multi-branch resolution needs its own path to `DIVERGENT_SUM`, not a shared fallback with
   CPU's own (now different) `WORST_CASE`.

**Sharpening point 3 into an actual decision rule, not just "force it for divergent branches"**: a
*uniform* GPU branch isn't a special case needing GPU-specific handling at all — every lane in the
wavefront evaluates the same condition and agrees, so exactly one arm executes for the whole
wavefront, the same as a single CPU thread would, just SIMD-replicated across the wavefront's 32/64
lanes. So the real per-branch-point rule is a three-way classification, not a two-way one:
- **Divergent** (condition provably tainted) → force `DIVERGENT_SUM`, regardless of the kernel's
  configured `Assumptions.branch_policy`. Not optional — as established above, every other policy
  structurally assumes exactly one arm executes, which is false here.
- **Unclear/unknown** (the taint analysis can't resolve the condition at all — an opaque call, an
  unresolved indirect load) → **also** defaults to `DIVERGENT_SUM`, for the same one-directional-bound
  reason as the rest of this section: assuming *uniform* when the branch might actually diverge would
  be an unsafe underestimate (missing a real, unbounded-in-this-analysis cost); assuming *divergent*
  when it's actually uniform is merely a safe overestimate. The unclear case must round toward the
  same side as the confirmed-divergent case, not toward the cheaper default.
- **Uniform** (condition provably untainted) → apply the kernel's configured `Assumptions.branch_policy`
  completely unchanged, exactly as for CPU — `WORST_CASE`/`BEST_CASE`/probability/`KEEP_SEPARATE` all
  keep their ordinary, now-post-fix meanings, since a uniform branch really is just one thread's
  control flow, replicated identically across every lane.

**A real, additional idea for `KEEP_SEPARATE` specifically, worth building rather than leaving as a
footnote**: today `KEEP_SEPARATE` returns one row per arm — genuinely correct and useful once a
branch is confirmed uniform (each row is a real, individually-meaningful cost), but for a branch that
*might* diverge, a GPU-aware `KEEP_SEPARATE` could return those same per-arm rows **plus one extra
`DIVERGENT_SUM` row**, rather than forcing a single static guess either way. That gives whoever reads
the report the full decision tree in one place: "if this branch turns out to behave uniformly at
runtime, here is each arm's own cost; if it actually diverges, here is what that costs instead" — letting
a human (or a downstream automated comparison) pick the right row once they know which regime actually
applies, rather than the tool silently committing to one. This is a genuinely strong fit for exactly
the "annotate a real measurement with a cheap static fact" pattern this whole document series keeps
converging on (`occupancy_from_ll_idea.md`'s LDS/launch-bound facts sitting next to `rocprof-compute`'s
real measured occupancy): `docs/rocprof_compute_counters_reference.md` already documents a real,
existing measured counter for exactly this question — **VALU Active Threads** ("the average level of
divergence within a wavefront... the number of work-items that were active in a wavefront during
execution of each VALU instruction, time-averaged over all VALU instructions run," l.195/428, flagged
as directly measuring "branch/lane divergence" at l.656). A `KEEP_SEPARATE` report carrying both the
per-arm rows and the `DIVERGENT_SUM` row, sitting next to a real profiling run's own **VALU Active
Threads** value, lets a user see directly whether the kernel's real behavior tracked the "uniform"
side (VALU Active Threads ≈ wavefront width — the per-arm rows were the right ones to trust) or the
"divergent" side (VALU Active Threads well below wavefront width — the `DIVERGENT_SUM` row was) for
their actual data, rather than the static tool ever needing to guess which regime applies on its own.

**This is the same one-directional-bound epistemic shape found repeatedly in this exploration now —
worth naming as a recurring pattern, not a coincidence.** Divergence tainting is sound ("this branch
*could* diverge, syntactically") but not complete ("this branch necessarily *does* diverge, for every
real launch") — a condition built from a thread-index value can still happen to be uniform in
practice for specific data (e.g. `threadIdx.x < blockDim.x`, always true, tainted but never actually
divergent). Forcing `DIVERGENT_SUM` on anything merely *classified* as divergent — and, sharper still,
on anything merely *unclassifiable* — is therefore itself a safe over-estimate, never an
under-estimate: the identical shape as `occupancy_from_ll_idea.md`'s VGPR lower bound (sound,
one-directional, "flag trouble, never prove safety") and this document's own USM-inference idea (an
inferred transfer is a real signal, not a certainty). Multiple independent parts of this exploration
have now converged on the same epistemic posture for GPU-specific static analysis: safe pessimism
where runtime data can't be seen, never a false "this is fine" — with the `KEEP_SEPARATE` extension
above as the one place that turns that pessimism into an explicit, both-sides-shown choice for a human
(or a real measurement) to resolve, rather than a silent default.

**4. No stream modeling — one kernel active at a time.** Falls out of the host-side call sequence
already visible in `.ll` (Q2): the order of `hipLaunchKernel`/`__tgt_target_kernel`/`hipMemcpy`/
`__tgt_target_data_*` calls in one host function is exactly the kind of straight-line sequence 5.4's
`SuccessorKind.SEQUENCE` already models for CPU code — generalizes directly, no new concept, as long
as multi-stream/async launch APIs (`hipLaunchKernelGGL` on a non-default `hipStream_t`,
`hipMemcpyAsync`) are explicitly treated as ordinary synchronous ones for modeling purposes (a
documented simplification, not a silent one).

**5. Dynamic grid size assumed large enough to reach BW/compute-bound.** This licenses *ignoring*
grid size (occupancy from insufficient parallelism/tail effects) and focusing entirely on **per-
thread** resource pressure (registers, LDS) as the only occupancy limiter worth modeling — which is
exactly what hypothesis 7 and `occupancy_from_ll_idea.md`'s addendum already narrow down to. No new
finding here beyond confirming the hypothesis is consistent with that prior work's scope.

**6. Support at least HIP and OpenMP offloading.** Confirmed both compile to the same underlying
`amdgpu_kernel` device shape (calling convention, `addrspace` scheme, LDS mechanism) with
language-specific differences that are each individually small and now concretely characterized
above: different thread-index call vocabularies (Q3), different host-side launch/transfer call
shapes (Q2), a default work-group-size ceiling that differs (1024 vs. 256, per
`occupancy_from_ll_idea.md`), and OpenMP's extra `__kmpc_alloc_shared` false-positive risk (Q1). No
blocker found for either language; `GPUVendor`'s existing AMD/NVIDIA split (reserved, unused) means
adding NVPTX (`llvm.nvvm.read.ptx.sreg.tid.*`, `ptx_kernel` calling convention) later is architecturally
free but genuinely untested here — no CUDA toolchain available on this machine to confirm against.

**A real third backend, checked directly since a fixture already exists in `test_apps/cpp_app`:
C++ stdpar (`std::execution::par`/`par_unseq`), via `amdclang++ --hipstdpar`.** Compiled
`test_apps/cpp_app/kernel_stdpar.cpp`'s real `std::for_each(std::execution::par_unseq, ...)` fixture
at Stage 1's exact contract. Confirmed: **it genuinely is the same backend**, not merely
vendor-compatible — `--hipstdpar` redirects the standard-library call, via `libstdc++`'s own SFINAE-gated
PSTL hook (`hipstd::is_offloadable_iterator`/`is_offloadable_callable`), straight into **rocThrust**
(`thrust::...::hip_rocprim::__parallel_for::kernel<256U, for_each_f<...>>`), which compiles to a real
`amdgpu_kernel` function using the *exact same* `__ockl_get_group_id`/`__ockl_get_local_id` call
vocabulary already found for hand-written HIP — no new thread-index recognition needed at all.
`--offload-device-only` also behaves identically to plain `hipcc` (a single, clean device-only module,
re-tested directly), so nothing new needed there either.

**The outer `amdgpu_kernel` function's own debug info is unusable — but this turns out to matter
less than first thought, confirmed by actually running the real, unmodified code.** The entry
kernel's `!DISubprogram`/`!DIFile` resolves to a ROCm-*installed library header*
(`.../thrust/system/hip/detail/parallel_for.h`), **not** `kernel_stdpar.cpp` — confirmed directly, and
`_is_candidate_function()`'s basename filter, applied at that level, would correctly reject it as
"not user code." Original assumption here was that this forces a call-graph-forward-tracing step into
candidate *selection itself*. **Tested directly against the real, unmodified
`stage1_frontend_llvm_ir.py`/`stage1_ir_walker.py`** (working around only the already-documented
`initialize_all_targets()` gap, plus one incidental new one below) — that assumption was wrong, for a
simpler reason: `iter_candidate_kernels()` doesn't walk the call graph from one entry point at all; it
flatly scans **every** function `module.functions` contains and filters each one independently by its
*own* debug info. The user's lambda (`_ZZ20launch_stdpar_kernelPdiENKUliE_clEi`) is a real, standalone
function definition (not inlined away at `-O0`) whose own `!DISubprogram` correctly names
`kernel_stdpar.cpp:20` — so it passes `_is_candidate_function()`'s existing filter **unmodified**, and
`walk_function()` produces a real `Kernel` for it via exactly the 5.4 straight-line/no-loop mechanism
(confirmed output: `target_kind=TargetKind.CPU_LOOP` (hardcoded, as expected), `loop_nest=0`, plus the
usual synthetic function-summary companion) — with zero code changes and no call-graph tracing
involved. The three-call-level nesting is real, but irrelevant to *discovery*, since discovery never
starts from the entry point in the first place. (It would still matter for anything that specifically
needs the *entry kernel's own* identity — e.g. tying a `LaunchConfig`/occupancy fact to "which
`amdgpu_kernel` dispatch is this," which does still need the outer function, found separately by
calling-convention/vocabulary recognition as already described, not by tracing from the lambda inward.)

**Getting this far exposed a real, concrete, pre-existing bug — not GPU-specific at all.** The
lambda's Python-visible accesses (via a throwaway test script that ran the real walker end to end)
came back as:
```
symbols=['arg0', 'arg1', 'arg0_target']            # only 3 symbols, not the expected 4
access: arg0        READ  index_expr=1              # loading `snapshot` (closure struct field 1)
access: arg0_target READ  index_expr=arg1-1         # snapshot[i-1]
access: arg0        READ  index_expr=1              # loading `snapshot` again
access: arg0_target READ  index_expr=arg1+1         # snapshot[i+1]
access: arg0        READ  index_expr=0              # loading `array` (closure struct field 0)
access: arg0_target WRITE index_expr=arg1           # array[i]
```
The lambda captures **two** pointers by value (`array`, `snapshot` — a read-only input and a
write-only output, genuinely different memory), compiled into one synthesized closure-struct
argument. Both loads resolve to the *same* pointee symbol, `arg0_target` — confirmed by reading
`get_or_create_pointer_target_symbol()` (`stage1_ir_walker.py:751-766`): its cache key is
`f"_target_of_{pointer_symbol.name}"`, keyed purely on the *base* symbol's own name, with no
dependence on which struct field/GEP offset produced the load. Since both pointers resolve back to
the same base argument symbol (`arg0`, the whole closure struct — its own two loads are modeled as two
different `index_expr` values, `0`/`1`, on that one symbol, rather than as two distinct pointers), the
second `get_or_create_pointer_target_symbol()` call hits the identical cache entry and returns the
*same* `Symbol` object — genuinely merging a read-only source and a write-only destination into one.
This is **not** a stdpar- or GPU-specific gap: nothing here depends on `amdgpu_kernel`,
`addrspace`, or any thread-index primitive — it's a latent gap in today's CPU walker, just one that an
ordinary CPU function rarely triggers (two separate pointer *parameters* wouldn't hit it; a closure
capturing two pointers by value, compiled into one synthesized struct argument, is exactly the shape
that does). Not GPU-specific, so **fixed directly, independently of any GPU work**: confirmed
by tracing `_gep_own_byte_offset()`/`_walk_gep_chain()` (`stage1_ir_walker.py`), a STRUCT-member
GEP descent is already classified separately from an ARRAY-element descent (`_descend_type()`'s own
`"struct"`/`"array"` distinction) — a struct member index is always a compile-time-constant,
heterogeneous field selection (genuinely different memory per field), while an array index is
usually a runtime value indexing into homogeneous elements of the *same* logical data (the jagged
`m[i]` row-pointer case this same cache key already handles correctly, and must go on handling
correctly — collapsing to one shared target symbol regardless of which row `i` names). Threaded that
existing struct/array classification (already computed, just not returned) through as a new
`struct_member_path` element on `_gep_own_byte_offset()`'s/`_walk_gep_chain()`'s return tuples, and
folded it into `get_or_create_pointer_target_symbol()`'s cache key only when non-empty — leaving the
jagged-matrix path's key completely unchanged (its GEP never produces a struct descent). Re-ran the
same reproduction: `array`/`snapshot` now resolve to distinct symbols
(`arg0_target_field0`/`arg0_target_field1`). All 350 existing tests in `parsing/tests/` still pass,
including the jagged-matrix fixture's own `test_row_pointer_and_target_are_distinct_symbols` — no
regression on the case this fix had to be careful not to break.

**One more incidental, real parsing gap found getting this far**: the stdpar module's `.ll` also
declares `@blockIdx`/`@threadIdx` as `extern_weak dso_local protected addrspace(1) global ... poison`
— real, unused HIP-builtin-compatibility declarations Clang emits regardless of whether the code
actually references `blockIdx`/`threadIdx` by name (this fixture doesn't; it goes through
`__ockl_get_group_id`/`get_local_id` instead, per the vocabulary already found above).
llvmlite's `parse_assembly()` (bundled LLVM 22, matching the compiler's own version) **rejects this
exact combination outright** as a syntax error (`extern_weak` combined with an initializer) even
though it's valid, real output from the same-major-version `clang`/`opt` toolchain — a real,
narrow, textual round-trip gap between what Clang emits and what llvmlite's parser accepts, confirmed
by direct test, unrelated to any of the semantic gaps above. Stripping those two declaration lines
was enough to unblock parsing; a real Stage 1 frontend would need the same (or an upstream
llvmlite/LLVM-version fix) before it could load a real stdpar-compiled module at all.

This also sharpens Q3's coalescing/index-resolution machinery, in the same direction: the per-element
index isn't a bare arithmetic combination of `group_id`/`local_id` at its point of use — it's threaded
through a real C++ iterator object (`__gnu_cxx::__normal_iterator<int*, vector<int>>`), requiring
resolution through several small, genuinely-uninlined-at-`-O0` accessor calls (`operator+`, `operator*`,
`thrust::raw_reference_cast`) before reaching the lambda's own parameter. This is the same *shape* of
problem the walker's existing pointer-induction machinery
(`_recover_pointer_loop_dim()`/`_find_pointer_induction_phi()`) already handles for a CPU range-for,
and the specific accessor calls involved are the *exact* ones `OpKind.OPAQUE_CALL_STL`'s own docstring
already names ("`__gnu_cxx::__normal_iterator`'s own methods") — but `OPAQUE_CALL_STL` today is a
**cost** classification (price this call as cheap), not a **value**-resolution rule; recovering the
real index for loop-dim/coalescing purposes needs either adding `inline` to the normalization pass
list so these accessors collapse away before the walk, or teaching `_resolve_index_operand()` to
trace through this specific, small, **Update, corrected after actually testing `inline` against the
`collapse()` fixture** (see `occupancy_from_ll_idea.md`'s follow-up section): `inline` turned out to
be neither sufficient (OpenMP's own thread-index wrapper has no body to inline at all) nor necessary
(by-name recognition works for HIP's wrapper without inlining it either) for the *thread-index*
by-name-recognition case above — the established, working pattern there is by-name recognition first,
not `inline`. Whether the *same* preference holds for stdpar's STL accessor calls specifically
(`operator+`/`operator*`/`raw_reference_cast` are more generic/templated than a fixed, small primitive
vocabulary) is a real, separate question, not yet tested either way — the choice here is still open,
just no longer leaning on a since-corrected justification.
already-named set of STL accessor calls the same way it already traces through a GEP chain. Not
resolved here — a real, concretely-scoped follow-up, not a new open-ended problem.

**7. Occupancy/register pressure modeling is necessary eventually.** Covered, in depth, by
`occupancy_from_ll_idea.md`'s addendum and two later follow-up passes: a design section ("Follow-up:
resolving the open decisions and specifying the exact model") and a validation pass that actually ran
the three concrete next steps that section proposed — every one of the original document's open
decisions is now resolved, concretely narrowed, or (in one case) genuinely corrected there, and the
exact model (inputs, normalization pipeline, taint seed, formula, output shape) is fully specified.
Four things worth surfacing here rather than only there:
1. **A real correction, not just a resolution**: `inline` — originally thought necessary for both this
   estimator and Q3's coalescing classifier — turned out to be neither sufficient (OpenMP's own
   thread-index wrapper has no body in the device `.ll` to inline at all, confirmed by direct test)
   nor necessary (by-name recognition works for HIP's wrapper without inlining it either). By-name
   recognition of a small, fixed vocabulary is now the primary mechanism for both languages, not
   `inline` — see this document's own Q3 text (already updated) and the other document's pipeline step
   2 for the corrected version.
2. **The `collapse()` taint-propagation question is now confirmed, not merely conjectured**: a direct
   manual trace through the real normalized IR shows ordinary operand-following taints the recovered
   `row`/`col`/`idx` values with zero special-case code, and — a bonus — the floor/mod round-trip
   `idx` involves reduces to the flat id by plain algebraic cancellation, an easier bar for sympy than
   originally worried about.
3. **The SGPR/control-divergence question was re-tested with a cleaner, nested fixture and refined
   into an actionable rule, not left as a bare negative**: `TotalNumSgprs` stays identical even under
   nesting, but VGPR count goes up by roughly one *per divergent branch condition* evaluated
   (replicated across both the flat and nested tests) — the liveness proxy should count a divergent
   branch's own condition value as a live tainted slot in its own right.
4. **The shared-infrastructure connection stands, corrected for (1)**: the divergence-tainting seed
   (now: by-name-recognized thread-index calls, not "after `inline`") is the same recognized-primitive
   concept Q3 needs for coalescing and `BranchPolicy.DIVERGENT_SUM` needs for branch-cost
   classification — one shared taint implementation, three independent consumers.

See the other document's follow-up section for the full detail — nothing here should be re-derived
independently of it.

## A real extension found while answering the above: infer USM transfers from cross-boundary "warmth," reusing an existing mechanism wholesale

The USM case above is unreadable *directly* — but the existing CPU "warmth" model
(`parsing/stage3/stage3_aggregate.py`, module docstring l.10-15) already solves a structurally
identical problem for a different boundary, and generalizes to this one with no new mechanism, only a
new place to apply it. Confirmed by reading the real code, not assumed: `stage3_aggregate.py`
maintains a `warm: Set[str]` of symbol names, threaded **in place** across a `Kernel.successors`
sequence (`_iter_pieces_with_branches()`/the branch-boundary handling around l.279-410) — a symbol
touched by an earlier piece in the sequence has its *next* touch cost zero bytes at whatever memory
level it's warm at, using exactly `Kernel.variable_interface()`'s existing `{symbol_name: {READ,
WRITE}}` view to know what each piece touched.

**A USM-backed transfer is the same shape of fact, one level up**: if a `hipMallocManaged` pointer is
touched by a CPU-side piece (a plain `TargetKind.CPU_LOOP`/function-summary kernel) immediately before
a `GPU_DEVICE_KERNEL`/`GPU_OFFLOAD_REGION` piece **in the same `Kernel.successors` sequence**, and that
GPU piece's own `variable_interface()` touches the same symbol name, that overlap — available today
with zero new extraction work, from machinery both Stage 1 (`variable_interface()`) and Stage 3
(`warm`/sequence-walking) already have — is a real, static, honest signal that an implicit H2D transfer
happens at that point, sized by the same byte-count expression already resolvable from the
`hipMallocManaged` call's own size argument (Q2, above — an ordinary SSA-resolvable value, identical to
`hipMemcpy`'s). The symmetric case (a GPU-side write followed by a later CPU-side read of the same
symbol, still in sequence) infers a D2H transfer the same way. Both should synthesize the same
`OpKind.OPAQUE_CALL_GPU_TRANSFER` `Operation` already proposed above for the *observed*
(`hipMemcpy`/`__tgt_target_kernel`) case — the honest difference is provenance, not shape: an inferred
transfer needs to be tagged as inferred (a new field, or reusing `SymbolOrigin`-style provenance
tagging) rather than silently presented as equally certain as an observed one, and must not
double-count a symbol an explicit `hipMemcpy`/map clause already accounts for.

Three honest limits on this, stated plainly rather than left implicit, all inherited from or matching
the scope of the mechanisms it reuses rather than new to this idea specifically:

- **Whole-symbol granularity, not byte-range.** `warm`/`variable_interface()` are both keyed by
  symbol name, not by intersecting `Access.index_expr` ranges — so this inherits exactly the same
  coarseness the CPU warmth model already has (a CPU touch of `x[0]` alone would still mark all of
  `x` as touched, same as today), not a new limitation introduced here. A sympy-range-intersection
  refinement is a real possible improvement, but isn't what today's warmth model does for CPU either,
  so it's parity, not a gap specific to the GPU case.
- **Same-function scope only, today.** This falls out for free only within one function's
  `Kernel.successors` sequence, matching 5.4's existing sequencing scope exactly. A CPU-side touch in
  a *different* function (e.g. a `setup()` that writes `x` before a separate `launch()` calls the
  kernel) is the same open cross-function call-argument-tracing problem
  `calltree_from_ll_idea.md`'s Use Case 3 already left unsolved — not newly solved by this idea.
- **A nice, free unifying consequence, not a separate mechanism**: a plain `hipMalloc`'d
  (device-only) pointer is never legally dereferenced by host code, so this same overlap check
  naturally produces *no* inferred transfer for it — the heuristic doesn't need to first classify
  "is this pointer USM-managed," that classification falls out of the overlap simply never firing for
  a non-USM device pointer. The interesting case (overlap fires, no explicit copy already accounts for
  it) is specifically the USM/managed-memory case this section is about.

## A concrete first deliverable: a standalone host↔device transfer table, deliberately outside the roofline model

Everything above (Q2, and the USM-warmth extension) established *what's readable*; this section
names *what to actually build first*, scoped down to something shippable without cross-function
analysis (still an open problem — `calltree_from_ll_idea.md`'s own Use Case 3 — not something this
idea depends on solving). Two things worth separating clearly:

**This is not roofline data, and shouldn't be folded into roofline numbers.** A roofline ceiling
answers "is this kernel's own execution compute-bound or bandwidth-bound." A host↔device transfer is
a different physical event entirely (PCIe/Infinity Fabric, not the GPU's own HBM/LDS/register
hierarchy) that happens *around* a kernel launch, not during it — and in real practice is very
often the actual bottleneck a roofline number alone would never surface (a kernel that's beautifully
compute-bound on paper can still be dominated end-to-end by an unnecessary or badly-overlapped
copy). This matches hypothesis 2's "never aggregate CPU and GPU" instinct one level further: transfer
time is a *third*, sibling category to CPU time and GPU-kernel time, not a component folded into
either — a standalone table, not a line item inside the existing bytes/ops totals Stage 3/4 already
compute.

**The table itself, scoped to what's honestly buildable today (per-function, no cross-function
tracing)**: one row per host↔device transfer, real or inferred, carrying at minimum symbol name,
direction (H2D/D2H), byte count (a resolved value or a free-param-bound sympy expression, same as
everywhere else in this schema), and a provenance tag distinguishing how it was determined —
because the three cases from Q2/the USM extension genuinely need different provenance handling, not
one shared confidence level:

1. **Explicit (`hipMemcpy`/OpenMP map clauses) — reported, not inferred.** Every observed transfer
   call becomes one exact row (direction + size both directly readable, per Q2). **A deliberate,
   stated v1 simplification for this case specifically**: a buffer touched by a GPU piece with *no*
   matching copy call anywhere in the visible (single-function) scope is assumed already correctly
   placed — reported as zero rows for it, not flagged as suspicious or inferred. This is a real,
   named assumption (an explicit-transfer codebase that happens to place data via a call outside the
   visible scope would be silently under-reported), but it's the honest, simple baseline for a
   language shape where the *whole point* is that the developer wrote the copy explicitly — no
   inference is attempted or warranted here, unlike the USM case below.
2. **Implicit (USM) — "normal" mode, the default.** Reuses the cross-boundary warmth-inference idea
   above exactly as already built: assume every USM-backed symbol is already correctly placed at the
   *start* of the analyzed scope (the same "trust the caller" assumption case 1 makes, just applied
   to the harder language shape where there's no call to observe at all), and report only the
   transfers the warmth overlap *within this one function* actually detects — genuinely the same
   mechanism, now given its proper name and default-mode status.
3. **Implicit (USM) — "expert" mode, opt-in.** For a user who actually knows where each variable
   really lives at scope entry (measured, or known from a larger program they've already reasoned
   about), an input file naming each variable's real starting placement (host/device) removes
   assumption (2) entirely and lets every USM-backed access get a real, non-inferred H2D/D2H row
   instead of only the intra-function-visible ones. This isn't a new kind of input mechanism for this
   project — it's the same shape as the `--param`/free-parameter-binding file this codebase already
   uses (referenced in `occupancy_from_ll_idea.md` for binding a runtime trip count to unroll a
   `collapse()`-shaped loop): a small, named-value binding file, just binding a placement tag to a
   symbol name instead of an integer to a free parameter. Reusing that existing mechanism's shape
   rather than inventing a new configuration format is itself a real, concrete design decision worth
   preserving when this gets built.

The honest scoping win here is that mode (2) needs **zero new mechanism** beyond what's already
proposed (the warmth-inference idea, unchanged), mode (1) needs only the already-established
`hipMemcpy`/`__tgt_target_kernel` reading from Q2 plus one explicit "no call found → assume placed"
rule, and mode (3) is a config-file-reading feature with a direct, already-established precedent to
copy rather than design from scratch — a genuinely scoped, buildable v1, not a research problem.

## What this changes about the 5.1 roadmap's scoping

Concretely, 5.5 needs real work at every stage, not just Stage 3/4:

- **Stage 1 frontend** (`stage1_frontend_llvm_ir.py`): call `llvm.initialize_all_targets()` /
  `llvm.initialize_all_asmprinters()` instead of the native-only initializers — otherwise
  `Target.from_triple("amdgcn-amd-amdhsa")` fails outright and no GPU `.ll` loads at all. A real,
  first blocker, not a design question.
- **Stage 1 walker** (`stage1_ir_walker.py`): admit `amdgpu_kernel`-calling-convention and
  `__omp_offloading_*`-named functions as candidates (hypothesis 3); recognize the thread-index call
  vocabulary from Q3 as a new `SymbolOrigin` feeding ordinary affine `index_expr` construction
  (already-existing GEP machinery, unchanged); recognize `__kmpc_alloc_shared`/`__kmpc_free_shared`
  as transparent bookkeeping (Q1); map `addrspace(3)`/`addrspace(1)`/`addrspace(5)` to
  `MemorySpace.SHARED_LOCAL`/`GLOBAL`/`REGISTER` (currently every `Symbol` is hardcoded `UNKNOWN`);
  trace `hipMemcpy`/`__tgt_target_kernel` call sites and the `dim3`-constructor pattern to populate
  `LaunchConfig` and a new transfer-cost `Operation` (Q2); for stdpar specifically (hypothesis 6),
  candidate selection needs *no* change at all — confirmed by direct test, since it flatly scans every
  function in the module rather than tracing from one entry point. (A real, pre-existing,
  non-GPU-specific bug in `get_or_create_pointer_target_symbol()`, also found by that same test — a
  closure capturing multiple pointers got them silently conflated into one symbol — has since been
  fixed directly, independently of this exploration; see the stdpar section above.)
- **Stage 2 schema** (`stage2_ir.py`): no structural changes needed — `TargetKind`, `ParallelKind`,
  `MemorySpace`, `LaunchConfig`, `GPUVendor` already exist exactly as needed. One small, additive gap:
  no `OpKind` currently fits "a host↔device transfer's own cost" (bandwidth-bound over
  PCIe/Infinity Fabric, not an ALU op) — a new `OpKind.OPAQUE_CALL_GPU_TRANSFER`, paired with the
  already-existing `Operation.call_arguments` (which already captures a call's byte-count argument by
  position), is the natural, minimal addition, mirroring `OPAQUE_CALL_MPI`'s precedent exactly. A
  provenance tag (observed vs. inferred, per the transfer-table section above) is a second small
  addition alongside it — reusing `SymbolOrigin`-style tagging rather than a wholly new concept.
  Also needs a new `BranchPolicy.DIVERGENT_SUM` value (the sum-every-arm semantics `WORST_CASE` no
  longer carries post-fix) plus its own small combination function in `stage3_aggregate.py`/
  `stage3_aggregate_branch.py` to dispatch to, per the branch-divergence section above.
- **Stage 3**: a new GPU-specific reuse/coalescing adapter (`reuse_gpu.py`, matching 5.1's original
  naming) implementing the sympy-derivative coalescing classification from Q3 — deliberately *not*
  the CPU cache-line-reuse-distance engine, per hypothesis 2. Also where the cross-boundary
  "warmth"-reuse extension above (now "normal mode" in the transfer-table section) would plug in — it
  needs `stage3_aggregate.py`'s existing `warm`/`Kernel.successors`-walking loop extended to recognize
  a CPU-piece→GPU-piece (or reverse) transition as an inferred-transfer trigger, not a wholly separate
  pass; and the per-branch-point divergence classifier (force `DIVERGENT_SUM` for divergent/unclear
  branch points, leave the configured policy alone for confirmed-uniform ones) from the same section.
- **Stage 4**: a new `backend_gpu.py` + GPU machine-spec schema (per hypothesis 2, above), plus
  wiring in `occupancy_from_ll_idea.md`'s VGPR lower-bound estimator as an explicitly-labeled
  one-directional bound.
- **Stage 6**: report composition keeps CPU/GPU-kernel/transfer totals separate, never summed
  (hypothesis 2) — concretely, the standalone host↔device transfer table above (explicit-observed,
  USM-normal-mode-inferred, or USM-expert-mode-file-informed rows) is its own report artifact, not a
  line item folded into either roofline total.

## Fixture gap

Only one real GPU fixture exists today (`test_apps/c_app/kernel_hip.cpp`'s `stencil_kernel`) and it
only exercises the unit-stride coalesced case with no `__shared__` memory — 5.1's own roadmap note
already flags needing broadcast-read, strided, and `__shared__`-using fixtures (l.688-692 per the
background survey); this exploration's own scratch fixtures (a `__shared__`-using HIP stencil, a
`map(to)`/`map(from)` OpenMP stencil, an `is_device_ptr` variant, and a `omp_pteam_mem_alloc` variant)
are real, tested, and reusable as a starting point for that gap, but live only in this exploration's
scratch directory, not `test_apps/`. **One real exception**: `test_apps/cpp_app/kernel_stdpar.cpp`
(the stdpar fixture examined above under hypothesis 6) already exists in the real test suite, already
compiles under Stage 1's own contract, and — confirmed by actually running it through the real,
unmodified frontend/walker — is exactly the fixture that exposed the (since-fixed)
`get_or_create_pointer_target_symbol()` conflation bug; no new fixture was needed to reproduce or fix
that, only the walker change.

## Open questions before this becomes a real plan

- Whether `_is_candidate_function()`'s existing debug-info/basename filter really does admit an
  `amdgpu_kernel` function unmodified, once the module loads — a five-minute check against a real
  fixture once `initialize_all_targets()` is added, not attempted in this pass.
- Whether the new `SymbolOrigin`/thread-index recognition belongs in `stage1_ir_walker.py` directly
  or as a separate GPU-specific companion module alongside it (mirroring how CPU-only
  `_recover_omp_for_loop_dim()` already lives inline) — an organizational choice, not explored here.
- Whether `LaunchConfig` tracing (the `dim3`-constructor pattern) is worth attempting for the general
  case (grid/block dims computed across multiple functions, or passed through a wrapper) or only the
  direct-call case confirmed here — the general case is a real, harder call-argument-tracing problem
  `calltree_from_ll_idea.md`'s Use Case 3 already left open, not re-litigated here.
- 2-D/3-D thread-index affine-index-expression coverage (Q3's stated gap) and NVPTX/CUDA support
  (hypothesis 6) are both plausible extensions of the same mechanisms found here, neither actually
  tested — real scope questions for whoever picks up 5.5, not decided by this exploration.
