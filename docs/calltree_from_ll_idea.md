# Building a calltree / dependency graph from LLVM IR — exploration

Status: **exploration only for this document's own ideas — no decisions made here.** (Stage 1 itself,
the real LLVM-IR frontend this exploration reads its examples from, is being actively implemented in
parallel on this branch; see "Checking the idea against the real Stage 1 implementation" below for an
analysis of that real, as-built code.) This is a feasibility sketch for an
idea adjacent to the `parsing/` static-roofline tool (which already parses `-O0 -gline-tables-only`
LLVM IR — see `docs/plans/5.1-theoretical-roofline-tool-design.md`), not a plan for it. Started from
two motivating use cases, plus two further facets explored later in the same conversation:

1. Coupling this data with the existing `postprocess/` calltree tools to enrich their views.
2. Using it standalone to identify workflows that could run in parallel (OMP tasks, or a GPU).
3. Tracing one variable's lifetime (allocation to deallocation) across every function that
   reads/writes it, in order — see "Use case 3" below.
4. The same independence question as (2), but *inside* one function, instruction-by-instruction —
   see "Use case 4" below.

## Verdict, up front

**The call graph itself (who calls whom, from where) is cheap and reliable to extract.** A real
*data-dependency* graph precise enough to safely auto-suggest parallelization is a much harder,
open-ended static-analysis problem that `llvmlite` cannot do for us — LLVM's own alias-analysis
machinery isn't exposed through the C API `llvmlite` wraps. The realistic path is a three-tier one:
a solid, cheap static call graph (real value on its own for use case 1); a **viewer** over that
graph (structural/annotated, no automated safety claims at all) that a human or an AI agent can
reason over directly — arguably the actual highest-value, lowest-effort v1; and, only beyond that,
a *heuristic, suggest-don't-decide* layer for use case 2 that leans on symbolic index-expression
machinery the roofline tool is already building for a different reason.

## What's cheaply extractable — confirmed with a real test

Compiled a tiny real C example (`helper()` called from `reduce_it()`'s loop, plus calls to
`fabs`/`sqrt`) at `-O0 -gline-tables-only -Xclang -disable-O0-optnone` and inspected the real IR:

```llvm
%4 = call double @llvm.fabs.f64(double %3), !dbg !13
%6 = call double @sqrt(double noundef %5) #3, !dbg !15
%17 = call double @helper(double noundef %16), !dbg !28
```

Every `call`/`invoke` instruction directly names its callee by symbol — trivially walkable via
`llvmlite`: iterate every basic block's instructions, filter `opcode == "call"`, read the callee's
name. This gives, essentially for free while already walking the IR for Stage 1's own purposes:

- **Direct call edges**, caller function → callee symbol, per call site.
- **Declaration vs. definition**: a callee with no body in this file (an external symbol, e.g.
  another translation unit's function, or `sqrt`/MPI/HIP runtime calls) is trivially distinguishable
  (`fn.is_declaration`) from one whose body is also walkable in the same module.
- A real, already-made design choice that happens to help here too: compiling at `-O0` means calls
  are **not inlined away** — a higher optimization level would have destroyed many of the exact
  edges this idea wants to see.

**C has no name mangling**, so the callee symbol is already the human-readable name. **C++ (Itanium
mangling) and Fortran (`__<module>_MOD_<name>`-style) do mangle** — callee symbols for those
languages need demangling to be presentable, though the mangled form is still perfectly usable as a
graph key even before demangling.

## A real limitation just found, affecting this idea *and* the existing roofline design equally

Tried to read the `!dbg !N` debug-info attached to each call instruction (needed for "this call, at
this exact source line") through `llvmlite`'s object API and found **neither `Module` nor
instruction objects expose any metadata accessor at all** (`ValueRef`'s public attributes have no
`.metadata`/`.debug_metadata`; `Module`'s public attributes have no metadata-node listing either).
The debug info is real and present in the IR text (confirmed: `!8 = distinct !DISubprogram(name:
"helper", ..., line: 2, ...)` is right there in the `.ll` file) — `llvmlite` just doesn't surface it
as structured Python objects, only as part of the opaque string form.

Practical consequence: getting "which source line" for a call site (or for anything else Stage 1
needs debug info for) means **text-scraping the `!dbg !N` / `!DILocation` / `!DISubprogram`
substrings directly**, not calling a clean `llvmlite` accessor — a hybrid of `llvmlite`'s structured
walk (for instructions/blocks/functions) plus regex-level text parsing (for debug info specifically).
This isn't a blocker, but it's a real implementation-shape detail that wasn't previously verified,
and it applies identically to the already-designed roofline Stage 1 (which also plans to "locate via
debug-info correlation") — worth fixing in that design regardless of whether this call-graph idea
goes anywhere.

## Use case 1 — coupling with `postprocess/`'s existing calltree tools

The existing calltree tools (`stage4_rocprofsys_*_tree.py` etc.) build a **dynamic, empirical** tree
from real profiling data: actual observed call stacks, real timing/self-time, real call counts, and
— importantly — real *resolved* behavior (an indirect call actually went somewhere specific at
runtime; a branch was or wasn't taken; a loop ran some concrete number of times). A **static** call
graph from `.ll` is a fundamentally different, complementary kind of information:

- It can show the **full possible** call graph, including paths a given profiling run never
  exercised (error handling, rarely-taken branches, a subroutine that exists but wasn't invoked this
  run) — a coverage/completeness lens the dynamic tree can't offer on its own.
- It can **attach static facts to a node the dynamic tree already flagged as hot** — concretely,
  since the roofline tool's Stage 3 already computes a function's op/byte counts, a hot node in the
  *empirical* tree could be annotated with its *theoretical* arithmetic intensity/roofline
  position — "why is this hot, and what's its ceiling" sitting right next to "how hot it actually
  was." This is the most concrete, genuinely valuable integration point found in this exploration.

**Feasibility of the join itself**: both sides ultimately key nodes by function/label name, so
merging is a name-keyed join in principle — but `postprocess/`'s existing label-cleaning machinery
(`clean_label()`, kernel-owner regexes, `PID_SUFFIX_RE`, etc.) already has to normalize rocprof-sys's
own naming/demangling conventions; matching IR-derived (demangled) names against that convention
needs real, if not conceptually hard, normalization work — not verified in this pass.

**Real limits**: indirect calls (function pointers, C++ virtual dispatch) can't be resolved
statically in general — the static graph would show "calls *something*" with an unknown target where
the dynamic tree, having actually run the program, knows exactly what was called. How much this
matters depends on how much indirection the actual target codebases use; HPC scientific code tends
to be more procedural than typical enterprise C++, but this project's own `test_apps/` and real
codebases haven't been checked for this specifically.

## Use case 2 — standalone parallelism-opportunity detection

This is the more ambitious, more uncertain half. Proving two computations are **safe** to run
concurrently fundamentally requires proving the **absence** of a dependency between them — a
conservative, often-undecidable static problem in C/C++ specifically because of pointer-aliasing
uncertainty (this is exactly why real compilers' auto-parallelizers, e.g. Polly, have had limited
real-world success at this for decades without `restrict` annotations or a real points-to analysis).
LLVM has real alias-analysis infrastructure for this — but, per the limitation above, it isn't
something `llvmlite` exposes usably from Python; building it ourselves would mean either a real
inter-procedural points-to/mod-ref analysis (a substantial undertaking) or writing a small companion
LLVM pass in C++ to do the analysis and hand results back as data (a heavier, cross-language tooling
investment, not evaluated further here).

**A realistic, much more modest first scope**: not a sound auto-parallelizer, but a **heuristic
suggestion tool** — flag plausible candidates for a human to verify, using machinery already being
built for the roofline tool rather than new analysis infrastructure:

- **Sibling statements/calls touching provably-disjoint `Symbol`s** (by our existing Access-based
  read/write-set extraction) are candidates — with the same explicit caveat the roofline design
  already documents (`AliasMode.NONE`, "assume no aliasing by default"), reused here rather than
  invented fresh.
- **Loop-carried-dependency checking for a single loop** (a real OMP `parallel for`/task candidate
  question: does iteration *i*'s write affect iteration *i+1*'s read?) is a smaller, well-studied
  problem than full alias analysis, and maps directly onto sympy-based affine index-expression
  comparison across two symbolic iteration values — i.e., **the same coefficient/stride machinery
  already built for Stage 3's reuse cascade** is a natural fit here, not new infrastructure. This is
  the strongest concrete synergy this exploration found.
- **Cross-function/call-level independence** (two separate function calls, not just two accesses
  within one loop body) needs the harder inter-procedural mod/ref summary noted above — a plausible
  later extension, not a first deliverable.
- Fortran's language-level no-aliasing guarantee for dummy arguments (already noted as a real
  advantage in the roofline design's Stage 1) would make this heuristic meaningfully *more* trustworthy
  for Fortran code than for C/C++, worth remembering when scoping which language to try this on first.

### A viewer alone, with no automated suggestion logic at all, is a legitimate MVP

Worth stating explicitly, since it changes what "useful v1" means: the hard part explored above is
*proving* independence well enough to safely auto-suggest it. A **viewer** doesn't need to prove
anything — it just needs to make the graph's structure (calls, read/write sets per node, loop
bounds, the roofline cost annotations from use case 1) legible enough that a human expert, who has
domain knowledge the tool doesn't, can spot a parallelization opportunity themselves. That sidesteps
the alias-analysis problem entirely rather than solving it, and is a dramatically smaller lift than
a confidence-scored automated suggester — most of the "hard" work becomes optional polish
(highlighting disjoint-symbol candidates as a hint) rather than a prerequisite.

Two distinct consumers, worth designing for separately since they want different output shapes from
the same underlying graph data:

- **A human**: something visual/interactive — even a static rendering (e.g. Graphviz `dot` output,
  in the same spirit as the roofline design's own opt-in `matplotlib` plot: a real dependency, only
  pulled in when actually rendering) would be new capability for this project — every existing
  `postprocess/` calltree tool renders as indented text, never a graph.
- **An AI coding agent**: a structured, machine-parseable form (JSON, or the same DOT text) is
  likely *more* useful than an image — agents reason over structured text far better than they read
  rendered pictures. This is effectively free once the graph is built (a different serialization of
  the same data), and is exactly the kind of thing an agent given access to a codebase could
  productively act on directly, letting an agent (rather than a human) drive the actual
  task/kernel-splitting rewrite once it has spotted a real opportunity in the structure.

## Use case 3 — a "data view": tracing one variable's lifetime across functions

A third facet of the same underlying graph, and genuinely **more tractable than use case 2**,
verified with a real example:

```c
double *data = malloc(n * sizeof(double));
fill(data, n);
double total = sum_it(data, n);
free(data);
```

After the same normalization pass list already planned for Stage 1 (SROA/mem2reg/instcombine/
simplifycfg), the compiled IR for this becomes:

```llvm
%4 = call noalias ptr @malloc(i64 noundef %3), !dbg !38
call void @fill(ptr noundef %4, i32 noundef %0), !dbg !39
%5 = call double @sum_it(ptr noundef %4, i32 noundef %0), !dbg !40
call void @free(ptr noundef %4), !dbg !41
```

One SSA value (`%4`), used directly at three call sites, no indirection at all — and note
`malloc`'s result is even auto-tagged `noalias` by LLVM/Clang convention already, for free. This is
the key reason it's more tractable than use case 2's independence question: tracing "where does
*this exact* value go" is an **exact** question (an SSA value's def-use chain), not an approximate
one ("could two arbitrary pointers alias") — the hard, conservative aliasing problem only resurfaces
for the specific case of a pointer being stashed into memory (a struct field, a global, an
array-of-pointers) and reloaded somewhere else, where "is this reload the same store" genuinely is
an aliasing question again. Direct parameter-passing chains (the common case) stay exact throughout.

**Allocation/deallocation detection reuses the call-graph machinery directly** — `malloc`/`free`,
`hipMalloc`/`hipFree`, or Fortran's runtime allocate/deallocate calls are just specific call-target
names to recognize, the same call-instruction walk already explored for the call graph itself; a
stack-allocated (`alloca`) variable's "deallocation" is implicitly function return, needing no
separate detection at all.

**A real `llvmlite` limitation found while checking this**: instructions expose `.operands` (forward:
what a given instruction consumes) but nothing like `.uses`/`.users` (reverse: given a value, what
consumes it) — confirmed by inspecting `ValueRef`'s full attribute list. Tracing "who uses this
value" therefore means scanning a function's instructions and checking whether each one's operands
match the tracked value, rather than a direct reverse-lookup — brute-force, not elegant, but cheap
in practice at realistic (dozens-to-low-hundreds of instructions) per-function scale, and yet another
data point (alongside the earlier metadata-access gap) that `llvmlite`'s Python surface is
narrower than LLVM's real analysis capability.

**Crossing function boundaries** is call-graph-edge-following plus matching a call site's argument
position to the callee's parameter position, then recursing into the callee's own local trace with
the same mechanism — nothing new, just the call graph (use case 1) and the per-function SSA trace
composed together.

**"In order" isn't a flat sequence once loops and branches are involved**, and this is where reusing
existing roofline machinery pays off again rather than inventing new structure: a touch inside a
loop is naturally represented via the *same* `LoopDim`/trip-count structure Stage 2/3 already
extract (a touch happens once per iteration of a known loop, not as an enumerated list of N
runtime events), and a touch that only happens down one branch arm is the *same* worst-case/one-path
shape `stage3_opcount.py`'s branch-policy handling already models — the trace is closer to an
annotated DAG than a single ordered list.

**Storing a pointer into memory isn't a hard wall — it's a gradient, and recursively following the
container is often still cheap, just not uniformly so.** "Reload from a different context" doesn't
have to mean giving up; it can mean recursively tracing the *container* (the struct field, global,
array slot) the same way the original pointer was traced. What that recursion actually costs depends
on how the storage location is addressed:

- **Fixed-offset storage** (a named global, or a struct field at a compile-time-constant offset):
  finding every other store/load to that *same* global+offset is a **syntactic** search (matching
  the same base symbol and the same constant GEP offset), not a real aliasing computation — it's
  exact, just scoped to the whole program (every `.ll` file, since a global is visible everywhere)
  instead of one function. Real cost, but linear in program size, not a different kind of problem.
- **Dynamically-indexed storage** (an array of pointers, keyed by a runtime-computed index): whether
  a later read at index `f(x)` is the same slot as an earlier write at index `g(y)` becomes a
  symbolic-equality question — the *same* sympy-based index-expression comparison Stage 3's reuse
  cascade already does, just answering "provably the same / provably different / unknown" instead of
  a hard yes/no, and only as reliable as the two expressions actually being comparable (identical
  variables in scope vs. two independently-opaque computed values).
- **Genuinely hard, and recursion alone doesn't fix it**: pointer-to-pointer indirection through
  arbitrary runtime values, type-punned/`void*` generic containers, and virtual dispatch — this tier
  really does need LLVM's real alias analysis, which `llvmlite` doesn't expose.

This is best framed as a **demand-driven** analysis, not a full points-to computation: since only
*one* traced value's fate is being asked about at a time, each recursive hop only costs a bounded,
whole-program search triggered by that one query — not the cost of computing points-to relationships
for every value in the program upfront. Real HPC code likely doesn't nest many such hops deep for any
one variable, but that's an assumption about the actual target codebases, not something confirmed —
this changes "how far to trace" (below) from a fixed stopping rule into a genuine cost/precision
dial: keep recursing through fixed-offset storage cheaply, degrade to symbolic-expression confidence
for dynamically-indexed storage, and stop (report "untraceable past this point," an honest boundary)
only at the genuinely hard tier.

Also worth stating plainly: granularity is ambiguous for
aggregates (an existing `Access.index_expr` gives *some* per-element precision within one function,
but reconciling "which element" across two functions that index the same array differently is a real
open question, not automatic); a host/device (HIP `hipMemcpy`, kernel-launch argument) crossing needs
its own recognized edge type — concretely pattern-matchable, like allocation calls, but distinct from
an ordinary function-argument edge; and cross-translation-unit tracing needs the same multi-file
stitching already noted as an open decision for the call graph itself.

## Use case 4 — intra-function dependency graph, for refactoring

Prompted by a real observation worth stating plainly: **"every function is a single, well-defined
thing" is a false assumption** — including one this exploration's own `Kernel` model (and the
roofline tool's "one target loop, chosen by file+line" framing) has been implicitly leaning on.
Real functions routinely do several genuinely independent sub-computations that happen to be
textually co-located, converging only at the end. This use case asks the same independence question
as use case 2, just scoped *inside* one function instead of across the whole call graph — and it
turns out to be the **easiest** version of that question explored so far, verified with a real
example:

```c
double norms(const double *a, const double *b, int n) {
    double norm_a = 0.0;
    for (int i = 0; i < n; ++i) norm_a += a[i] * a[i];
    double norm_b = 0.0;
    for (int j = 0; j < n; ++j) norm_b += b[j] * b[j];
    return combine(norm_a, norm_b);
}
```

After the same normalization already planned for Stage 1, the two loops are trivially, mechanically
separable straight from the IR: **disjoint Access sets** (the first loop's `getelementptr`s are
based on parameter `%0`, the second's on `%1` — the exact `Access.symbol` comparison already built
for the roofline tool, no new mechanism) and **no control dependency between them** (the second
loop's entry-branch condition (`icmp slt i32 %.0, %2`) only references its own induction variable and
`n` — never anything the first loop computed). Both facts needed to call these two loops safe to run
concurrently are directly visible in the normalized IR; the call to `combine(norm_a, norm_b)` is the
explicit convergence/sink point where both results are consumed together — exactly the
fork-join shape described.

**Why this is easier than use case 3's cross-function tracing**: it mostly stays entirely within one
function's already-fully-extracted SSA form — no whole-program search, no memory-indirection
tiering, most of the time. The graduated tiers from use case 3 only resurface if a candidate
"independent" region touches something that escapes the function (a global, a call to another
function with side effects that might touch the same state) — calling into another function from
inside a candidate region reintroduces the same inter-procedural mod/ref question already flagged as
hard/deferred elsewhere in this doc; same answer applies here too: flag it, don't try to resolve it
automatically.

**What's genuinely new machinery, beyond what's already been discussed elsewhere in this doc**:
- **Resolved** (see "Checking the idea against the real Stage 1 implementation" below): identifying
  candidate "regions" needed walking a function's full basic-block/CFG structure, not just the loop
  nests the roofline `LoopDim` model already captures. Stage 1 now does exactly this.
- The **narrow "does this branch's condition data-depend on that other region's output"** check
  remains new, unbuilt machinery — see below.
- Representing the result as an explicit **graph** is **partially resolved** (see below) — three
  typed relationships (nesting, sequence, branch-choice) now exist and convergence/sink code is no
  longer dropped, but the edges are still structural/sequential, not verified data/control-dependency
  edges.

**A real, valuable unifying insight, not just a new feature**: decomposing a function into its
independent regions isn't only useful for this refactoring-support idea — the *existing* roofline
tool needs it too. "One function = one `Kernel`" was always a simplification; a messy real function
like `norms()` above has no single well-defined "the loop" to target by file+line, and would benefit
from separate AI/cost numbers per independent sub-computation rather than an arbitrary single-loop
pick or an awkward blended number for the whole function. The same region-decomposition work would
improve both tools at once.

**Real limits**: proper control-dependence (not just this narrow, bounded check) needs real
CFG/dominance reasoning that doesn't exist anywhere in this design yet — standard, well-understood
compiler theory, not a research problem, but genuinely unbuilt, confirmed still true against the real
5.4 implementation (see below). Anything a candidate region delegates to a called function still
re-enters the harder, deferred tiers from use case 3, though the new `Operation.call_arguments` field
(see below) now resolves the simple pass-through case. The nested/interleaved-loop decomposition
concern originally raised here turned out not to be a real limit: Stage 1's actual implementation
decomposes deeply nested and interleaved loop/segment structures cleanly (confirmed against a real
`triple_nest_with_middle_segments` fixture).

## Checking the idea against the real Stage 1 implementation (5.3 → 5.4, as built)

Stage 1 (`parsing/stage1/`) has moved substantially since the previous pass through this section —
plan `5.4-function-level-kernels-and-aggregation.md`'s two objectives (function-level kernels +
sequencing, then `Symbol.origin`/shared identity/call arguments) directly targeted several of the
gaps raised here, and one new field's own docstring (`Kernel.is_function_summary`, `stage2_ir.py:290`)
explicitly cites this document by name. This section records what's actually resolved at the
data-model level versus what's still open. Still analysis, not a plan — nothing here has been
implemented as a result of this section.

### The convergence-node and edge-typing gaps are resolved

Both concrete gaps previously flagged against `_build_shape_forest`/`_build_kernels_for_shape` are
fixed, confirmed against the code:

- **Convergence/sink nodes are no longer dropped.** The whole function is now folded into the same
  containment tree as a synthetic root shape, and a function's straight-line content (setup, teardown,
  and critically the code joining two sibling loops back together) is walked into its own sequenced
  kernel(s) instead of being silently excluded. `function_level_kernels.c`'s own test fixture *is* the
  `norms()`/`combine()` example from Use case 4 above (`two_norms_then_combine`) — it now produces
  exactly the 3 kernels this doc's own example wanted to see: the two independent loops, plus the
  `combine()` call as its own sequenced kernel.
- **Edges are now typed, not one conflated list.** Three distinct relationships now exist: nesting (a
  real loop's own ancestor `loop_nest` chain, via the shape forest), branch choice
  (`Kernel.child_source_lines`, now narrowed to conditional-arm "OR" children only), and sequence
  (`Kernel.next_sibling_source_line`, new — "A fully finishes, then B starts," with its own docstring
  explicitly noting this is what lets a downstream consumer reason about whether B might still find
  A's data warm). Still encoded as `source_line` integers requiring a lookup rather than direct object
  references — a minor remaining gap, not a conceptual one.

What's still open: **no independence/dependency check actually runs anywhere in this decomposition.**
It remains a structural walk (which pieces exist, and in what order/nesting) — it still doesn't test
Access-disjointness or control-dependence between siblings. That's expected: 5.4's own objective 2 ("a
cross-kernel data-dependency graph") explicitly separates this structural foundation (done) from the
dependency analysis itself (not started) — the two steps landed so far build the model the dependency
graph will need, not the graph itself.

### The free-param/opaque-symbol conflation is resolved at the data level

This was the sharpest gap raised previously, and it's addressed directly. `stage2_ir.py`'s new
`SymbolOrigin` enum (`ARGUMENT`/`ALLOCA`/`INDUCTION_VAR`/`CALL_SITE`/`POINTER_TARGET`/`FREE_PARAM`/
`UNRESOLVED`) tags every `Symbol` with where it actually came from — its own docstring states the
intent almost exactly as raised here: *"lets a downstream consumer tell a real, recognized quantity
apart from an opaque 'gave up' placeholder, which otherwise look identical."* `FREE_PARAM` = a
synthesized scalar Stage 1 couldn't trace; `UNRESOLVED` = an untraceable base pointer — both now
distinguishable from `ARGUMENT`/`ALLOCA` (a real, structurally-grounded function parameter).

Practical effect: a downstream independence check or dependency-graph builder can now check
`Symbol.origin` and treat `FREE_PARAM`/`UNRESOLVED` as "unknown, don't claim independence" rather than
silently accepting them as evidence of disjointness — the earlier false-positive risk (an untraceable
shared array read as "disjoint" because both sides got anonymous, unrelated names) is now detectable
in the data. **The remaining gap is that no consumer does this yet** — no independence checker exists
at all, so this is a resolved *data-availability* problem, not yet a resolved *behavior*: anyone
building on today's `Kernel` output still has to remember to check `origin` themselves, and a naive
name-based Access-disjointness test that ignores it would still produce the same false "independent"
verdict as before.

### Symbol identity is now shared within one function

Previously, comparing two kernels' Access sets had to go by `Symbol.name`, never object identity,
because each kernel built its own private symbol cache. Now `SymbolRegistry` is constructed once per
function and shared across every sibling kernel's own `WalkContext` — confirmed directly in
`_build_kernel_from_blocks`'s own docstring: *"the same real alloca/argument/induction-variable
resolves to the same `Symbol` object across sibling kernels."* Two sibling kernels from the same
function that reference the same real parameter now get the literal same `Symbol` object — safe to
compare by identity, not just by name.

This is still function-scoped: kernels from *different* functions still get independent registries, so
cross-function comparison still needs name-based matching, or the mechanism below.

### New: `Operation.call_arguments` gives a real, if narrow, cross-function identity link

A new field drags the caller-side `Symbol` passed at each argument position of every call (recognized
or opaque). This is a real, if deliberately narrow, answer to Use case 3's "matching a call site's
argument position to the callee's parameter position": for the clean pass-through case (a bare
function argument, a recognized induction variable, or a bare pointer/array forwarded directly to
another call), a caller's own kernel now carries a genuine reference to which of its own symbols went
where. This was made **deliberately conservative** after a real regression: resolving anything beyond
a bare pass-through (a loaded scalar local, a GEP-computed pointer, the address of an accumulator)
risked silently minting new required `--param` bindings as a side effect, so those cases resolve to
`None` instead — confirmed directly against `combine(norm_a, norm_b)`'s own arguments, which resolve
to `None` since `norm_a`/`norm_b` are accumulators, not bare arguments. So: the easy case from Use case
3 (`fill(data, n)`, `sum_it(data, n)`) is now mechanically supported; the harder derived/accumulator
case is unchanged.

### What's still genuinely open

- **No actual independence/dependency check exists.** The structural and identity foundations above
  make one buildable, but nothing computes Access-disjointness, affine-range disjointness, or
  control-dependence today.
- **Cross-kernel reuse-aware byte-counting is still unbuilt.** `next_sibling_source_line`'s own
  docstring now explicitly names the cache-warmth question this raises — the ordering data needed to
  reason about it now exists — but Stage 3/4's own byte-counting still analyzes every kernel as an
  independent, cold-start unit (confirmed by the 5.4 as-built CLI spot-check: op/byte counts for
  individually-analyzed kernels are numerically unaffected by any of this work). So the array
  byte-count-overestimation risk raised previously — kernel B charged a cold read for data kernel A
  just wrote — is unchanged in practice, though the prerequisite data is now in place to eventually
  fix it.
- **Symbol-name/affine-range disjointness across kernels still isn't wired up** — the sympy machinery
  exists in Stage 3, nothing calls it for this purpose yet.
- **Cross-function identity beyond a bare pass-through argument is still unresolved** —
  `call_arguments` only covers the simple case; a value threaded through a struct field, computed and
  then passed, or crossing a translation-unit boundary is still untraceable.

## Decisions that would need to be made before this becomes a real plan

- **Scope of "dependency"**: pure call graph (cheap, high-value, no open research problem) vs. real
  data/alias dependency (hard, no off-the-shelf `llvmlite` support, open-ended).
- **Per-translation-unit vs. whole-program**: this project's own per-TU `.ll` compilation model
  (one file per source file, no linking) means a *whole-program* call graph needs stitching multiple
  files' graphs together by symbol name (declared-but-undefined callee in file A matched to a
  definition in file B) — straightforward in principle, not yet attempted.
- **Static-only vs. hybrid with the existing dynamic profiling data**: a static-only graph is
  simpler to build but weaker (unresolved indirect calls, no notion of "did this path actually run");
  a hybrid using the empirical tree to resolve/prune the static one is more valuable but is real
  additional integration work against `postprocess/`'s existing label/naming conventions.
- **Where this lives architecturally**: the call-graph *extraction* is a near-free byproduct of
  Stage 1's planned "walk every function" mode (already under discussion for the roofline tool's own
  5.3) — but *rendering*/coupling with `postprocess/` and the *parallelism-suggestion* logic are
  naturally separate, later tools built on top of that byproduct, not part of the roofline pipeline's
  own Stage 1-5. Mirrors this project's existing pattern of one parsing frontend feeding multiple
  distinct downstream tools.
- **The now-confirmed debug-info-access gap**: needs a real (if small) design decision either way —
  text-scrape `!dbg`/`DILocation`/`DISubprogram` directly, or find/build some other structured
  extraction path — before anything depending on source-line correlation (which is most of what
  makes either use case valuable) can be built.
- **How much indirection/virtual dispatch actually shows up** in the real target codebases is an
  open empirical question, not yet checked — worth a quick look before investing further, since it
  directly bounds how complete a static call graph could ever be for this project's actual code.
- **Where exactly to draw the three-tier line for use case 3** (exact syntactic search for
  fixed-offset globals/struct fields, symbolic-confidence search for dynamically-indexed storage,
  honest stop for the genuinely-hard tier) — not whether to draw it, since a flat "stop at the first
  indirection" rule would throw away a lot of real, still-tractable cases.
- **Object vs. sub-object granularity for use case 3**: whether "touches this variable" means the
  whole allocation or a specific element/field — the existing `Access.index_expr` gives per-element
  precision *within* one function already, but reconciling that across two functions that index the
  same array under different names/bases isn't automatic.
- **Resolved**: whether use case 4's region-decomposition becomes a shared primitive or a one-off —
  it landed as shared infrastructure, built directly into the roofline tool's own Stage 1 (5.4's
  function-level kernels + sequencing) rather than standalone, addressing the roofline tool's "one
  function = one `Kernel`" oversimplification and this document's Use case 4 with the same code.
- **How much real control-dependence machinery use case 4 actually needs**: the narrow "does this
  branch depend on that region's output" check covers the clean fork-join case, but deciding how much
  further to invest in real CFG/dominance analysis (vs. leaving messier control-flow shapes as an
  honest "can't tell" the same way the hard tiers elsewhere in this doc do) is a real scoping choice,
  not yet made.
- **Resolved**: whether Stage 1 should distinguish a recognized, user-facing free parameter from an
  opaque "couldn't trace this" symbol in its exported `Kernel` — `Symbol.origin` now does this (see
  "Checking the idea against the real Stage 1 implementation" above). Still open on top of that: no
  downstream consumer actually checks `origin` yet, since no independence check has been built at all.
- **Whether a future independence check built on `Kernel` output should default to treating any
  opaque/unresolved array-base symbol (`SymbolOrigin.FREE_PARAM`/`UNRESOLVED`) as "unknown"
  (conservative) rather than "disjoint" (optimistic)** — the data to support either choice now exists;
  the choice itself is still unmade, since nothing consumes it yet.
- **Whether/how to make Stage 3's byte-counting reuse-aware across sequenced kernels** — the ordering
  data (`next_sibling_source_line`) now exists specifically to support this, but every kernel is still
  analyzed as an independent, cold-start unit today; a kernel whose array input was actually produced
  by an immediately-preceding sibling kernel still gets charged a full cold read.
