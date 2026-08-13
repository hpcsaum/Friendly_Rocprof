# Add three GPU-specific metrics to `extract_pop_metrics.py`

> **Implementation note (divergence):** after implementing, validating `GPUOff` against the real
> 2-rank vs. 4-rank strong-scaling data revealed a real problem: it went *up* (0.48 → 0.84)
> between the two runs even though nothing about offload improved — normalizing the CPU-only-compute
> bucket against *total* elapsed time (which includes communication) meant the nearly-tripled
> communication overhead diluted the ratio and made offload look artificially better. Rather than
> changing `GPUOff`'s formula, added a **fourth** metric, **GPU Utilization** (`GPU-Util` =
> `max GPU busy time / max total elapsed time`), which moves the way intuition expects (0.30 →
> 0.19) since it measures the GPU's raw wall-clock share directly rather than complementing the
> CPU-only bucket. Both are kept — they answer genuinely different questions (see
> `docs/pop_metrics_reference.md`'s "Why both `GPU-Off` and `GPU-Util` exist" note). Also, per two
> follow-up user requests: column order became `LB, CommE, PE, GPU-Util, GPU-Off, GPU-LB, CompE,
> GE, GPU-Eff` (GPU-Util before GPU-Off), and every GPU column name got a hyphen
> (`GPUOff→GPU-Off`, `GPULB→GPU-LB`, `GPUEff→GPU-Eff`) for readability — the plan text below
> still shows the original camelCase names and 3-metric/no-GPU-Util column order throughout; the
> implementation reflects the corrected version. See `docs/DEVELOPMENT_HISTORY.md` for the full
> account.

## Context

Looking at POP's hybrid (MPI+OpenMP) metrics page prompted a question: can the OpenMP-level
metrics (Thread Load Balance, Serial Region Efficiency, etc.) be adapted into GPU metrics by
substituting "time on GPU" for "time in parallel region"? Conclusion from that discussion: the
multiplicative model's OpenMP-level formulas rely on per-thread timing across a handful of
persistent, comparable workers (each OpenMP thread does a slice of the same region, for the
region's whole duration) — data `rocprof-sys` genuinely has for CPU threads. GPU kernels don't
have that shape: `rocprofv3` only gives per-dispatch aggregate duration, not per-GPU-thread
timing, so a naive OpenMP-style translation would measure the wrong thing at the intra-kernel
level. Three genuinely well-posed GPU-specific questions came out of that discussion instead, all
directly computable from data the tool already gathers (or nearly gathers) per rank:

**1. GPU Offload Efficiency** (per run) — an Amdahl's-law-style question: "how much of this
rank's time is still CPU-only compute — either genuinely serial code, or code not yet ported to
the GPU, or code not worth porting?" The tool already computes exactly this quantity internally
(`cpu_pure` in `compute_run_metrics()`, `postprocess/extract_pop_metrics.py:158`) as an
intermediate step toward `useful_compute`, but never surfaces it as its own number.
- **Framed as an efficiency, higher = better** (`1 - fraction`), consistent with every other
  column (LB, CommE, PE, CompE, GE all trend toward 1.0 = good) — confirmed with the user.
- **Named "GPU Offload Efficiency"** (column `GPUOff`), deliberately distinct from POP's own
  Serialisation Efficiency (Dimemas-based, already listed as "not computed" in the same report) —
  confirmed with the user.
- Formula: `1 - (max cpu_only_time across ranks / max total_time across ranks)`.

**2. GPU Load Balance** (per run, new this round) — the existing overall Load Balance already
mixes GPU kernel time into the combined `useful_compute` pool per rank, so real imbalance
*between GPUs* (as opposed to between whole ranks, CPU+GPU together) is currently invisible on
its own — per the user, worth separating out explicitly.
- **Named "GPU Load Balance"** (column `GPULB`), same avg/max shape as the existing `LB`, just
  restricted to the GPU-busy-time bucket instead of the combined pool: `avg(GPU busy time across
  ranks) / max(GPU busy time across ranks)`.

**3. GPU Efficiency** (scaling level) — the user's own framing: GPU kernel time that's compute-
or memory-bound should scale roughly linearly with the work assigned to that GPU; when it
doesn't, something's wrong. In **strong scaling** specifically, the classic failure mode is the
per-rank problem size shrinking below what the GPU needs to stay saturated (kernel launch
overhead stops being amortized, occupancy drops, the GPU becomes latency- rather than
throughput-bound) — "the GPU lost its power when the problem size/rank becomes too small." This
is structurally identical to POP's own Computation Efficiency (ref/scaled ratio of useful compute
time; strong scaling uses the *total* across ranks, weak scaling uses the *average* per rank) but
restricted to just the GPU-kernel-time bucket instead of the whole CPU+GPU `useful_compute` pool
— isolating whether it's specifically the GPU's contribution that stopped scaling, as opposed to
Computation Efficiency's whole-pool view, which could equally reflect a CPU-side or
communication effect.
- **Named "GPU Efficiency"** (column `GPUEff`), a *scaling*-level metric like CompE/GE — only
  meaningful, and only shown, for a multi-run (scaling-study) report where both the reference run
  and the run being compared have paired GPU data.
- Formula (strong): `total GPU busy time (reference, summed across ranks) / total GPU busy time
  (this run)`. Formula (weak): same ratio using the *average* per-rank GPU busy time instead of
  the total — same reasoning as CompE's own strong/weak split.

**Column order** (per the user's correction): non-scaling (per-run) metrics first, all scaling
metrics grouped at the end —
`LB, CommE, PE, GPUOff, GPULB, CompE, GE, GPUEff`. The two GPU-specific per-run extensions
(`GPUOff`, `GPULB`) sit together right after the official POP per-run trio; the official
scaling pair (`CompE`, `GE`) is immediately followed by the GPU-specific scaling metric
(`GPUEff`), so all three "scaling" columns are contiguous at the end as requested.

## Approach

### `postprocess/extract_pop_metrics.py`

**`compute_run_metrics()`**, per-rank loop (~line 149): when `gpu_per_rank is not None`, also
store two new per-rank fields already available as local variables in that branch:
- `"cpu_only_time": cpu_pure` (feeds GPU Offload Efficiency)
- `"gpu_busy_time": sum(gpu_per_rank[i].values())` (feeds GPU Load Balance and GPU Efficiency)

Both are `None` per rank when there's no GPU pairing for this run.

After the per-rank loop, alongside the existing `total_useful_compute`/`avg_useful_compute`
computation, add:
```python
cpu_only_values = [r["cpu_only_time"] for r in per_rank]
gpu_busy_values = [r["gpu_busy_time"] for r in per_rank]
has_gpu_data = all(v is not None for v in gpu_busy_values)  # all-or-nothing per run, same as today
max_gpu_busy = max(gpu_busy_values) if has_gpu_data else None

gpu_offload_efficiency = (1.0 - max(cpu_only_values) / max_total) if has_gpu_data and max_total > 0 else None
gpu_load_balance = (statistics.mean(gpu_busy_values) / max_gpu_busy) if has_gpu_data and max_gpu_busy > 0 else None
total_gpu_busy_time = sum(gpu_busy_values) if has_gpu_data else None
avg_gpu_busy_time = statistics.mean(gpu_busy_values) if has_gpu_data else None
```
Add `gpu_offload_efficiency`, `gpu_load_balance`, `total_gpu_busy_time`, and `avg_gpu_busy_time`
to the returned dict.

**New function `compute_gpu_efficiency(ref_metrics, run_metrics, scaling)`**, mirroring
`compute_scaling_metrics()`'s strong/weak split exactly but over the GPU-only totals:
```python
def compute_gpu_efficiency(ref_metrics, run_metrics, scaling):
    key = "avg_gpu_busy_time" if scaling == "weak" else "total_gpu_busy_time"
    ref_value, run_value = ref_metrics[key], run_metrics[key]
    if ref_value is None or run_value is None or run_value <= 0:
        return None
    return ref_value / run_value
```
For the reference run compared to itself: `1.0` if it has GPU data, else `None` (same
special-case shape `write_report()` already uses for CompE/GE's reference row).

**`write_report()`**: columns appended in the order `LB, CommE, PE, GPUOff, GPULB, [CompE, GE],
GPUEff`:
- `GPUOff`/`GPULB`: shown when `any(m["gpu_offload_efficiency"] is not None for m in
  all_metrics)` (one shared condition — both come from the same `has_gpu_data` gate per run, so
  they always appear/disappear together) — per-run, so these can appear even in a single-run
  report.
- `GPUEff`: shown only when `multi_run` AND at least one non-reference run has a computable value
  against the reference (`any(compute_gpu_efficiency(ref_metrics, m, scaling) is not None for m
  in all_metrics if m is not ref_metrics)`) — a scaling-only metric, never shown for a
  single-run report.

Add three more "Metric explanation" bullets (only shown alongside their respective columns), each
explicitly noting it's a project-specific extension, not part of the official POP catalog, and —
for `GPUEff` — the strong-scaling interpretation the user described (a low value flags the
per-rank problem size shrinking below what keeps the GPU saturated).

### Docs

- `docs/pop_metrics_reference.md`: new "GPU-specific extensions (not part of the official POP
  catalog)" section covering all three metrics — formulas, rationale, and for `GPU Efficiency`
  specifically, the compute/memory-bound-scales-linearly reasoning and contrast with
  `Computation Efficiency` (whole CPU+GPU pool vs. GPU-only contribution).
- `README.md`'s "Computing POP metrics" section: one short paragraph covering all three new
  columns and when each does/doesn't appear.

### Tests

Extend `postprocess/tests/test_extract_pop_metrics.py` (reusing existing fixtures, no new ones
needed):
- `pop_combined_2rank`: assert `gpu_offload_efficiency`, `gpu_load_balance`,
  `total_gpu_busy_time`, and `avg_gpu_busy_time` all match hand-computable values from the
  existing fixture's kernel CSVs (rank pid 2001 GPU total ≈0.284724933s, pid 2002 ≈0.285s →
  `GPULB = mean/max ≈ 0.9995`).
- `pop_ref_2rank` / `pop_combined_mismatch` (no effective GPU pairing): all four new fields are
  `None`.
- `compute_gpu_efficiency()`: strong vs. weak give different results (same style as the existing
  `test_strong_and_weak_give_different_computation_efficiency`); reference-vs-itself is `1.0`;
  `None` when either side lacks GPU data.
- `write_report`: `GPUOff`/`GPULB` columns appear together for a single-run report using
  `pop_combined_2rank`, and are absent for `pop_ref_2rank`; `GPUEff` appears only in a multi-run
  report where both runs have GPU data, and is absent otherwise (including a multi-run report
  where only one side has GPU data); column order in the header line matches
  `LB, CommE, PE, GPUOff, GPULB, CompE, GE, GPUEff`.

No CLI changes — all three metrics are computed automatically whenever the underlying data
supports them, no new flags.

## Verification

- `python3 -m unittest discover -s postprocess/tests` — full suite green.
- Regenerate the real 2-vs-4-rank report (both directories already GPU-paired) with
  `--scaling strong` and eyeball all three new columns: `GPUEff` in particular is the interesting
  number here — a real strong-scaling pair on a stencil code, so this is a legitimate test of
  whether the GPU's own contribution scaled well going from 2 to 4 ranks, independent of the
  communication-efficiency drop already observed.
