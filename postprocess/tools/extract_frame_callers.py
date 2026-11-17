#!/usr/bin/env python3
"""Show which call chains lead to any frame you name (default: GPU-wait frames such as
rocr::core::*Signal::WaitAcquire/WaitRelaxed) in rocprof-sys sampling tables.

Scope: reads rocprof-sys "timemory" text tables (sampling_wall_clock-<pid>.txt,
sampling_cpu_clock-<pid>.txt) and, for every row whose label matches --target, reports the
chain of ancestors above it, rolled up two ways: by the nearest caller outside the matched
frames, and by the nearest "anchor" ancestor (--anchor; default: hip*/OpenMP-offload sync calls
and runtime helper loops). It does not rank hotspots, filter noise, or touch GPU-side data.
Time is each row's SELF time by default (inclusive time would count nested matches such as
std::pair -> std::pair twice); --time inclusive switches to the table's SUM column.

Deliberately self-contained (standard library only, no imports from the rest of postprocess/):
the profiled code and its profiles may be unable to leave the machine they were collected on,
so this one file can be copied there and run in place.

Design: the DEPTH column plus depth-first row order give every row's ancestor chain; a row's self
time is its SUM minus its direct children's SUM. Labels in the table are truncated by rocprof-sys
with "..."; regexes match on the part that survives (the function name, before its arguments).

Functions: parse_rows(), build_nodes(), collect_matches(), anchor_of(), main().
"""

import argparse
import collections
import glob
import os
import re
import sys

ROW_RE = re.compile(
    r"^\|\s*\|(\d+)>>>(.*?)\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*([-+0-9.eE]+)\s*\|"
)
LABEL_PREFIX_RE = re.compile(r"^\s*\|?_*")
# The sampler's signal-return trampoline: frames below it are the sampler's own handler code, so
# a trampoline's time is credited to the function it interrupted (its parent) in self-time mode.
TRAMPOLINE_RE = re.compile(r"restore_rt|rt_sigreturn")
DEFAULT_TARGET = r"Signal::Wait(Acquire|Relaxed)"
# Callers that explain why a wait exists: HIP sync/blocking APIs, the OpenMP-offload plugin's
# stream/device synchronize, and the runtime helper loops that wait on behalf of no caller
# (OpenMP-offload RPC server thread, ROCr async-events loop).
DEFAULT_ANCHOR = (r"^(hip::)?hip[A-Z]|AsyncInfoTy::synchronize|AMDGPUStreamTy::synchronize"
                  r"|RPCServerTy::ServerThread::run|Runtime::AsyncEventsLoop")
TABLE_GLOBS = ("sampling_wall_clock-*.txt", "sampling_cpu_clock-*.txt")

HELP_BLURB = """\
A profile table tells you how much time a function took, but not who called it. This tool
answers "who called it?" for any function name (or pattern) you choose: for every table row
whose label matches --target, it follows the call chain upward and prints (1) how much time
those rows account for, (2) the nearest caller that is not itself a match, (3) the nearest
"anchor" caller (e.g. hipStreamSynchronize), and (4) the full distinct call chains.

By default --target looks for the frames where the CPU is waiting for the GPU
(Signal::WaitAcquire / Signal::WaitRelaxed). Examples:
  extract_frame_callers.py <dir>                              # who asks for GPU waits?
  extract_frame_callers.py <dir> --target 'std::pair' --anchor 'restore_rt'
                                                              # where do std::pair rows hang?

It reads sampling_wall_clock-*.txt and sampling_cpu_clock-*.txt, only prints text, and needs
nothing beyond Python 3. Limitations: rocprof-sys truncates long function names in these
tables, so patterns should match the start of a name; times are SELF times (sampler
signal-handler time credited to the interrupted function) unless you pass --time inclusive.

Under the hood, this parses output written by AMD's rocprof-sys -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.
"""


def parse_rows(path):
    """Yield (thread, label, depth, metric, seconds) for every data row of a timemory table."""
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ROW_RE.match(line)
            if not m:
                continue
            thread, raw_label, _count, depth, metric, _units, total = m.groups()
            label = LABEL_PREFIX_RE.sub("", raw_label).strip()
            yield thread, label, int(depth), metric, float(total)


def func_name(label):
    """Label without its argument list/template tail, for compact display."""
    return label.split("(")[0].strip()


def anchor_of(ancestors, anchor_re):
    """Nearest ancestor (walking upward from the match) whose label matches anchor_re, else None."""
    for label in reversed(ancestors):
        if anchor_re.search(label):
            return func_name(label)
    return None


def build_nodes(path, fold_signal_frames=True):
    """Parse one table into nodes [label, depth, sum_sec, self_sec, parent_index, thread], in file
    order. self_sec = sum_sec minus the SUM of direct children, except that with
    fold_signal_frames a signal-return trampoline child is not subtracted (its time stays with the
    interrupted parent). Returns (metric, nodes)."""
    stack = []  # indices of the current ancestor chain
    nodes = []
    metric = os.path.basename(path)
    for thread, label, depth, metric, secs in parse_rows(path):
        while stack and nodes[stack[-1]][1] >= depth:
            stack.pop()
        parent = stack[-1] if stack else None
        if parent is not None and not (fold_signal_frames and TRAMPOLINE_RE.search(label)):
            nodes[parent][3] -= secs
        nodes.append([label, depth, secs, secs, parent, thread])
        stack.append(len(nodes) - 1)
    return metric, nodes


def collect_matches(path, target_re, anchor_re, use_self=True, fold_signal_frames=True):
    """Scan one table. Returns (metric, file_total_sec, thread_totals, matches); thread_totals maps
    thread id -> sum of that thread's top-level rows; each match is (target_name, anchor,
    outside_caller, ancestor_names_root_first, seconds, thread), where outside_caller is the
    nearest ancestor that does not itself match target_re."""
    metric, nodes = build_nodes(path, fold_signal_frames)
    file_total = sum(n[2] for n in nodes if n[1] == 0)
    thread_totals = collections.defaultdict(float)
    for n in nodes:
        if n[1] == 0:
            thread_totals[n[5]] += n[2]
    matches = []
    for label, _depth, total, self_sec, parent, thread in nodes:
        if not target_re.search(label):
            continue
        ancestors = []
        while parent is not None:
            ancestors.append(nodes[parent][0])
            parent = nodes[parent][4]
        ancestors.reverse()
        outside = next((func_name(a) for a in reversed(ancestors) if not target_re.search(a)), None)
        matches.append((func_name(label), anchor_of(ancestors, anchor_re), outside,
                        [func_name(a) for a in ancestors], self_sec if use_self else total, thread))
    return metric, file_total, thread_totals, matches


def gather_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for pattern in TABLE_GLOBS:
                files += glob.glob(os.path.join(p, "**", pattern), recursive=True)
        else:
            files.append(p)
    return sorted(set(files))


def collapse_repeats(names):
    """Merge consecutive identical frames: [a, b, b, c] -> [a, b (x2), c]."""
    out = []
    for n in names:
        if out and out[-1][0] == n:
            out[-1][1] += 1
        else:
            out.append([n, 1])
    return [n if k == 1 else f"{n} (x{k})" for n, k in out]


def print_rollup(title, bucket, total):
    print(f"\n{title}")
    for name, secs in sorted(bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {secs:12.3f} s  {100 * secs / total:6.2f}%  {name}")


def main():
    ap = argparse.ArgumentParser(
        description=HELP_BLURB, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="rocprof-sys output directory (searched recursively) and/or "
                         "individual sampling_*_clock-*.txt files")
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help="regex for the frames to trace (default: %(default)s)")
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR,
                    help="regex for the ancestor to group by (default: %(default)s)")
    ap.add_argument("--time", choices=("self", "inclusive"), default="self",
                    help="which time to sum for each matched row (default: %(default)s)")
    ap.add_argument("--no-fold", action="store_true",
                    help="in self-time mode, do NOT credit signal-handler (___restore_rt) time to "
                         "the function it interrupted")
    ap.add_argument("--max-chains", type=int, default=10,
                    help="distinct full call chains to print per metric (default: %(default)s)")
    ap.add_argument("--max-depth", type=int, default=0,
                    help="show only the N nearest ancestors in each chain (default: whole chain)")
    args = ap.parse_args()

    target_re, anchor_re = re.compile(args.target), re.compile(args.anchor)
    files = gather_files(args.paths)
    if not files:
        sys.exit("error: no sampling_wall_clock-*.txt / sampling_cpu_clock-*.txt files found")

    new = lambda: collections.defaultdict(lambda: collections.defaultdict(float))  # noqa: E731
    totals = collections.defaultdict(float)                      # metric -> sum of file totals
    by_target, by_outside, by_anchor, chains = new(), new(), new(), new()
    by_thread = collections.defaultdict(lambda: collections.defaultdict(float))  # metric -> (file, thread)
    thread_total = {}
    for path in files:
        metric, file_total, thread_totals, matches = collect_matches(
            path, target_re, anchor_re, use_self=args.time == "self",
            fold_signal_frames=not args.no_fold)
        totals[metric] += file_total
        for thread, tsecs in thread_totals.items():
            thread_total[(metric, os.path.basename(path), thread)] = tsecs
        for name, anchor, outside, ancestors, secs, thread in matches:
            by_thread[metric][(os.path.basename(path), thread)] += secs
            by_target[metric][name[:100]] += secs
            by_outside[metric][(outside or "(none -- top of thread)")[:100]] += secs
            by_anchor[metric][(anchor or "(no anchor ancestor)")[:100]] += secs
            shown = ancestors[-args.max_depth:] if args.max_depth else ancestors
            chains[metric][tuple(n[:100] for n in shown + [name])] += secs

    print(f"Read {len(files)} table file(s); target /{args.target}/, anchor /{args.anchor}/, "
          f"time = {args.time}")
    if not any(by_target.values()):
        print("No rows matched --target. Check the pattern against `grep -c` on your files.")
        return
    for metric in sorted(by_target):
        total = totals[metric] or 1.0
        if all(len(c) == 1 for c in chains[metric]):
            print(f"\nNote: every match in {metric} is at the top level (no callers recorded) -- "
                  "this looks like a flat table, so nesting cannot be determined from it.")
        print(f"\n=== {metric}  (summed over files; table total {totals[metric]:.3f} s) ===")
        print_rollup("Matched frames:", by_target[metric], total)
        print_rollup("Nearest caller outside the matched frames:", by_outside[metric], total)
        print_rollup("Nearest anchor ancestor (--anchor):", by_anchor[metric], total)
        print("\nPer thread (file, thread id): matched time vs that thread's whole time:")
        for (fname, thread), secs in sorted(by_thread[metric].items(), key=lambda kv: -kv[1]):
            ttotal = thread_total.get((metric, fname, thread), 0.0) or 1.0
            print(f"  {secs:12.3f} s of {ttotal:10.3f} s ({100 * secs / ttotal:6.2f}% of thread)"
                  f"  thread {thread}  {fname}")
        print(f"\nTop {args.max_chains} distinct call chains (root first):")
        ranked = sorted(chains[metric].items(), key=lambda kv: -kv[1])[:args.max_chains]
        for chain, secs in ranked:
            print(f"  {secs:12.3f} s  {100 * secs / total:6.2f}%")
            for i, name in enumerate(collapse_repeats(chain)):
                print("      " + "  " * i + ("|_" if i else "") + name)


if __name__ == "__main__":
    main()
