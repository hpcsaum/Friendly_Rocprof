#!/usr/bin/env python3
"""Resolve CPU hotspot function names into rocprof-sys-instrument "-R" input.

Three unrelated jobs live in this one module, all in service of
scripts/instrument_hotspots.sh:

1. Resolve mode (default): read a hotspots report (written by
   extract_CPU_hotspots.py or extract_hotspots.py) or a rocprof-sys output
   directory directly, select CPU hotspot function names, optionally widen
   that selection (see below), and print "label<TAB>regex" pairs -- the
   regex being an escaped, unanchored substring pattern safe to pass to
   rocprof-sys-instrument's "-R/--function-restrict".

2. Selection-widening (on by default): a hotspot's raw self/inclusive-time
   selection can leave a GPU kernel's real CPU owner, or an MPI call's real
   caller, uninstrumented even though the kernel/MPI call itself is
   expensive -- leaving it structurally disconnected in the resulting
   trace. --gpu-output-dir resolves GPU kernel hotspots into their owning
   CPU subroutine names (stage4_rocprofsys_common.resolve_kernel_owners());
   --ancestor-depth pulls in each selected function's N nearest real
   callers (stage4_rocprofsys_common.expand_labels_with_ancestors()) so its
   position in the resulting tree stays meaningful. Both additions are
   built from generic, reusable stage4/stage5 primitives -- this module is
   just their one concrete consumer, feeding instrument_hotspots.sh.

3. --check-instrumented mode: after rocprof-sys-instrument has produced its
   own instrumented.json (documenting exactly which functions actually got
   instrumented, post-filtering), compare it against the requested labels
   and warn about any that didn't make it in. Never fails -- a lost
   function is a warning, not an error.

Only CPU-side function names are ever selected for instrumentation --
--gpu-output-dir only ever contributes a kernel's CPU *owner* name, never
the kernel name itself, since kernel instrumentation is a different
mechanism entirely.

Functions: escape_for_instrument_regex(), labels_from_output_dir(), labels_from_report(),
flat_tree_for_ancestors(), find_lost_functions(), main().
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

import select_hotspot_kernels
from stage4_rocprofsys_common import expand_labels_with_ancestors, flatten_tree, merge_rank_trees, resolve_kernel_owners
from stage4_rocprofsys_sample_flat import aggregate
from stage4_rocprofsys_sample_tree import load_rank_trees
from stage5_calltree_text_parser import flat_rows_from_calltree_text
from stage5_calltree_view import strip_wrapper_noise
from stage5_table_render import iter_table_rows, select_entries
import stage6_cli_common
import stage6_noise_config

HELP_BLURB = """\
Turns a profile_hotspots.sh (or extract_CPU_hotspots.py) report -- or a
rocprof-sys output directory -- into a list of CPU hotspot function names,
ready to feed into AMD's rocprof-sys-instrument as a "-R" (restrict) regex
list. This is what scripts/instrument_hotspots.sh uses to instrument only
the functions that already showed up as hotspots, instead of every
function in the binary.

By default, the raw hotspot selection is widened a bit so the resulting
trace's shape still makes sense: each selected function's immediate real
caller is pulled in too (--ancestor-depth, default 1; 0 disables it), and,
when --gpu-output-dir points at a paired rocprofv3 run, a hot GPU kernel's
real CPU owner is pulled in even if that owner wasn't itself a hotspot.
Only CPU-side functions are ever selected -- GPU kernel names from a
combined report can't be targeted this way, since kernel instrumentation
is a different mechanism entirely.

This tool also has a second, unrelated job: after rocprof-sys-instrument
has run, --check-instrumented compares its own instrumented.json output
against the functions that were requested and warns (without failing
anything) about any that didn't make it into the binary.

Under the hood, this prepares input for AMD's rocprof-sys-instrument -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/instrumenting-rewriting-binary-application.html
for details.
"""

_REGEX_METACHARS = set(".^$*+?()[]{}|\\")


class InstrumentedFileError(Exception):
    """Raised when instrumented.json can't be read -- never lets this abort
    the caller, since a lost-function check is a nicety, not a requirement."""


def escape_for_instrument_regex(name):
    """Escape only characters that are metacharacters in std::regex's default
    ECMAScript grammar -- the grammar rocprof-sys-instrument's -R/-I/-E
    matching uses via std::regex_search -- so an arbitrary function-name
    substring embeds safely in an unanchored pattern. '<', '>', '~', '::'
    etc. are literal in this grammar and deliberately left untouched."""
    return "".join(("\\" + c) if c in _REGEX_METACHARS else c for c in name)


def labels_from_output_dir(rocprof_sys_dir, top=None, threshold=None, show_all=False, unfiltered=False):
    cpu_entries, _gpu_entries, scanned, total = aggregate(rocprof_sys_dir)
    if not scanned:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found in {rocprof_sys_dir!r} -- "
            "nothing to select hotspot functions from"
        )
    rank_by = "inclusive" if unfiltered else "self"
    key_field = "sum" if rank_by == "inclusive" else "self_sum"

    def _set_pct_total(entries):
        for e in entries:
            e["pct_total"] = (e[key_field] / total * 100.0) if total > 0 else None

    selected, _desc = select_entries(
        cpu_entries, rank_field=key_field, threshold_field="pct_total", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="of total runtime", prepare=_set_pct_total,
    )
    return sorted({e["label"] for e in selected})


def labels_from_report(report_path):
    """Reads the 'CPU compute hotspots' table from a report written by
    extract_CPU_hotspots.py or extract_hotspots.py -- both render it via the
    exact same CPU_HOTSPOTS_COLUMNS layout, so one parser covers both. Rows are
    taken as-is: whatever selection produced the report is trusted."""
    with open(report_path, errors="replace") as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if "CPU compute hotspots" in line:
            start = i
            break
    if start is None:
        raise SystemExit(
            f"error: no 'CPU compute hotspots' section found in {report_path!r} -- "
            "expected a report written by extract_CPU_hotspots.py or extract_hotspots.py"
        )

    header = None
    for i in range(start, len(lines)):
        if lines[i].rstrip().endswith("function"):
            header = i
            break
    if header is None:
        raise SystemExit(
            f"error: found a 'CPU compute hotspots' section in {report_path!r} but no table header after it"
        )

    labels = []
    # CPU_HOTSPOTS_COLUMNS: # self(s) %total total(s) calls %self function -- 7 columns, the
    # function name (which may itself contain spaces, e.g. a C++ signature, and may itself span
    # 2+ physical lines if wrap_trailing_label() hard-wrapped it) intact as the 7th and last token.
    for row in iter_table_rows(lines[header + 1:], num_columns=7):
        if len(row) < 7:
            continue
        labels.append(row[6].strip())

    if not labels:
        raise SystemExit(
            f"error: 'CPU compute hotspots' section in {report_path!r} has no rows -- nothing to instrument"
        )

    return sorted(set(labels))


def flat_tree_for_ancestors(report_path, output_dir):
    """The flat, parent-linked row pool expand_labels_with_ancestors() needs, built from
    whichever source mode is active. --report mode requires a sibling 'calltree.txt' next to
    the report (every launcher that writes a hotspots.txt also writes one, per this project's
    own convention) parsed back into rows via flat_rows_from_calltree_text(); --output-dir mode
    builds it directly via load_rank_trees()/merge_rank_trees()/flatten_tree(), the exact same
    pipeline extract_hotspot_callers.py already uses for its own caller-chain lookups."""
    if report_path:
        calltree_path = os.path.join(os.path.dirname(os.path.abspath(report_path)), "calltree.txt")
        if not os.path.isfile(calltree_path):
            raise SystemExit(
                f"error: --ancestor-depth > 0 with --report needs a sibling 'calltree.txt' next to "
                f"{report_path!r}, but {calltree_path!r} wasn't found -- pass --ancestor-depth 0 to "
                "skip ancestor expansion, or use --output-dir instead"
            )
        with open(calltree_path, errors="replace") as f:
            text = f.read()
        return flat_rows_from_calltree_text(text)

    ranks = load_rank_trees(output_dir, "wall_clock-*.txt", "sampling_wall_clock-*.txt", postprocess=strip_wrapper_noise)
    if not ranks:
        return []
    return flatten_tree(merge_rank_trees(ranks))


def _iter_instrumented_entries(data):
    """instrumented.json's exact top-level shape (bare array vs. wrapped in an
    object) isn't independently confirmed beyond the per-entry field names --
    handle the plain-array case (the expected one) and, best-effort, a
    dict wrapping a single list, same spirit as this project's other
    undocumented-JSON-schema handling (metadata.json, config.json)."""
    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                yield from value
                return


def find_lost_functions(instrumented_json_path, requested_labels):
    try:
        with open(instrumented_json_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentedFileError(str(exc)) from exc

    names = set()
    for entry in _iter_instrumented_entries(data):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if isinstance(function, str):
            names.add(function)
        signature = entry.get("signature")
        if isinstance(signature, dict):
            sig_name = signature.get("name")
            if isinstance(sig_name, str):
                names.add(sig_name)

    return [label for label in requested_labels if not any(label in name for name in names)]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="select_instrumented_functions.py",
        description=HELP_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--report", dest="report", default=None,
                         help="hotspots.txt report to read CPU hotspot functions from")
    source.add_argument("--output-dir", dest="output_dir", default=None,
                         help="rocprof-sys output directory to read CPU hotspot functions from")
    stage6_cli_common.add_selection_args(parser, "functions", "of total runtime (default: 1.0)",
                                          top_noun="hotspot functions",
                                          top_help_suffix=" (default: 1%% threshold)", verb="select")
    parser.add_argument("--unfiltered", dest="unfiltered", action="store_true",
                         help="with --output-dir, select by inclusive (total) time instead of "
                              "self time -- can pick a function that just calls other functions "
                              "rather than one that does real work; ignored with --report (that "
                              "just reads whatever's in the file)")
    parser.add_argument("--gpu-output-dir", dest="gpu_output_dir", default=None,
                         help="rocprofv3 output directory paired with this run -- when given, GPU "
                              "kernel hotspots are selected from it (same top/threshold/--all as "
                              "the CPU side) and resolved into their real CPU owner subroutine "
                              "names (stage4_rocprofsys_common.resolve_kernel_owners()), which are "
                              "added to the selection even if the owner wasn't itself a hotspot")
    parser.add_argument("--ancestor-depth", dest="ancestor_depth", type=int, default=None,
                         help="pull in each selected function's N nearest real callers, so its "
                              "position in the resulting trace stays meaningful (default: 1; pass "
                              "0 to disable). With --report, requires a sibling 'calltree.txt' "
                              "next to it (written automatically by every launcher that writes a "
                              "hotspots.txt)")
    parser.add_argument("--check-instrumented", dest="check_instrumented", default=None,
                         help="switch to lost-function mode: read requested labels from stdin "
                              "(one per line) and warn about any missing from this "
                              "rocprof-sys-instrument instrumented.json file")
    parser.add_argument("--extra-noise-config", dest="extra_noise_config", default=None,
                         help="with --output-dir, path to a JSON file customizing noise-tag "
                              "patterns (add/remove substrings, disable a tag) -- see "
                              "stage6_noise_config.py's configure() for the file schema; falls "
                              "back to $FRIENDLY_ROCPROF_NOISE_CONFIG if not given; can't be "
                              "combined with --report (that just reads whatever's in the file, "
                              "already tagged)")
    args = parser.parse_args(argv)

    if args.report and args.extra_noise_config:
        raise SystemExit("error: --extra-noise-config can't be combined with --report")

    if args.check_instrumented:
        if (args.report or args.output_dir or args.top is not None or args.threshold is not None
                or args.show_all or args.unfiltered or args.extra_noise_config
                or args.gpu_output_dir or args.ancestor_depth is not None):
            raise SystemExit(
                "error: --check-instrumented can't be combined with "
                "--report/--output-dir/selection flags/--gpu-output-dir/--ancestor-depth"
            )
        labels = [line.strip() for line in sys.stdin if line.strip()]
        try:
            lost = find_lost_functions(args.check_instrumented, labels)
        except InstrumentedFileError as exc:
            print(
                f"note: couldn't read {args.check_instrumented!r} ({exc}) -- skipping lost-function check",
                file=sys.stderr,
            )
            return
        for label in lost:
            print(
                f"warning: hotspot function '{label}' wasn't found in the instrumented binary -- "
                "it may be inlined, optimized out, or use a name rocprof-sys-instrument didn't match; "
                "this doesn't stop anything, but that function won't show up in the trace",
                file=sys.stderr,
            )
        return

    if not args.report and not args.output_dir:
        raise SystemExit("error: one of --report or --output-dir is required")

    if args.gpu_output_dir:
        stage6_cli_common.require_directory(args.gpu_output_dir)

    if args.top is None and args.threshold is None and not args.show_all:
        args.threshold = 1.0
    if args.ancestor_depth is None:
        args.ancestor_depth = 1

    if args.report:
        hotspot_labels = labels_from_report(args.report)
    else:
        stage6_cli_common.require_directory(args.output_dir)
        stage6_noise_config.configure_from_args(args)
        hotspot_labels = labels_from_output_dir(
            args.output_dir, top=args.top, threshold=args.threshold, show_all=args.show_all,
            unfiltered=args.unfiltered,
        )

    if not hotspot_labels:
        raise SystemExit("error: no hotspot functions resolved -- nothing to instrument")

    print(
        f"{len(hotspot_labels)} hotspot function(s) selected by "
        f"{'inclusive' if args.unfiltered else 'self'} time", file=sys.stderr,
    )
    selected = set(hotspot_labels)

    owner_labels = set()
    if args.gpu_output_dir:
        kernel_labels = select_hotspot_kernels.labels_from_output_dir(
            args.gpu_output_dir, top=args.top, threshold=args.threshold, show_all=args.show_all,
        )
        owner_labels = resolve_kernel_owners(kernel_labels) - selected
        if owner_labels:
            print(
                f"{len(owner_labels)} GPU-kernel-owner function(s) added from --gpu-output-dir",
                file=sys.stderr,
            )
        selected |= owner_labels

    if args.ancestor_depth > 0:
        flat = flat_tree_for_ancestors(args.report, args.output_dir)
        ancestor_labels = expand_labels_with_ancestors(flat, selected, args.ancestor_depth)
        if ancestor_labels:
            print(
                f"{len(ancestor_labels)} ancestor function(s) added for tree connectivity "
                f"(--ancestor-depth {args.ancestor_depth})", file=sys.stderr,
            )
        selected |= ancestor_labels

    for label in sorted(selected):
        print(f"{label}\t{escape_for_instrument_regex(label)}")


if __name__ == "__main__":
    main()
