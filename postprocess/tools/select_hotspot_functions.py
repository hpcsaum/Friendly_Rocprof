#!/usr/bin/env python3
"""Resolve CPU hotspot function names into rocprof-sys-instrument "-R" input.

Two unrelated jobs live in this one module, both in service of
scripts/instrument_hotspots.sh:

1. Resolve mode (default): read a hotspots report (written by
   extract_CPU_hotspots.py or extract_hotspots.py) or a rocprof-sys output
   directory directly, and print "label<TAB>regex" pairs -- the regex being
   an escaped, unanchored substring pattern safe to pass to
   rocprof-sys-instrument's "-R/--function-restrict".

2. --check-instrumented mode: after rocprof-sys-instrument has produced its
   own instrumented.json (documenting exactly which functions actually got
   instrumented, post-filtering), compare it against the requested labels
   and warn about any that didn't make it in. Never fails -- a lost
   function is a warning, not an error.

Only CPU-side function names are handled here. GPU kernel names (from a
combined report's fused/GPU tables) are a different mechanism entirely and
are never read by this module.

Functions: escape_for_instrument_regex(), labels_from_output_dir(), labels_from_report(),
find_lost_functions(), main().
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _stage_paths  # noqa: E402  (adds every stageN/ dir to sys.path)

from stage4_rocprofsys_sample_flat import aggregate
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

Only CPU-side functions are selected -- GPU kernel names from a combined
report can't be targeted this way, since kernel instrumentation is a
different mechanism entirely.

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
        prog="select_hotspot_functions.py",
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
                or args.show_all or args.unfiltered or args.extra_noise_config):
            raise SystemExit("error: --check-instrumented can't be combined with --report/--output-dir/selection flags")
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

    if args.top is None and args.threshold is None and not args.show_all:
        args.threshold = 1.0

    if args.report:
        labels = labels_from_report(args.report)
    else:
        stage6_cli_common.require_directory(args.output_dir)
        stage6_noise_config.configure_from_args(args)
        labels = labels_from_output_dir(
            args.output_dir, top=args.top, threshold=args.threshold, show_all=args.show_all,
            unfiltered=args.unfiltered,
        )

    if not labels:
        raise SystemExit("error: no hotspot functions resolved -- nothing to instrument")

    for label in labels:
        print(f"{label}\t{escape_for_instrument_regex(label)}")


if __name__ == "__main__":
    main()
