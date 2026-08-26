"""Resolves the process-wide --time-range definition every trace-CSV report tool shares for its
run, and builds the one report-header line describing whatever time window is (or isn't) active.

Scope: parsing/storing the requested range (a module-level singleton, not threaded as a parameter
through every function between a tool's main() and stage4_rocprofsys_trace_aggregate.py -- same
reasoning as stage6_noise_config.py's own _TAG_DEFS: every real invocation of these tools is a
single, one-shot CLI process with exactly one active time range for its whole run), and describing
that active range for a report header. The "describing" half is a deliberate exception to this
codebase's usual stage6-doesn't-import-stage4 layering: producing a report header/footer note from
a stage4 fact (a trace's real time extent) is a genuinely different kind of output than a rendered
table or tree, has no reason to route through stage5's rendering machinery at all, and is shared
identically by every trace report tool -- so it's handled once, here, rather than duplicated per
tool. See postprocess/README.md's "Cross-stage imports" section for this pattern stated generally.

Functions: parse_time_range(), configure(), active_ranges(), add_cli_argument(),
configure_from_args(), describe_time_range().
"""

import stage4_rocprofsys_trace_aggregate

_RANGES = None


def _parse_bound(text):
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        raise SystemExit(f"error: --time-range: {text!r} isn't a number")


def _merge_ranges(segments):
    """Sorts (start_or_None treated as -inf for ordering) and merges overlapping-or-touching
    segments into a canonical, minimal, disjoint list -- a call landing exactly on the boundary
    between two given windows (end_i == start_{i+1}) is merged, not left ambiguous."""
    def sort_key(segment):
        start, _end = segment
        return float("-inf") if start is None else start

    merged = []
    for start, end in sorted(segments, key=sort_key):
        if not merged:
            merged.append([start, end])
            continue
        last_start, last_end = merged[-1]
        last_hi = float("inf") if last_end is None else last_end
        cur_lo = float("-inf") if start is None else start
        if cur_lo <= last_hi:
            merged[-1][1] = None if (end is None or last_end is None) else max(last_end, end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def parse_time_range(raw):
    """None in, None out. Otherwise splits `raw` on "," -- each segment must contain a literal ":"
    (a bare "10" is rejected, not silently guessed as a start or an end); splits into
    (start_str, end_str), an empty half means an open bound (None); at least one bound is required
    per segment, and start < end is enforced when both are given. Raises SystemExit with a clear
    message on any parse/validation failure. Returns the canonical, merged (see _merge_ranges())
    [(start_or_None, end_or_None), ...] list."""
    if raw is None:
        return None

    segments = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            raise SystemExit(
                f"error: --time-range segment {chunk!r} has no ':' -- expected START:END, "
                "'START:' (open end), or ':END' (open start)"
            )
        start_text, end_text = (part.strip() for part in chunk.split(":", 1))
        start = _parse_bound(start_text)
        end = _parse_bound(end_text)
        if start is None and end is None:
            raise SystemExit(f"error: --time-range segment {chunk!r} needs at least a start or an end")
        if start is not None and end is not None and start >= end:
            raise SystemExit(f"error: --time-range segment {chunk!r}: start must be less than end")
        segments.append((start, end))

    return _merge_ranges(segments)


def configure(raw):
    """Resolves and stores the active range for this process -- called once, early, by each CLI
    tool's main() right after parsing --time-range. Safe to call again (e.g. between tests, to
    reset or change the active range) -- always fully replaces any prior value, never merges."""
    global _RANGES
    _RANGES = parse_time_range(raw)


def active_ranges():
    """The current canonical ranges list, or None if no --time-range is active."""
    return _RANGES


def add_cli_argument(parser):
    """Adds --time-range to parser -- pair with configure_from_args() once the tool's own main()
    has parsed args."""
    parser.add_argument(
        "--time-range", dest="time_range", default=None,
        help="restrict this report to one or more time windows, in seconds -- 'START:END', "
             "'START:' (to the end), ':END' (from the start), or a comma-separated list of "
             "windows to combine (e.g. '5:12.5,20:'). A call straddling a window boundary still "
             "contributes its in-window portion of self/inclusive time; a call-tree subtree with "
             "no overlap anywhere within it is cut from the tree entirely, with the chain to any "
             "in-window descendant kept intact. Default: the whole run.",
    )


def configure_from_args(args):
    """Resolves --time-range from a parsed argparse Namespace and calls configure() with it -- the
    one line each tool's main() needs, right after parser.parse_args()."""
    configure(args.time_range)


def _format_note(run_extent, ranges):
    """One report-header line describing the time window a report reflects -- always a real
    string, never conditional: with no active range, formats run_extent itself (the real, full
    span); with an active range, formats each window, resolving any open start/end against
    run_extent's corresponding bound. Kept pure/argument-only (not reading the module's own global)
    so it's directly unit-testable without any global-state setup."""
    run_start, run_end = run_extent
    if not ranges:
        return f"  time range: {run_start:.3f}s-{run_end:.3f}s (full run)\n"
    parts = []
    for start, end in ranges:
        resolved_start = run_start if start is None else start
        resolved_end = run_end if end is None else end
        parts.append(f"{resolved_start:.3f}s-{resolved_end:.3f}s")
    return f"  time range: {', '.join(parts)}\n"


def describe_time_range(rank_inputs, cache_dir=None):
    """The one call each trace report tool's write_report() makes for its always-printed header
    note. Combines every rank's real, unfiltered extent (stage4_rocprofsys_trace_aggregate.
    get_rank_time_extent() -- the deliberate stage6 -> stage4 dependency this module's docstring
    explains) into one run-wide (start, end), reads the currently active range, and formats one
    line covering both cases via _format_note()."""
    extents = [
        stage4_rocprofsys_trace_aggregate.get_rank_time_extent(csv_paths, rank_key, cache_dir=cache_dir)
        for rank_key, csv_paths in rank_inputs
    ]
    run_extent = (min(s for s, _e in extents), max(e for _s, e in extents)) if extents else (0.0, 0.0)
    return _format_note(run_extent, active_ranges())
