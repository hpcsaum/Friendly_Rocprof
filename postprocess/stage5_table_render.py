"""Generic stage-5 table backend: ranking/filtering entries and rendering them as text.

Scope: two domain-agnostic primitives every stage-5 table module builds on. select_entries() ranks,
threshold-filters, and truncates a list of entries by whichever fields the caller names -- it has
no idea whether "entries" are CPU hotspots, GPU kernels, or per-label load-imbalance stats, just
that each entry is a dict with numeric fields. render_table() turns a list of entries into aligned
text given a column spec (a list of {"header", "width", "value"} dicts) -- it has no idea what a
column means, only how wide it is and how to read its value out of an entry. What differs between
tables (which columns, which field to rank by, which field means "% of total") is data supplied by
the caller, not new code here.

Functions: select_entries(), render_table().
"""


def select_entries(entries, rank_field, threshold_field=None, top=None, threshold=None,
                    show_all=False, tie_break_field="label", threshold_unit="of total",
                    rank_label=None, prepare=None):
    """Ranks/filters/truncates entries for a report table. prepare, if given, is called as
    prepare(entries) first, for any in-place data prep a caller needs before ranking (e.g. CPU
    hotspots' self-vs-inclusive pct_total recompute) -- omit it for callers with nothing to prep.

    entries are sorted descending by rank_field, ties broken ascending by tie_break_field --
    deterministic regardless of dict/set iteration order (Python's per-process string-hash
    randomization otherwise lets two entries with an identical rank_field value print in either
    order between runs).

    Exactly one selection mode applies (show_all > threshold > top, in that precedence, though
    callers should only set one): show_all returns everything; threshold keeps entries whose
    threshold_field is at or above threshold (as a %) -- unless EVERY entry's threshold_field is
    None (the denominator that would have produced it was unknown/zero), in which case every entry
    is returned instead of an empty list, with a message saying so; top keeps the top N (default
    20) by rank_field. rank_label, if given, is appended to the "top N" description (e.g. "top 20
    of 40 entries by std_dev") -- omit it where the ranked field is already obvious from context.
    Returns (selected_entries, description_for_report_header).
    """
    if prepare is not None:
        prepare(entries)

    entries_sorted = sorted(entries, key=lambda e: (-e[rank_field], e[tie_break_field]))
    total_count = len(entries_sorted)

    if show_all:
        return entries_sorted, f"all {total_count} entries"

    if threshold is not None:
        if all(e.get(threshold_field) is None for e in entries_sorted):
            unknown_label = threshold_unit[3:] if threshold_unit.startswith("of ") else threshold_unit
            return entries_sorted, f"all {total_count} entries ({unknown_label} unknown, threshold ignored)"
        filtered = [e for e in entries_sorted if e[threshold_field] is not None and e[threshold_field] >= threshold]
        return filtered, f">= {threshold:g}% {threshold_unit} ({len(filtered)} of {total_count} entries)"

    n = 20 if top is None else top
    suffix = f" by {rank_label}" if rank_label else ""
    return entries_sorted[:n], f"top {n} of {total_count} entries{suffix}"


def render_table(columns, entries):
    """columns: list of {"header": str, "width": int|None, "align": "left"|"right" (default
    "right"), "value": callable(entry, index) -> str}. width=None means unpadded (used for the
    trailing name/label column, which is never truncated -- its own value is printed as-is,
    whatever length it is). Renders one header row plus one row per entry, 1-indexed; returns
    "  (none found)\\n" for an empty entries list (no header printed with nothing under it)."""
    if not entries:
        return "  (none found)\n"

    def _cell(text, col):
        width = col.get("width")
        if width is None:
            return text
        align = ">" if col.get("align", "right") == "right" else "<"
        return f"{text:{align}{width}}"

    lines = ["  " + "  ".join(_cell(col["header"], col) for col in columns)]
    for i, e in enumerate(entries, 1):
        lines.append("  " + "  ".join(_cell(col["value"](e, i), col) for col in columns))
    return "\n".join(lines) + "\n"
