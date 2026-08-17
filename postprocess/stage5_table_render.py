"""Generic stage-5 table backend: ranking/filtering entries and rendering them as text.

Scope: two domain-agnostic primitives every stage-5 table module builds on. select_entries() ranks,
threshold-filters, and truncates a list of entries by whichever fields the caller names -- it has
no idea whether "entries" are CPU hotspots, GPU kernels, or per-label load-imbalance stats, just
that each entry is a dict with numeric fields. render_table() turns a list of entries into aligned
text given a column spec (a list of {"header", "width", "value"} dicts) -- it has no idea what a
column means, only how wide it is and how to read its value out of an entry. What differs between
tables (which columns, which field to rank by, which field means "% of total") is data supplied by
the caller, not new code here.

Functions: select_entries(), render_table(), wrap_trailing_label(), iter_table_rows(),
pct_total_note(), ranking_note().
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


def wrap_trailing_label(prefix, label, width=120):
    """Formats one row whose line already begins with `prefix` (everything printed before the
    label -- e.g. "  1  12.340000  10.0  ...  ") and ends in an unpadded trailing label. Returns
    the row as a single string (1+ physical lines joined by "\\n"): unchanged (`prefix + label`)
    if it already fits in `width` columns; otherwise hard-wrapped -- a raw character-level cut, no
    word-boundary search, so concatenating every chunk back together reconstructs `label` exactly
    -- with each continuation line indented by len(prefix) spaces so it lines up exactly under
    where the label itself started (never flush-left). The 20-column floor on the available width
    guards against a pathological prefix that alone already exceeds `width` (none do today; a
    defensive minimum, not a real code path). Reusable as-is by any future stage-5 table module
    sharing this "fixed columns, then one unpadded trailing label" layout, not just render_table().
    """
    if len(prefix) + len(label) <= width:
        return prefix + label
    available = max(width - len(prefix), 20)
    chunks = [label[i:i + available] for i in range(0, len(label), available)]
    indent = " " * len(prefix)
    return "\n".join([prefix + chunks[0]] + [indent + c for c in chunks[1:]])


def render_table(columns, entries):
    """columns: list of {"header": str, "width": int|None, "align": "left"|"right" (default
    "right"), "value": callable(entry, index) -> str}. width=None means unpadded (used for the
    trailing name/label column, which is never truncated -- its own value is printed as-is,
    however long it is, hard-wrapped via wrap_trailing_label() if that would exceed a shared
    120-column target). Renders one header row plus one row per entry, 1-indexed; returns
    "  (none found)\\n" for an empty entries list (no header printed with nothing under it)."""
    if not entries:
        return "  (none found)\n"

    def _cell(text, col):
        width = col.get("width")
        if width is None:
            return text
        align = ">" if col.get("align", "right") == "right" else "<"
        return f"{text:{align}{width}}"

    # Only the trailing column is ever allowed width=None (an unpadded label, per this function's
    # own contract) -- every other column is always fixed-width, so only that specific shape needs
    # wrap_trailing_label()'s hard-wrap; a table with no such trailing column (e.g. a table made
    # entirely of fixed-width columns) renders exactly as it always has.
    has_trailing_label = bool(columns) and columns[-1].get("width") is None
    fixed_cols = columns[:-1] if has_trailing_label else columns

    def _row(fixed_values, label_value):
        cells = [_cell(text, col) for text, col in zip(fixed_values, fixed_cols)]
        prefix = "  " + "  ".join(cells)
        if not has_trailing_label:
            return prefix
        return wrap_trailing_label(prefix + ("  " if fixed_cols else ""), label_value)

    lines = [_row([col["header"] for col in fixed_cols], columns[-1]["header"] if has_trailing_label else None)]
    for i, e in enumerate(entries, 1):
        lines.append(_row(
            [col["value"](e, i) for col in fixed_cols],
            columns[-1]["value"](e, i) if has_trailing_label else None,
        ))
    return "\n".join(lines) + "\n"


def iter_table_rows(lines, num_columns):
    """Yields each logical row as a list of num_columns tokens, with a hard-wrapped trailing
    label already rejoined across any continuation lines wrap_trailing_label() introduced --
    callers never see the physical line breaks. lines is the table's own row lines (header
    already skipped by the caller); stops at the first blank line, matching render_table()'s own
    "blank line ends the table" convention.

    Detection: a line is a continuation of the previous row, not a new row, iff its first
    whitespace-split token doesn't parse as a plain integer -- every real row's leading '#'
    column always is one; a continuation line (a raw fragment of a wrapped label) essentially
    never is. Reconstruction is direct concatenation, no separator inserted -- the wrap is a
    lossless character-level cut, so undoing it is just gluing the pieces back in order.

    Known, accepted limitations: a wrap point landing exactly on a digit run could leave a
    continuation line whose own first token is pure digits, misread as a new row -- real
    C/C++/Fortran identifiers can't start with a digit, so this only bites mid-identifier at an
    exact cut point. A wrap point landing exactly on a space is lossy: each continuation line is
    stripped before being appended back, so that one space is dropped rather than preserved --
    real long labels needing this reader (namespace/template chains) are essentially one long
    contiguous identifier with no interior whitespace, so this practically never bites. Neither is
    worth the added complexity of closing here.
    """
    current = None
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            break
        first = line.split(maxsplit=1)[0] if line.split() else ""
        if first.lstrip("-").isdigit():
            if current is not None:
                yield current
            current = line.split(maxsplit=num_columns - 1)
        elif current is not None:
            current[-1] += line.strip()
    if current is not None:
        yield current


def pct_total_note(entry_noun, threshold_unit):
    """Bulleted note explaining what this table's %total column means -- threshold_unit is the
    exact same string already given to select_entries() (its "showing top N ... {threshold_unit}"
    line and this note now share one source, so they can't drift the way this codebase's %total
    sentences used to, hand-typed separately once per table)."""
    return f"  - '%total' is each {entry_noun}'s share {threshold_unit}.\n"


def ranking_note(unfiltered, extra_clause=""):
    """Bulleted note explaining which time basis (self vs. inclusive) ranked this table --
    extra_clause lets a caller append one more tool-specific sentence (e.g. extract_hotspots.py
    noting GPU kernels are unaffected, already leaf events)."""
    basis = (
        "inclusive (total) time -- a function that just calls other functions can still rank high"
        if unfiltered else
        "self time -- each function's own work, not counting time spent in whatever it calls, so "
        "pass-through functions fall out of the ranking on their own"
    )
    return f"  - Ranked by {basis}.{extra_clause}\n"
