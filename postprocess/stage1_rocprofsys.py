"""Stage 1 (read profile) for rocprof-sys text-table output.

Scope: parsing rocprof-sys's pipe-delimited timemory text tables (e.g. wall_clock-<pid>.txt,
sampling_wall_clock-<pid>.txt) into plain row dicts. Owns the header/column format and the
per-row LABEL-field cleanup (rank/thread prefix, hierarchy indentation, thread id). Has no
opinion on what a row means (noise, GPU, MPI, ...) or what happens to it afterward -- every
later stage builds on top of the rows this module returns.

Functions: parse_table_file(), clean_label(), thread_id_from_raw_label().
"""

import re

# Matches rocprof-sys's per-process filename pattern "<component>-<pid>.txt" -- used to
# group files by rank, and as a fallback rank count when metadata.json has none.
PID_SUFFIX_RE = re.compile(r"-(\d+)\.txt$")

# The table's header row, matched as one fixed block rather than by individual field name:
# a row's LABEL can itself contain a literal "|" (MPI's rank/thread prefix is
# "|MM|NN>>>label"), which would misalign a naive per-field column split.
EXPECTED_HEADER_FIELDS = ["LABEL", "COUNT", "DEPTH", "METRIC", "UNITS", "SUM", "MEAN", "MIN", "MAX", "VAR", "STDDEV", "% SELF"]
FIXED_FIELDS_AFTER_LABEL = len(EXPECTED_HEADER_FIELDS) - 1


def parse_table_file(path):
    """Parse one timemory pipe-delimited text table. Returns a list of dict rows, or None if this
    file doesn't look like a timemory table at all."""
    with open(path, "r", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    header_idx = None
    for i, line in enumerate(lines):
        fields = [c.strip() for c in line.strip("|").split("|")]
        if fields == EXPECTED_HEADER_FIELDS:
            header_idx = i
            break
    if header_idx is None:
        return None

    rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        fields = [c.strip() for c in stripped.strip("|").split("|")]
        if len(fields) < FIXED_FIELDS_AFTER_LABEL + 1:
            continue
        # Everything before the last FIXED_FIELDS_AFTER_LABEL fields is the label --
        # rejoined with "|" in case a rank prefix split it into more than one piece.
        raw_label = "|".join(fields[:-FIXED_FIELDS_AFTER_LABEL])
        count, depth, metric, units, total, mean, vmin, vmax, var, stddev, pct_self = fields[-FIXED_FIELDS_AFTER_LABEL:]
        label = clean_label(raw_label)
        try:
            total_f = float(total)
            rows.append({
                "label": label,
                "count": int(count),
                "sum": total_f,
                # % SELF is only meaningful per call-tree node, so it's converted to an
                # absolute self-time here and kept as a plain seconds value from then on.
                "self_sum": total_f * float(pct_self) / 100.0,
                "depth": int(depth),
                "thread_id": thread_id_from_raw_label(raw_label),
            })
        except ValueError:
            continue
    return rows


def thread_id_from_raw_label(raw_label):
    """The OS-thread index a raw LABEL field's prefix identifies -- the last
    "|"-delimited segment before ">>>" (e.g. "|1>>>foo" -> "1"; the MPI form
    "00|00>>>foo" -> "00", the rank/thread pair's thread half)."""
    prefix = raw_label.split(">>>", 1)[0]
    segments = [s for s in prefix.split("|") if s != ""]
    return segments[-1] if segments else ""


def clean_label(raw_label):
    """Strip rocprof-sys's thread/rank prefix (|NN>>> or |MM|NN>>>) and hierarchy
    indentation (|_ repeated per call-stack depth) from a raw LABEL field."""
    label = raw_label
    if ">>>" in label:
        label = label.split(">>>", 1)[1]
    while label.startswith("|_"):
        label = label[2:]
    return label.strip()
