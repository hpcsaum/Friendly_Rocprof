"""Stage 5 load-imbalance table: shared by both the CPU and GPU hotspots tools.

Scope: per-label avg/std_dev/min/max of each rank's own contribution to that label, across ranks --
one table concept used identically by both the CPU (rocprof-sys) and GPU (rocprofv3) tools, so it
lives in one file rather than being duplicated per domain. compute_load_imbalance() is the only
consumer of stage4_rank_merge_math's per-rank stats; ranking/filtering/rendering themselves are generic
(see stage5_table_render.py).

Functions: load_imbalance_columns(), compute_load_imbalance(), imbalance_note().
"""

from stage4_rank_merge_math import stats_across_ranks
from stage5_table_render import select_entries

_VERB_BY_LABEL = {"function": "called", "kernel": "launched"}


def imbalance_note(item_label, time_kind):
    """Bulleted note explaining this table's per-rank convention -- item_label is "function" or
    "kernel" (matches load_imbalance_columns()'s own item_label), time_kind is "self"/"inclusive"
    (CPU tool, selected by --unfiltered) or "total" (GPU tool, no self/inclusive split since a
    kernel has no callees)."""
    verb = _VERB_BY_LABEL[item_label]
    return (
        f"  - Each {item_label}'s own {time_kind} time on each rank, compared across ranks -- a "
        f"rank that never {verb} a {item_label} counts as 0.0 for that rank, not omitted.\n"
    )


def load_imbalance_columns(item_label="function"):
    """item_label is the trailing column's header text -- "function" for the CPU tool, "kernel"
    for the GPU tool; everything else about the table is identical between the two."""
    return [
        {"header": "#", "width": 3, "value": lambda e, i: str(i)},
        {"header": "avg(s)", "width": 12, "value": lambda e, i: f"{e['avg']:.6f}"},
        {"header": "std_dev", "width": 10, "value": lambda e, i: f"{e['std_dev']:.6f}"},
        {"header": "min(s)", "width": 12, "value": lambda e, i: f"{e['min']:.6f}"},
        {"header": "max(s)", "width": 12, "value": lambda e, i: f"{e['max']:.6f}"},
        {"header": item_label, "width": None, "value": lambda e, i: e["label"]},
    ]


def compute_load_imbalance(per_file_totals, top=None, threshold=None, show_all=False):
    """Per-label avg/std_dev/min/max of each rank's own total time in that
    label, across all ranks in per_file_totals. A rank that never shows up
    for a given label contributes 0.0 (it genuinely spent no time there),
    not a skipped/missing value -- a function that only runs on some ranks
    is real, extreme imbalance, not something to hide.

    Selection mirrors select_entries()'s top/threshold/show_all shape, but
    ranked by std_dev (not total time), and --threshold here means
    coefficient of variation (std_dev / avg, as a %) instead of % of total
    runtime -- a %-of-runtime cutoff has no equivalent meaning for a
    std_dev ranking. Returns (selected, description), same shape as
    select_entries().
    """
    labels = {label for ft in per_file_totals for label in ft}
    entries = []
    for label in labels:
        values = [ft.get(label, 0.0) for ft in per_file_totals]
        stats = stats_across_ranks(values)
        entries.append({
            "label": label,
            "avg": stats["avg"],
            "std_dev": stats["std_dev"],
            "min": stats["min"],
            "max": stats["max"],
            "cv_pct": (stats["std_dev"] / stats["avg"] * 100.0) if stats["avg"] > 0 else None,
        })

    return select_entries(
        entries, rank_field="std_dev", threshold_field="cv_pct", top=top, threshold=threshold,
        show_all=show_all, threshold_unit="coefficient of variation", rank_label="std_dev",
    )
