"""Cross-rank statistics primitive.

Scope: turning a list of per-rank numeric values into avg/std_dev/min/max. Has no opinion on
where those values came from (a call-tree node's self-time, a function's total time by label,
a kernel's total device time, ...) or what a caller does with the result -- every per-node tree
statistic and load-imbalance table in this codebase is built on top of this one primitive.

Functions: stats_across_ranks().
"""

import statistics


def stats_across_ranks(values):
    """avg/std_dev/min/max across a list of per-rank values -- an empty list (no ranks to
    compare) returns all zeros rather than raising. std_dev uses statistics.pstdev()
    (population, not sample): every rank that exists is already the whole population being
    compared, not a sample drawn from a larger one."""
    if not values:
        return {"avg": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": statistics.mean(values),
        "std_dev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }
