"""Stage 1 (read profile) for rocprofv3 kernel_stats.csv output.

Scope: parsing rocprofv3's per-rank kernel_stats.csv files (produced by
`rocprofv3 --kernel-trace --stats --output-format csv`) into plain row dicts. Owns the CSV
column schema only -- has no opinion on which kernels matter or how they get aggregated; every
later stage builds on top of the rows this module returns.

Functions: parse_kernel_stats_csv().
"""

import csv

# "Name" is the only quoted CSV field; numeric columns may be plain-decimal or
# scientific notation depending on magnitude -- float()/int(float()) handles both.
REQUIRED_COLUMNS = {"Name", "Calls", "TotalDurationNs"}


def parse_kernel_stats_csv(path):
    """Parse one kernel_stats.csv. Returns a list of dict rows, or None if this
    file doesn't have the expected columns at all."""
    with open(path, "r", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            return None
        rows = []
        for row in reader:
            try:
                rows.append({
                    "label": row["Name"],
                    "count": int(float(row["Calls"])),
                    "total_ns": float(row["TotalDurationNs"]),
                })
            except (ValueError, TypeError, KeyError):
                continue
        return rows
