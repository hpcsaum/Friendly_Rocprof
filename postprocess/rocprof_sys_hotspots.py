#!/usr/bin/env python3
"""Extract a short CPU-side hotspots report from rocprof-sys timemory text output.

Only reads the well-documented pipe-delimited "timemory" text tables
(e.g. wall_clock-<pid>.txt) that rocprof-sys writes for CPU-side timing.
GPU device kernel execution time is NOT present in this data -- see the
footer note this script writes into its own output.
"""

import argparse
import glob
import os
import sys
from datetime import datetime

NON_TIMING_FILES = {"available.txt", "instrumented.txt", "excluded.txt", "overlapping.txt"}

GPU_API_PREFIXES = ("hip", "hsa", "roctx", "kfd", "rocdecode", "rocjpeg")
GPU_FILE_HINTS = ("roctracer", "hsa")

EXPECTED_HEADER_FIELDS = ["LABEL", "COUNT", "DEPTH", "METRIC", "UNITS", "SUM", "MEAN", "MIN", "MAX", "VAR", "STDDEV", "% SELF"]
# COUNT..% SELF -- everything after LABEL. Kept as a count, not the literal names,
# because under MPI the docs describe the row prefix as "|MM|NN>>>label" -- an
# embedded pipe that would otherwise misalign a naive fixed-column split.
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
            rows.append({
                "label": label,
                "count": int(count),
                "sum": float(total),
                "pct_self": float(pct_self),
            })
        except ValueError:
            continue
    return rows


def clean_label(raw_label):
    """Strip rocprof-sys's thread/rank prefix (|NN>>> or |MM|NN>>>) and hierarchy
    indentation (|_ repeated per call-stack depth) from a raw LABEL field."""
    label = raw_label
    if ">>>" in label:
        label = label.split(">>>", 1)[1]
    while label.startswith("|_"):
        label = label[2:]
    return label.strip()


def is_gpu_entry(label, filename):
    lname = label.lower()
    if lname.startswith(GPU_API_PREFIXES):
        return True
    fname = os.path.basename(filename).lower()
    return any(hint in fname for hint in GPU_FILE_HINTS)


def aggregate(output_dir):
    """Scan output_dir for timemory text tables and aggregate rows by clean function name.

    Returns (cpu_entries, gpu_entries, scanned_files) where each entries list is
    [{"label", "count", "sum", "pct_self_last"}], sorted by nothing yet.
    """
    scanned_files = []
    totals = {}  # label -> {"count": int, "sum": float, "pct_self": float, "gpu": bool}

    candidates = sorted(glob.glob(os.path.join(output_dir, "*.txt")))
    for path in candidates:
        if os.path.basename(path) in NON_TIMING_FILES:
            continue
        rows = parse_table_file(path)
        if rows is None:
            continue
        scanned_files.append(path)
        for row in rows:
            label = row["label"]
            gpu = is_gpu_entry(label, path)
            entry = totals.setdefault(label, {"count": 0, "sum": 0.0, "pct_self": 0.0, "gpu": gpu})
            entry["count"] += row["count"]
            entry["sum"] += row["sum"]
            entry["pct_self"] = row["pct_self"]
            entry["gpu"] = entry["gpu"] or gpu

    cpu_entries = []
    gpu_entries = []
    for label, entry in totals.items():
        item = {"label": label, "count": entry["count"], "sum": entry["sum"], "pct_self": entry["pct_self"]}
        (gpu_entries if entry["gpu"] else cpu_entries).append(item)

    return cpu_entries, gpu_entries, scanned_files


def format_table(entries, top_n):
    entries = sorted(entries, key=lambda e: e["sum"], reverse=True)[:top_n]
    if not entries:
        return "  (none found)\n"
    lines = []
    lines.append(f"  {'#':>3}  {'total(s)':>12}  {'calls':>10}  {'%self':>7}  function")
    for i, e in enumerate(entries, 1):
        lines.append(f"  {i:>3}  {e['sum']:>12.6f}  {e['count']:>10}  {e['pct_self']:>7.1f}  {e['label']}")
    return "\n".join(lines) + "\n"


def find_extra_artifacts(output_dir):
    proto_files = sorted(glob.glob(os.path.join(output_dir, "*.proto")))
    db_files = sorted(glob.glob(os.path.join(output_dir, "*.db")))
    return proto_files, db_files


def write_report(output_dir, dest_path, top_n):
    cpu_entries, gpu_entries, scanned_files = aggregate(output_dir)
    if not scanned_files:
        raise SystemExit(
            f"error: no rocprof-sys timemory text table found in {output_dir!r} "
            "(expected files like wall_clock-<pid>.txt with the documented "
            "LABEL|COUNT|DEPTH|METRIC|... header) -- nothing to report"
        )

    proto_files, db_files = find_extra_artifacts(output_dir)

    parts = []
    parts.append("rocprof-sys hotspots report (CPU-side only)\n")
    parts.append(f"generated: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"source directory: {os.path.abspath(output_dir)}\n")
    parts.append("files scanned:\n")
    for f in scanned_files:
        parts.append(f"  - {os.path.basename(f)}\n")
    parts.append("\n")

    parts.append(f"Top {top_n} CPU compute hotspots (candidates for GPU offload)\n")
    parts.append(format_table(cpu_entries, top_n))
    parts.append("\n")

    parts.append(f"Top {top_n} GPU API / launch overhead\n")
    parts.append("(host-side call overhead only -- NOT device kernel execution time)\n")
    parts.append(format_table(gpu_entries, top_n))
    parts.append("\n")

    parts.append(
        "Note: true GPU kernel execution time is not present in this data. "
        "rocprof-sys's text/JSON output only captures host-side timing; "
        "GPU-launch-looking rows above are launch/API overhead, not device time.\n"
    )
    if proto_files:
        parts.append("A Perfetto trace was also found (not parsed by this tool):\n")
        for f in proto_files:
            parts.append(f"  - {f}\n")
    if db_files:
        parts.append("A rocpd database was also found (ROCm 7.1+ only, not parsed by this tool):\n")
        for f in db_files:
            parts.append(f"  - {f}\n")
    parts.append(
        "For real GPU kernel hotspots, run: rocprofv3 --stats --kernel-trace --summary -- <app>\n"
    )

    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="rocprof-sys output directory to read")
    parser.add_argument("-o", "--output", dest="dest", default=None,
                         help="path to write the hotspots report (default: <output_dir>/hotspots.txt)")
    parser.add_argument("-n", "--top", dest="top_n", type=int, default=20,
                         help="number of hotspots to list per section (default: 20)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"error: no such directory: {args.output_dir!r}")

    dest = args.dest or os.path.join(args.output_dir, "hotspots.txt")
    write_report(args.output_dir, dest, args.top_n)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
