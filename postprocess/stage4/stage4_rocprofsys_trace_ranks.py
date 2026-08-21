"""Stage 4 (multi-rank file discovery) for rocprof-sys's Perfetto trace-CSV pipeline.

Scope: turning a directory of trace-CSV files into the `rank_inputs` shape
(stage4_rocprofsys_trace_tree.merge_ranks()/stage4_rocprofsys_trace_flat.*() already expect --
a list of (rank_key, csv_paths) tuples) -- nothing else. This is *this project's own* documented
naming convention (`perfetto-trace-<N>.csv` / `perfetto-trace-<N>-{gpu,mpi,other}.csv`), not
something any upstream AMD tool guarantees -- the `.proto`-to-CSV conversion step is a separate,
user-run tool with no fixed filename convention of its own. A user with differently-named CSVs
renames them to match.

Functions: discover_ranks().
"""

import glob
import os
import re

_RANK_FILE_RE = re.compile(r"-(\d+)(?:-(gpu|mpi|other))?\.csv$")


def discover_ranks(trace_dir):
    """Scans trace_dir (recursively) for *.csv files matching this project's documented rank/
    category naming convention, groups them by rank, and returns a list of (rank_key, csv_paths)
    tuples sorted by NUMERIC rank (not string order, so rank 10 doesn't sort before rank 2) --
    ready to feed directly into merge_ranks()/aggregate()/aggregate_per_rank()/
    gather_timing_summary_per_rank().

    Per rank: the category-partitioned trio (-gpu/-mpi/-other) is preferred over the single
    unfiltered file when both exist for that rank. The two forms are an exact row-for-row
    partition of the same underlying data (the partitioned trio's row counts sum to the unfiltered
    file's own row count), so that alone wouldn't justify a preference -- but the unfiltered file
    isn't guaranteed to carry every column the partitioned files do (in particular, it can be
    missing corr_id and the other wide GPU-arg columns). Preferring the partitioned set when
    present avoids silently degrading the corr_id join to "nothing ever joins" purely because of
    which file happened to be picked. The unfiltered file is used only when no partitioned file
    exists for that rank at all.

    Raises SystemExit if nothing under trace_dir matches the naming convention at all.
    """
    unfiltered_by_rank = {}
    partitioned_by_rank = {}

    for path in glob.glob(os.path.join(trace_dir, "**", "*.csv"), recursive=True):
        m = _RANK_FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        rank_key, category = m.group(1), m.group(2)
        if category is None:
            unfiltered_by_rank[rank_key] = path
        else:
            partitioned_by_rank.setdefault(rank_key, []).append(path)

    rank_keys = set(unfiltered_by_rank) | set(partitioned_by_rank)
    if not rank_keys:
        raise SystemExit(
            f"error: no trace-CSV files matching '-<rank>[-gpu|-mpi|-other].csv' found under "
            f"{trace_dir!r}"
        )

    rank_inputs = []
    for rank_key in sorted(rank_keys, key=int):
        if rank_key in partitioned_by_rank:
            rank_inputs.append((rank_key, sorted(partitioned_by_rank[rank_key])))
        else:
            rank_inputs.append((rank_key, unfiltered_by_rank[rank_key]))

    return rank_inputs
