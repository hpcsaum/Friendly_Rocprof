"""Stage 1+2 (read profile, resolve ancestry) for rocprof-sys's Perfetto trace-CSV output.

Scope: parsing the flat trace-CSV export (one row per real call/event instance, converted from a
rocprof-sys `.proto` trace via Perfetto's own `trace_processor` -- a separate, user-run step, not
part of this codebase) and resolving the call-tree links Perfetto already computed. Every CSV
column is preserved on every row, unfiltered -- this module has no opinion on what a row means
(noise, GPU, MPI, ...), what a "count" or "self time" should be, or what happens to a row
afterward. In particular: `label`/`count`/`self_sum` are NOT synthesized here -- deciding what
those mean for a given consumer is that consumer's job (a later stage4 module, building the shape
`stage4_rocprofsys_sample_tree.merge_rank_trees()` needs), not this parser's. Only `slice_id`/
`parent_slice_id` (needed for the ancestry lookup below) and `ts`/`dur` (converted from the CSV's
nanoseconds to the seconds convention every other stage in this codebase uses -- a lossless,
reversible normalization, not a loss of information) get any type handling at all; every other
column, including the long, sparse tail of GPU-dispatch-specific columns, passes through exactly
as the CSV gave it.

Functions: parse_trace_csv(), attach_ancestry().
"""

import csv

# slice_id/parent_slice_id need to be real ints to work as attach_ancestry()'s lookup keys;
# ts/dur get the nanoseconds-to-seconds conversion every other stage's "sum"/"self_sum"
# convention assumes. Every other column (name, category, tid, pid, depth, the sparse GPU-arg
# columns, ...) is left exactly as csv.DictReader parsed it -- this module has no opinion on it.
_INT_COLUMNS = ("slice_id", "parent_slice_id")
_SECONDS_COLUMNS = ("ts", "dur")


def parse_trace_csv(paths):
    """Reads one or more flat trace-CSV files (a single path, or a list of paths -- e.g. this
    session's own gpu/mpi/other category-partitioned split for one rank) into a list of row
    dicts, one per CSV row, concatenated in the order `paths` were given. Every CSV column becomes
    a dict key, unfiltered; an empty cell becomes None, everything else stays a raw string except
    for the minimal type handling described in this module's own docstring. Rows from files with
    different headers can end up with different key sets in the same returned list -- callers
    should read with `.get()`, the same convention `stage4_rocprofsys_sample_tree.merge_rank_trees()`
    already uses for optional fields.

    A row is skipped entirely (matching `stage1_rocprofsys_sample.parse_table_file()`'s existing
    skip-on-bad-row precedent) only if `ts`/`dur` don't parse as numbers at all -- not for any
    other column being missing or empty, since sparse columns are the normal case here.
    """
    if isinstance(paths, str):
        paths = [paths]

    rows = []
    for path in paths:
        with open(path, "r", newline="", errors="replace") as f:
            for raw_row in csv.DictReader(f):
                row = {k: (v if v else None) for k, v in raw_row.items()}

                try:
                    for col in _SECONDS_COLUMNS:
                        row[col] = float(row[col]) / 1e9
                except (TypeError, ValueError):
                    continue

                for col in _INT_COLUMNS:
                    if row.get(col) is not None:
                        row[col] = int(row[col])

                rows.append(row)

    return rows


def attach_ancestry(rows):
    """Resolves each row's `parent_slice_id` into an actual `parent` object reference (another
    dict in `rows`, or None) -- a direct lookup, not a stack walk, since Perfetto already computed
    the real tree. `parent_slice_id` is left on the row afterward (this module never removes a
    column it was given). Mutates `rows` in place and returns it, same signature shape as
    `stage2_rocprofsys_sample.attach_ancestry()`.

    A row with `parent_slice_id is None` is a legitimate untethered root (e.g. a
    rocm_kernel_dispatch row on its own GPU-queue track, never nested in any CPU thread's tree) --
    `parent` becomes None silently, no warning. A row whose `parent_slice_id` is set but doesn't
    match any row actually present in `rows` (the partial-input case -- e.g. only one of a
    category-partitioned file set was given) also gets `parent = None`, but is counted for one
    summary warning covering every such row, not one warning each.

    This function computes nothing else -- no self_sum, no roots list. `roots = [r for r in rows
    if r["parent"] is None]` is one line the caller makes itself, same as
    `stage4_rocprofsys_sample_tree.load_rank_trees()` already does for the existing text-table
    format.
    """
    by_slice_id = {row["slice_id"]: row for row in rows if row.get("slice_id") is not None}

    unresolved_count = 0
    for row in rows:
        parent_id = row.get("parent_slice_id")
        if parent_id is None:
            row["parent"] = None
            continue
        parent = by_slice_id.get(parent_id)
        if parent is None:
            unresolved_count += 1
        row["parent"] = parent

    if unresolved_count:
        print(
            f"warning: {unresolved_count} row(s) reference a parent_slice_id not present in the "
            "given data -- treating as new roots rather than guess a parent (pass the complete "
            "category-partitioned file set, or the single unfiltered CSV, for this rank to avoid "
            "this)",
        )

    return rows
