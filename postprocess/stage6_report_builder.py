"""Stage 6 report assembly, shared by every report-writing tool.

Scope: every structural piece a report-writing tool needs that isn't specific to its own data --
assembling a header + N title/body sections + a footer into one report's text (render_report()),
the standard "tool/description/one-or-more-runs" header block every tool renders
(standard_header()), the two small formatting primitives used to build a footer
(command_header(), help_redirect()), and writing the finished text to a file while also returning
it (write_report_file()). Report *content* -- what a specific table's numbers mean -- stays in
each stage5 table module; this module only owns the shape every report shares.

Functions: write_report_file(), render_report(), command_header(), help_redirect(),
standard_header().
"""

import os
import sys
from datetime import datetime


def write_report_file(dest_path, parts):
    """Joins parts (a list of strings) into one report string, writes it to dest_path, and
    returns the string -- the file-writing primitive every write_report() in this codebase needs,
    whether or not it also uses render_report() below."""
    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def render_report(header, sections, footer=""):
    """Builds the parts list for one report: header (a single pre-built string, typically
    standard_header()'s output), an auto-generated "tables:" listing of every titled section (see
    below), then each (title, body) pair in `sections` rendered as a title line + body + one blank
    line, then footer (another pre-built string, e.g. a help_redirect() line, or a command_header()
    line -- the report's own invocation reads naturally as the last thing in the file). Returns a
    list ready for write_report_file().

    `title` is plain, undecorated text (a bare sentence ending in "\\n", never "===" or a number
    -- render_report() decides that itself, once, for the whole report): 0 titled sections in the
    list (e.g. a calltree tool's untitled tree/fallback blocks) means no numbering and no
    "tables:" listing at all; exactly 1 titled section still gets "=== Title ===" (which table is
    being shown is always worth stating, even when there's only one -- it just isn't numbered,
    since there's nothing to number it against); 2+ get "=== N. Title ===", numbered by position
    among titled sections only (an untitled prose block in between doesn't consume a number).
    `title=None`/falsy is a titleless prose block -- never counted, never decorated. Every section
    body is expected to end with exactly its own content's newline and no more (the same
    convention render_table()/format_aligned_rows() already follow) -- the blank line between
    sections, and any title decoration, is always this function's job, never a leaf renderer's.
    """
    titled = [(title.rstrip("\n"), body) for title, body in sections if title]
    parts = [header]
    if titled:
        parts.append("tables:\n" + "".join(f"  - {t}\n" for t, _body in titled) + "\n")
    numbered = 0
    multi = len(titled) >= 2
    for title, body in sections:
        if title:
            plain = title.rstrip("\n")
            if multi:
                numbered += 1
                parts.append(f"=== {numbered}. {plain} ===\n")
            else:
                parts.append(f"=== {plain} ===\n")
        parts.append(body)
        parts.append("\n")
    if footer:
        parts.append(footer)
    return parts


def command_header(argv0, tokens, width=100):
    """Builds the report's 'command:' line: sys.executable + this script's own absolute path,
    followed by tokens -- CLI arguments the tool itself already assembled from its parsed
    argparse Namespace (never raw sys.argv, which can't tell a user-typed relative path from a
    coincidentally-identical option value elsewhere in the command). Wraps onto continuation
    lines, indented to align under the first token, once a line would exceed width columns."""
    prefix = "command: "
    indent = " " * len(prefix)
    all_tokens = [sys.executable, os.path.abspath(argv0), *tokens]
    lines = []
    current = prefix
    for tok in all_tokens:
        on_fresh_line = current in (prefix, indent)
        piece = tok if on_fresh_line else f" {tok}"
        if not on_fresh_line and len(current) + len(piece) > width:
            lines.append(current + " \\")
            current = indent + tok
        else:
            current += piece
    lines.append(current)
    return "\n".join(lines) + "\n"


def help_redirect(topics, script_name=None):
    """One redirect line pointing a reader to --help for general/methodology content, instead of
    repeating it in every report. script_name defaults to os.path.basename(sys.argv[0]) so the
    line can never point at a stale tool name after a rename."""
    script_name = script_name or os.path.basename(sys.argv[0])
    return f"For details on {topics}, see {script_name} --help.\n"


def standard_header(tool_name, description, runs):
    """The metadata block every report shares: one line naming the tool and when this report was
    generated, this tool's own short description (1-2 lines, distinct from HELP_BLURB's full
    detail -- just enough to identify the report out of context), and one or more runs it read
    data from.

    Each entry in runs is a dict: {"directories": [(label, path), ...], "executable",
    "run_datetime", "runtime", "num_ranks", "scanned_files", "extra_lines"}. "directories" is
    required -- one entry for a tool reading a single directory (CPU/GPU hotspots, POP metrics),
    two for a tool that pairs a CPU-side and a GPU-side directory into one shared run (the
    combined hotspots tool, both calltree tools) -- either way, exactly one executable/run-date-
    time/runtime/MPI-ranks block follows, since it's the same underlying execution regardless of
    how many separate profiling tools captured it. Every other field is optional and prints blank/
    omitted when absent, the same "never error on missing metadata" convention gather_run_info()
    already follows. "scanned_files" (a list of paths, present only for a tool that has one)
    renders as an indented bullet list, paths relative to the run's first directory. "extra_lines"
    (a list of pre-formatted, already-indented strings, present only when needed) is an escape
    hatch for a fact that doesn't fit this shape at all.

    Whenever 2+ directories are shown in total -- whether from one run pairing a CPU and a GPU
    directory, or from multiple independent runs (e.g. a POP-metrics scaling study) -- one
    trailing caveat notes they aren't cross-checked against each other, regardless of which shape
    produced the count.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    parts = [f'"{tool_name}" report generated "{timestamp}".\n', description.rstrip("\n") + "\n", "\n"]
    for run in runs:
        for label, path in run["directories"]:
            parts.append(f"{label}: {os.path.abspath(path)}\n")
        parts.append(f"  executable: {run.get('executable') or ''}\n")
        parts.append(f"  run date/time: {run.get('run_datetime') or ''}\n")
        parts.append(f"  runtime: {run.get('runtime') or ''}\n")
        parts.append(f"  MPI ranks: {run['num_ranks'] if run.get('num_ranks') is not None else ''}\n")
        if "scanned_files" in run:
            first_dir = run["directories"][0][1]
            parts.append("  files scanned:\n")
            parts.extend(f"    - {os.path.relpath(f, first_dir)}\n" for f in run["scanned_files"])
        parts.extend(run.get("extra_lines") or [])
    total_dirs = sum(len(run["directories"]) for run in runs)
    if total_dirs > 1:
        parts.append(
            f"note: the {total_dirs} directories above are not checked against each other "
            "(same executable/test case/run) -- that's the caller's responsibility.\n"
        )
    parts.append("\n")
    return "".join(parts)
