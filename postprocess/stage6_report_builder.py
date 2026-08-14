"""Stage 6 report assembly, shared by every report-writing tool.

Scope: the two structural pieces every report-writing tool needs -- assembling a header + N
title/body sections + a footer into one report's text (render_report()), and writing that text to
a file while also returning it (write_report_file()). Has no opinion on report content, wording,
or formatting rules (see docs/plans/2.1-postprocess-consolidation-refactor.md §10 for those --
roadmap step 9, not this one).

Functions: write_report_file(), render_report().
"""


def write_report_file(dest_path, parts):
    """Joins parts (a list of strings) into one report string, writes it to dest_path, and
    returns the string -- the file-writing primitive every write_report() in this codebase needs,
    whether or not it also uses render_report() below."""
    report = "".join(parts)
    with open(dest_path, "w") as f:
        f.write(report)
    return report


def render_report(header, sections, footer=""):
    """Builds the parts list for one report: header (a single pre-built string -- metadata/run-info
    lines, entirely up to the caller), then each (title, body) pair in `sections` rendered as
    title line (skipped if title is falsy -- a titleless prose block) + body + one blank line,
    then footer (another pre-built string, e.g. closing notes/redirects). Returns a list ready for
    write_report_file().

    Every tool's write_report() has the same real structure underneath its own wording: compute
    data, then assemble a header + a sequence of title+body sections + a footer. This function
    generalizes the "assemble a sequence of sections" part; each tool keeps full control of what
    its own header/sections/footer text actually says -- this step changes HOW that text gets
    assembled, never WHAT it says. Every section body is expected to end with exactly its own
    content's newline and no more (the same convention render_table()/format_aligned_rows() already
    follow) -- the blank line between sections is always this function's job, never a leaf
    renderer's.
    """
    parts = [header]
    for title, body in sections:
        if title:
            parts.append(title)
        parts.append(body)
        parts.append("\n")
    if footer:
        parts.append(footer)
    return parts
