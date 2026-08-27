"""stage5-specific test plumbing for postprocess/tests/ -- not a test file itself.

Shared between test_stage5_calltree_view.py and test_stage5_wallclock_calltree_view.py: both
render a tree through the exact same shape (resolve run_dir -> build a view -> concatenate its
tree_text + fallback_text), just via a different tool-specific build_calltree_view() each, and both
need the same trailing-numeric-column-stripping trick to search a hard-wrapped label.

Exposes: render_calltree_view(), labels_only().
"""

import re

import _test_helpers  # noqa: F401  (side effect only: bootstraps sys.path + _stage_paths)
from stage1_run_dirs import resolve_run_dirs  # noqa: E402  (needs _test_helpers' bootstrap first)


def render_calltree_view(build_calltree_view, run_dir, **kwargs):
    """Resolves run_dir into (cpu_dir, gpu_dir), builds the view via the given tool-specific
    build_calltree_view(run_dir, cpu_dir, gpu_dir, **kwargs), and returns its rendered text
    (tree + fallback) concatenated -- the common shape every calltree-view test file's own thin
    render(run_dir, **kwargs) wrapper delegates to."""
    cpu_dir, gpu_dir = resolve_run_dirs(run_dir)
    view = build_calltree_view(run_dir, cpu_dir, gpu_dir, **kwargs)
    return view["tree_text"] + view["fallback_text"]


_NUMERIC_SUFFIX_RE = re.compile(r"(?:\s{2,}-?[\d.]+)+\s*$")


def labels_only(report):
    """Strips each physical line's trailing numeric-cell columns (if any), then joins every line
    back together with no separator -- reconstructs each row's own label text contiguously, so a
    test can search for a label substring regardless of exactly where
    stage5_tree_render.wrap_leading_labels() cut a too-long label across physical lines."""
    return "".join(_NUMERIC_SUFFIX_RE.sub("", line) for line in report.splitlines())
