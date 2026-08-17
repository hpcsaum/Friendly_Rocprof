"""Shared argparse/validation boilerplate every CLI tool's main() repeats.

Scope: the argument-handling and setup steps that are the same across most of the 8 CLI tools --
validating one or more required directories, resolving a default output destination, and the
-n/--top/--threshold/--all ranked-selection group, --max-depth, and --show-* noise-visibility
flags several tools share verbatim. Has no opinion on report content, table columns, or noise
classification -- those stay owned by stage5/stage6_noise_config respectively; this module only
owns the argparse wiring and the directory/destination checks every tool's own main() needs before
it gets to its real work. --extra-noise-config's own CLI wiring lives in stage6_noise_config.py
instead, co-located with the module that owns that behavior.

Functions: require_directory(), require_directories(), resolve_dest(), add_selection_args(),
add_max_depth_arg(), add_noise_tier_args().
"""

import os

NOISE_TIER_FLAGS = {
    "gpu_api": ("--show-gpu-api", "also show GPU-API/offload-runtime noise instead of hiding it"),
    "rocprofsys_internals": ("--show-rocprofsys-internals",
                              "also show rocprof-sys's own instrumentation/GOTCHA/dynamic-linker "
                              "frames instead of splicing them out"),
    "mpi_internals": ("--show-mpi-internals",
                       "also show MPI library internals below the first MPI frame, instead of "
                       "collapsing them"),
    "compiler_runtime": ("--show-compiler-runtime",
                          "also show compiler-runtime allocator/intrinsic helper noise instead of "
                          "hiding it"),
}


def require_directory(path):
    """Raises SystemExit with this project's standard error message if path isn't a directory --
    the base check every tool's main() does on each of its own directory arguments."""
    if not os.path.isdir(path):
        raise SystemExit(f"error: no such directory: {path!r}")


def require_directories(paths):
    """require_directory() applied to every path in paths -- the shape every tool validating 2+
    required directories at once needs (a pop_metrics-style scaling study's reference + scaled
    dirs, a resolved CPU+GPU directory pair). None entries are skipped: an optional directory that
    wasn't given/resolved at all isn't an existence error by itself -- whether that's an error (a
    tool that requires its GPU side to resolve to something) or fine (a tool whose GPU side is
    genuinely optional) is tool-specific business logic that stays in the caller, checked before or
    after this call as appropriate."""
    for path in paths:
        if path is not None:
            require_directory(path)


def resolve_dest(explicit_dest, default_dir, default_filename):
    """The -o/--output resolution every tool does identically: explicit_dest if given, else
    default_dir/default_filename."""
    return explicit_dest or os.path.join(default_dir, default_filename)


def add_selection_args(parser, plural_noun, threshold_unit_help, singular_noun=None,
                        top_noun=None, top_help_suffix="", verb="list"):
    """Adds the -n/--top / --threshold / --all mutually exclusive group every ranked-table tool
    shares. plural_noun is the --threshold line's own noun (e.g. "entries", "kernels",
    "functions"); singular_noun defaults to plural_noun with a trailing "s" stripped (true for
    every noun this codebase's tools use today) -- pass it explicitly for an irregular plural
    (plural_noun="entries" needs singular_noun="entry"). top_noun (the -n/--top line's own noun,
    often a more specific/qualified phrase like "hotspot functions" or just "hotspots") defaults to
    plural_noun. verb defaults to "list" (the report tools' convention); the select_hotspot_*.py
    tools pass verb="select" instead, matching their own "resolve/select a set of labels" framing.
    Returns the group, in case a caller needs to add one more mutually exclusive option to it."""
    top_noun = top_noun or plural_noun
    singular_noun = singular_noun or (plural_noun[:-1] if plural_noun.endswith("s") else plural_noun)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-n", "--top", dest="top", type=int, default=None,
                            help=f"number of {top_noun} to {verb}{top_help_suffix}")
    selection.add_argument("--threshold", dest="threshold", type=float, default=None,
                            help=f"only {verb} {plural_noun} at or above this %% {threshold_unit_help}")
    selection.add_argument("--all", dest="show_all", action="store_true",
                            help=f"{verb} every {singular_noun}, no truncation")
    return selection


def add_max_depth_arg(parser):
    """Adds --max-depth, the tree-truncation flag every tree-rendering tool shares."""
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=None,
                         help="truncate the tree at this depth (default: unlimited, print the whole tree)")


def add_noise_tier_args(parser, tiers, all_shorthand=False):
    """Adds one --show-<tier> flag per name in tiers (any subset/order of NOISE_TIER_FLAGS' 4
    keys), each setting dest="show_<tier>". If all_shorthand and len(tiers) > 1, also adds
    --show-all-internals as a shorthand for all of them at once -- the caller's own code still
    decides how to combine it with the individual flags."""
    for tier in tiers:
        flag, help_text = NOISE_TIER_FLAGS[tier]
        parser.add_argument(flag, dest=f"show_{tier}", action="store_true", help=help_text)
    if all_shorthand and len(tiers) > 1:
        parser.add_argument("--show-all-internals", action="store_true",
                             help=f"shorthand for all {len(tiers)} --show-* flags above at once")
