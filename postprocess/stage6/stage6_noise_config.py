"""Owns default_noise_patterns.json's bundled loading and resolves the noise-pattern definitions
every stage-3-consuming tool shares for the lifetime of one process, from an optional user diff
file layered on top of it.

Scope: the one place default_noise_patterns.json (or a user's own diff file) is read.
stage3_rocprofsys_common.py's tag_rows() reads the result via tag_defs() -- a plain one-directional
dependency (stage3 -> stage6), never the reverse, and no other module in this package reads either
file directly. A process-wide global, not threaded as a parameter through every function between a
tool's main() and tag_rows() -- every real invocation of this codebase is a single, one-shot CLI
process with exactly one noise-pattern configuration for its whole run, so a module-level singleton
is simpler and just as correct as parameter-threading.

Functions: load_default_patterns(), configure(), tag_defs(), add_cli_argument(),
configure_from_args().
"""

import json
import os

DEFAULT_PATTERNS_PATH = os.path.join(os.path.dirname(__file__), "default_noise_patterns.json")

_TAG_DEFS = None


def load_default_patterns(path=None):
    """The bundled tag -> pattern-definition mapping (see stage3_rocprofsys_common.py's module docstring
    for the schema)."""
    with open(path or DEFAULT_PATTERNS_PATH) as f:
        return json.load(f)


def configure(extra_config_path=None):
    """Resolves and stores the final tag_defs for this process -- called once, early, by each CLI
    tool's main() right after parsing --extra-noise-config/$FRIENDLY_ROCPROF_NOISE_CONFIG. Safe to
    call again (e.g. between tests, to reset or change the active config) -- always fully replaces
    any prior value, never merges with it.

    extra_config_path's diff schema: {"add": {tag: [substring, ...]}, "remove": {tag: [substring,
    ...]}, "disable": [tag, ...]}. add/remove only ever touch a tag's own "substrings" list (not
    prefixes/suffixes/ancestor rules -- those govern more structural matching behavior than a
    simple "does this text appear" tweak needs). disable is resolved first and always wins: a
    disabled tag is dropped entirely, and any add/remove naming it is then a no-op, not an error.
    Raises SystemExit for an unknown tag name anywhere, or for add/remove targeting a purely
    derived tag (currently just wrapper_branch_noise -- it has no self-scope patterns to tweak,
    only disable is meaningful for it).
    """
    global _TAG_DEFS
    patterns = load_default_patterns()
    patterns.setdefault("other", {"substrings": []})
    if extra_config_path is not None:
        _apply_diff(patterns, extra_config_path)
    _TAG_DEFS = patterns


def tag_defs():
    """The current resolved tag_defs -- lazily configured with no override on first use, so any
    caller that never touches configure() (every existing test, any tool run without
    --extra-noise-config) sees exactly today's bundled defaults, unchanged."""
    if _TAG_DEFS is None:
        configure(None)
    return _TAG_DEFS


def add_cli_argument(parser):
    """Adds --extra-noise-config to parser, with the standard help text every noise-tagging tool
    shares -- pair with configure_from_args() once the tool's own main() has parsed args."""
    parser.add_argument("--extra-noise-config", dest="extra_noise_config", default=None,
                         help="path to a JSON file customizing noise-tag patterns (add/remove "
                              "substrings, disable a tag) -- see stage6_noise_config.py's "
                              "configure() for the file schema; falls back to "
                              "$FRIENDLY_ROCPROF_NOISE_CONFIG if not given")


def configure_from_args(args):
    """Resolves --extra-noise-config (falling back to $FRIENDLY_ROCPROF_NOISE_CONFIG) from a
    parsed argparse Namespace and calls configure() with it -- the one line each tool's main()
    needs, right after parser.parse_args(), before doing any real work."""
    configure(args.extra_noise_config or os.environ.get("FRIENDLY_ROCPROF_NOISE_CONFIG"))


def _apply_diff(patterns, extra_config_path):
    with open(extra_config_path) as f:
        diff = json.load(f)
    disable = set(diff.get("disable", []))
    for tag in disable:
        if tag not in patterns:
            raise SystemExit(f"error: {extra_config_path!r}: unknown noise tag {tag!r} in \"disable\"")
    for tag in disable:
        patterns.pop(tag, None)

    patterned = {name for name, td in patterns.items() if "sibling_group_source_tag" not in td}
    for op in ("remove", "add"):
        for tag, substrings in diff.get(op, {}).items():
            if tag in disable:
                continue
            if tag not in patterned:
                raise SystemExit(
                    f"error: {extra_config_path!r}: {op} targets {tag!r}, which isn't a known "
                    "pattern tag (derived tags like wrapper_branch_noise have no patterns to "
                    f"{op}; only \"disable\" applies to them)"
                )
            # Matching is against the row's own lowercased label (see stage3_rocprofsys_common.py's
            # _label_matches()), which only works case-insensitively if the pattern side is
            # already lowercase too -- true of every bundled pattern today by convention, but not
            # guaranteed for a user's own input, so it's normalized here rather than silently
            # failing to match a substring typed in any other case.
            substrings = [s.lower() for s in substrings]
            existing = patterns.setdefault(tag, {}).setdefault("substrings", [])
            if op == "remove":
                patterns[tag]["substrings"] = [s for s in existing if s not in substrings]
            else:
                existing.extend(substrings)
