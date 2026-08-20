"""Stage 3 (category-based tagger) for rocprof-sys's Perfetto trace-CSV rows.

Scope: turning ancestry-linked trace rows (stage1_rocprofsys_trace.py's output) into a per-row
set of TAGS -- and nothing else. Same philosophy as stage3_rocprofsys_common.py: a tag is a fact
about a row, never a decision about what a tool does with that fact (drop it, splice it out,
collapse its children, route it to a different table) -- that mapping stays the calling tool's
job, not this module's.

Two independent sources feed row["tags"] here:
  - `gpu_api`, `gpu_kernel`, `gpu_memcpy`, `mpi_territory`, and the `other` fallback come from
    `tag_for_category()`, a direct `{category: tag}` lookup built from AMD's documented
    categories.h enum -- a trace row already carries an exact, ground-truth `category` per event,
    so no pattern-matching is needed for these. This map is a plain Python dict, not a JSON file
    like stage6's noise patterns: a `category` value is a closed, ROCm-defined enum (not
    open-ended app text), so there's nothing here for a user to tune -- the "other" fallback is
    what actually gives forward-compatibility with newer ROCm releases, not a config file. The
    numa/rocm(generic)/amd_smi_* -> "other" routing is a deliberate, documented placeholder: these
    are real, observed categories, but their row-level semantics aren't validated against real
    trace data yet -- not a final taxonomy decision.
  - `wrapper_noise`, `compiler_runtime_noise`, and the derived `wrapper_branch_noise` are produced
    by delegating straight to `stage3_rocprofsys_common.tag_rows()`, restricted to just those
    three tag defs, passing `label_key=stage1_rocprofsys_trace.LABEL_KEY` ("name" for this
    format). Those three patterns classify by *who's calling* (rocprofsys/gotcha instrumentation,
    libc allocator internals), not by category, so a trace's CPU-side rows (host, ompt, pthread,
    sampling, ...) are exactly as exposed to this risk as the sample pipeline's rows are --
    reusing the same substring lists in default_noise_patterns.json avoids duplicating that domain
    knowledge a second time. None of these three patterns use "ancestor_for_thread_roots" or
    "first_real_descendant_skip_tag" today, so no is_thread_root handling is needed here.
    `gpu_api`/`mpi_territory` are deliberately excluded from this delegated call, since they're
    already exact from the category map.

remove_tagged_subtrees(), splice_by_tag(), make_collapses_children(), make_is_pruned() are
re-exported unchanged from stage3_rocprofsys_common -- see there for semantics. None of them read
a row's display name or category, so they work on tagged trace rows with zero modification.

Explicitly out of scope: any tool-level {tag: action} wiring, corr_id-based kernel/launch-site
correlation, cross-rank merging, count/self_sum computation (all later plans).

Functions: tag_for_category(), tag_rows(), plus the four re-exported primitives above.
"""

from stage1_rocprofsys_trace import LABEL_KEY
from stage3_rocprofsys_common import (
    make_collapses_children,
    make_is_pruned,
    remove_tagged_subtrees,
    splice_by_tag,
    tag_rows as _tag_rows_by_pattern,
)
from stage6_noise_config import tag_defs as _stage6_tag_defs

_CATEGORY_TAG_MAP = {
    # Host-side API-call overhead.
    "rocm_hip_api": "gpu_api",
    "rocm_hsa_api": "gpu_api",
    "rocm_marker_api": "gpu_api",
    "rocm_rocdecode_api": "gpu_api",
    "rocm_rocjpeg_api": "gpu_api",
    "rocm_rccl_api": "gpu_api",
    "rocm_counter_collection": "gpu_api",
    # Actual device execution time.
    "rocm_kernel_dispatch": "gpu_kernel",
    "rocm_rccl": "gpu_kernel",
    # Device memory movement.
    "rocm_memory_copy": "gpu_memcpy",
    "rocm_scratch_memory": "gpu_memcpy",
    "rocm_page_migration": "gpu_memcpy",
    # MPI.
    "mpi": "mpi_territory",
    # Observed, but not yet validated against real data -- explicit fallback rather than a guess.
    "numa": "other",
    "rocm": "other",
}

# Plausible homes for real app/OpenMP/Kokkos code -- left untagged by the category map itself,
# same as the sample engine leaves ordinary app-code rows untagged. Still eligible for the
# wrapper_noise/compiler_runtime_noise name-based pass in tag_rows() below.
_CPU_PASSTHROUGH_CATEGORIES = {
    "host", "ompt", "pthread", "sampling", "python", "user", "kokkos", "none",
}

_CPU_NOISE_TAG_NAMES = ("wrapper_noise", "compiler_runtime_noise", "wrapper_branch_noise")


def tag_for_category(category):
    """Maps a trace row's `category` to a tag name, or None if it's a CPU-ish category with no
    category-level tag of its own (still eligible for the name-based pass in tag_rows()). Falls
    back to "other" for the amd_smi_* family and for any category not recognized at all --
    the roadmap's required "explicit fallback bucket for unmapped categories" (e.g. a category
    from a newer ROCm release than this map knows about)."""
    cat = (category or "").lower()
    if cat in _CATEGORY_TAG_MAP:
        return _CATEGORY_TAG_MAP[cat]
    if cat.startswith("amd_smi"):
        return "other"
    if cat in _CPU_PASSTHROUGH_CATEGORIES:
        return None
    return "other"


def tag_rows(rows):
    """Mutates every row in place, same as stage3_rocprofsys_common.tag_rows(): sets
    row["tags"]/row["self_tags"]/row["structural_drop_tags"] from a restricted, delegated
    wrapper_noise/compiler_runtime_noise/wrapper_branch_noise pass (see module docstring), then
    unions each row's tag_for_category() result into row["tags"] -- must happen after the
    delegated call, since it assigns (not unions) row["tags"]. Returns None."""
    restricted_defs = {
        name: td for name, td in _stage6_tag_defs().items() if name in _CPU_NOISE_TAG_NAMES
    }
    _tag_rows_by_pattern(rows, tag_defs=restricted_defs, label_key=LABEL_KEY)

    for row in rows:
        tag = tag_for_category(row.get("category"))
        if tag is not None:
            row["tags"].add(tag)
