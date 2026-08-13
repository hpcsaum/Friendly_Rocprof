"""Run-directory resolution -- the one piece of stage 1 (reading a profile off disk)
that both rocprof-sys and rocprofv3 tools need before either format-specific parser
(stage1_rocprofsys.py, stage1_rocprofv3.py) can even start.

Scope: turning a single "run" directory into the (cpu_dir, gpu_dir) pair the rest of
the pipeline reads from. Knows nothing about either tool's file formats -- just the
rocprof-sys/ and rocprofv3/ subdirectory convention profile_hotspots.sh and friends
produce, plus the un-nested fallback profile_CPU_hotspots.sh uses on its own. Used by
extract_calltree.py, extract_calltree_traced.py, and extract_pop_metrics.py.

Functions: resolve_run_dirs().
"""

import os


def resolve_run_dirs(run_dir):
    """A "run" is one directory that may contain a rocprof-sys/ subdir (CPU timing) and/or a
    rocprofv3/ subdir (GPU kernel timing). Falls back to treating run_dir itself as the CPU dir
    when there's no rocprof-sys/ subdir (profile_CPU_hotspots.sh's un-nested layout). Returns
    (cpu_dir, gpu_dir_or_None) without checking either actually contains data -- that's each
    caller's own job."""
    cpu_subdir = os.path.join(run_dir, "rocprof-sys")
    gpu_subdir = os.path.join(run_dir, "rocprofv3")
    cpu_dir = cpu_subdir if os.path.isdir(cpu_subdir) else run_dir
    gpu_dir = gpu_subdir if os.path.isdir(gpu_subdir) else None
    return cpu_dir, gpu_dir
