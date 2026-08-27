"""Shared plumbing for postprocess/tests/ -- not a test file itself, nothing here runs as a test.

Owns the by-path module-loading mechanism every test file that can't do a plain `from stageN_x
import y` needs: tool tests (tools are meant to be run as scripts, not imported) and the handful of
stage tests that specifically want a fresh, isolated module instance -- see postprocess/README.md's
"Tests" section for why both styles exist. Bootstraps postprocess/_stage_paths.py itself on import,
so a caller only needs this module on sys.path, not also _stage_paths separately.

Exposes one function: load_module_by_path().
"""

import importlib.util
import os
import sys

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))
import _stage_paths  # noqa: E402  (adds every stageN/tools dir to sys.path)


def load_module_by_path(name, *path_parts, isolated=False):
    """Load postprocess/<path_parts joined> as a module named `name`, registered in sys.modules.

    isolated=True additionally restores whatever sys.modules[name] held before this call once
    loading finishes, instead of leaving the freshly-loaded copy registered globally. Needed by
    stage6_noise_config's own tests: stage3_rocprofsys_common.py captures a direct reference to its
    tag_defs function at import time, so permanently overwriting sys.modules["stage6_noise_config"]
    here would silently detach that reference from whichever instance this test's own configure()
    calls actually mutate, breaking --extra-noise-config everywhere else for the rest of the
    process. The returned module object is unaffected either way -- only the global registry entry.
    """
    file_path = os.path.join(POSTPROCESS_DIR, *path_parts)
    previous = sys.modules.get(name) if isolated else None
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if isolated:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module
