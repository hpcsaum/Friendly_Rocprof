"""Adds every postprocess/stageN/ and postprocess/tools/ directory to sys.path, once, so every
source and test file's existing flat "from stage1_rocprofsys_sample import X"-style imports keep
resolving no matter which subdirectory the importing file itself lives in. Import this only after
putting postprocess/'s own absolute path on sys.path (see any tool or test file for the one-line
pattern) -- this module finds its sibling directories via its own __file__, so it works
regardless of who imports it or from where.
"""

import glob
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _dir in sorted(glob.glob(os.path.join(_ROOT, "stage*"))) + [os.path.join(_ROOT, "tools")]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
