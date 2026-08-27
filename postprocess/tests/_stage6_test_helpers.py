"""stage6-specific test plumbing for postprocess/tests/ -- not a test file itself.

Owns write_noise_config(), the --extra-noise-config diff-file writer every test exercising
stage6_noise_config.configure() needs -- directly, in stage6's own tests, or indirectly through a
tool's --extra-noise-config CLI flag, in tools/'s tests. See stage6_noise_config.configure()'s own
docstring for the {"add"/"remove"/"disable"} diff schema this writes.

Exposes: write_noise_config().
"""

import json
import os

import _test_helpers  # noqa: F401  (side effect only: bootstraps sys.path + _stage_paths)


def write_noise_config(tmp_dir, diff):
    """Writes a --extra-noise-config diff dict to noise_config.json under tmp_dir, returning its
    path -- the setup step every --extra-noise-config test needs, regardless of which tool or
    which specific diff it's proving reaches the tool's own tagging/aggregation."""
    path = os.path.join(tmp_dir, "noise_config.json")
    with open(path, "w") as f:
        json.dump(diff, f)
    return path
