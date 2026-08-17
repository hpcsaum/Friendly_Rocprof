"""Stage 6 run-metadata guessing, shared by extract_CPU_hotspots.py and extract_GPU_hotspots.py.

Scope: best-effort, never-raises guesses about a run (executable name, run date/time, total
runtime, rank count) from whatever undocumented JSON metadata file a tool finds, plus the shared
file-discovery/key-search primitives underneath. Neither rocprof-sys's metadata.json nor
rocprofv3's *_config.json schema is documented anywhere, so every guess here is exactly that -- a
guess tried against a short list of plausible key names, never a parse of a known format. A tool
whose own file/key layout has more structure to it than these primitives cover (e.g. a
directory-name fallback for run date/time) passes that extra structure in as a parameter rather
than this module trying to special-case it.

Functions: load_json_file(), find_first_key(), guess_executable(), guess_total_runtime(),
guess_run_datetime(), guess_num_ranks().
"""

import glob
import json
import os


def load_json_file(output_dir, glob_pattern):
    """Best-effort find-and-parse the first file under output_dir matching glob_pattern
    (recursive) as JSON -- returns {} on missing file, read error, or non-dict content, never
    raises."""
    candidates = sorted(glob.glob(os.path.join(output_dir, "**", glob_pattern), recursive=True))
    if not candidates:
        return {}
    try:
        with open(candidates[0]) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def find_first_key(d, candidate_keys, _depth=0):
    """Best-effort case-insensitive key search, one level of nested dicts deep -- the JSON
    schema isn't documented, so this is a guess, not a parse."""
    if not isinstance(d, dict):
        return None
    lower_map = {k.lower(): v for k, v in d.items()}
    for key in candidate_keys:
        if key in lower_map and lower_map[key] not in (None, "", []):
            return lower_map[key]
    if _depth == 0:
        for v in d.values():
            if isinstance(v, dict):
                found = find_first_key(v, candidate_keys, _depth=1)
                if found is not None:
                    return found
    return None


def guess_executable(data, keys):
    val = find_first_key(data, keys)
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, str) and val.strip():
        return os.path.basename(val.split()[0])
    return None


def guess_total_runtime(data, keys):
    val = find_first_key(data, keys)
    if isinstance(val, (int, float)):
        return f"{val:.6f} sec"
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def guess_run_datetime(data, keys, output_dir=None, scanned_files=(), dir_pattern=None):
    """First non-empty string found under any of keys. If that fails and dir_pattern is given,
    falls back to matching dir_pattern against output_dir, then against each scanned file's own
    directory name -- omit dir_pattern for a tool with no such directory-name convention to fall
    back to."""
    val = find_first_key(data, keys)
    if isinstance(val, str) and val.strip():
        return val.strip()
    if dir_pattern is None:
        return None
    m = dir_pattern.search(output_dir)
    if m:
        return m.group(0)
    for path in scanned_files:
        m = dir_pattern.search(os.path.dirname(path))
        if m:
            return m.group(0)
    return None


def guess_num_ranks(data, pid_suffix_re, scanned_files, keys=None):
    """If keys is given, tries a numeric value under any of them first. Otherwise (or if that
    fails), counts distinct PIDs found by pid_suffix_re across scanned_files' basenames."""
    if keys is not None:
        val = find_first_key(data, keys)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    pids = set()
    for path in scanned_files:
        m = pid_suffix_re.search(os.path.basename(path))
        if m:
            pids.add(m.group(1))
    return len(pids) if pids else None
