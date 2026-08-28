"""tools/-specific test plumbing for postprocess/tests/ -- not a test file itself.

Owns assert_extract_tool_cli_contract() (the baseline CLI contract
extract_calltree.py/extract_wallclock_calltree.py's tests both independently verified in full
before this existed), clear_agg_cache() (the get_rank_aggregate() on-disk-cache cleanup the
three test_extract_trace_*.py files each need in their own tearDown() -- a stale rank1.agg.json/etc.
left over from a prior test run would otherwise mask what the test under test actually produced),
and assert_help_leads_with_explanation() (CLAUDE.md's -h/--help convention every tool must follow).

Exposes: assert_extract_tool_cli_contract(), clear_agg_cache(), assert_help_leads_with_explanation().
"""

import contextlib
import glob
import io
import os
import tempfile

import _test_helpers  # noqa: F401  (side effect only: bootstraps sys.path + _stage_paths)
from _stage6_test_helpers import write_noise_config


def assert_extract_tool_cli_contract(test_case, tool_module, run_dir, empty_dir, noise_label):
    """Runs the baseline CLI contract every extract_*_calltree.py-shaped tool shares (main(argv)
    takes a run directory, an optional second explicit directory, -o/--output, --max-depth, and
    --extra-noise-config): missing/empty input both raise SystemExit, main() writes a file
    end-to-end, the explicit-two-directories code path works, and --extra-noise-config excludes a
    configured label. noise_label must be a real label run_dir's own default report includes, so
    excluding it is an observable effect. Call from a test method that has already registered its
    own stage6_noise_config reset (self.addCleanup(stage6_noise_config.configure, None)) --
    this helper only writes the config, it doesn't own resetting the process-wide singleton."""
    with test_case.subTest(check="missing_directory_raises_clear_error"):
        with test_case.assertRaises(SystemExit):
            tool_module.main(["/no/such/directory"])

    with test_case.subTest(check="empty_input_raises_clear_error"):
        with test_case.assertRaises(SystemExit):
            tool_module.main([empty_dir])

    with test_case.subTest(check="end_to_end_writes_file"):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            tool_module.main([run_dir, "-o", dest, "--max-depth", "1"])
            test_case.assertTrue(os.path.isfile(dest))

    with test_case.subTest(check="explicit_two_directories"):
        # run_dir passed twice: exercises the two-directory code path end to end without
        # needing a second, differently-shaped fixture.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            tool_module.main([run_dir, run_dir, "-o", dest])
            with open(dest) as f:
                report = f.read()
            test_case.assertIn("CPU run directory:", report)
            test_case.assertIn("GPU run directory:", report)

    with test_case.subTest(check="extra_noise_config_flag_excludes_a_configured_row"):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            config_path = write_noise_config(tmp, {"add": {"other": [noise_label]}})
            tool_module.main([run_dir, "-o", dest, "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
            test_case.assertNotIn(noise_label, report)


def clear_agg_cache(directory):
    """Deletes every get_rank_aggregate()-written *.agg.json under directory -- call from
    tearDown() in any test that runs a trace-CSV tool against a real fixture directory, so a
    later test (or a later run) never reads a cache file an earlier test left behind."""
    for path in glob.glob(os.path.join(directory, "*.agg.json")):
        os.remove(path)


def assert_help_leads_with_explanation(test_case, tool_module, docs_url_substring):
    """Confirms main(["--help"]) exits 0 and prints tool_module.HELP_BLURB's own explanation text
    before the "positional arguments:"/"options:" sections argparse appends -- CLAUDE.md's
    "-h/--help must lead with a short, jargon-free explanation... before the existing
    options/flags table" convention every postprocess/tools/*.py CLI must follow. docs_url_substring
    should be a fragment of the specific documentation URL this tool's own HELP_BLURB ends with
    (not always a ROCm URL -- e.g. convert_trace_to_csv.py's real "under the hood" dependency is
    Perfetto's trace_processor_shell, not an AMD tool, so its closing line and docs link differ
    accordingly), confirming the closing "under the hood" line survived, not just that the blurb
    is nonempty."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with test_case.assertRaises(SystemExit) as ctx:
            tool_module.main(["--help"])
    test_case.assertEqual(ctx.exception.code, 0)

    output = buf.getvalue()
    first_blurb_line = tool_module.HELP_BLURB.strip().splitlines()[0]
    test_case.assertIn(first_blurb_line, output)
    test_case.assertIn(docs_url_substring, output)
    test_case.assertLess(output.index(first_blurb_line), output.index("options:"))
