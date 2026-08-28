"""Tests for stage6_report_builder.py's shared report-assembly primitives.

WriteReportFileTests -- write_report_file() joins parts, writes to disk, and returns the same string
RenderReportTests    -- render_report()'s header/tables-listing/numbered-section/footer assembly rules
CommandHeaderTests   -- command_header()'s single-line vs wrapped-continuation rendering
HelpRedirectTests    -- help_redirect()'s default script-name resolution
StandardHeaderTests  -- standard_header()'s per-run metadata block, blank fields, and multi-directory caveat
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

rb = load_module_by_path("stage6_report_builder", "stage6", "stage6_report_builder.py")


class WriteReportFileTests(unittest.TestCase):
    def test_joins_parts_writes_and_returns_same_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = rb.write_report_file(dest, ["a\n", "b\n", "c\n"])
            self.assertEqual(report, "a\nb\nc\n")
            with open(dest) as f:
                self.assertEqual(f.read(), report)


class RenderReportTests(unittest.TestCase):
    def test_two_titled_sections_get_numbered_and_listed(self):
        parts = rb.render_report(
            "HEADER\n",
            [("Title 1\n", "Body 1\n"), ("Title 2\n", "Body 2\n")],
            "FOOTER\n",
        )
        self.assertEqual(
            "".join(parts),
            "HEADER\ntables:\n  - Title 1\n  - Title 2\n\n"
            "=== 1. Title 1 ===\nBody 1\n\n=== 2. Title 2 ===\nBody 2\n\nFOOTER\n",
        )

    def test_exactly_one_titled_section_gets_bracketed_but_unnumbered(self):
        parts = rb.render_report("HEADER\n", [("Only Title\n", "Body\n")])
        self.assertEqual(
            "".join(parts), "HEADER\ntables:\n  - Only Title\n\n=== Only Title ===\nBody\n\n",
        )

    def test_untitled_section_between_two_titled_ones_does_not_consume_a_number(self):
        parts = rb.render_report(
            "HEADER\n",
            [("Title 1\n", "Body 1\n"), (None, "Untitled prose\n"), ("Title 2\n", "Body 2\n")],
        )
        joined = "".join(parts)
        self.assertIn("=== 1. Title 1 ===\n", joined)
        self.assertIn("=== 2. Title 2 ===\n", joined)
        self.assertNotIn("=== 3.", joined)
        # the untitled section's own text still appears, unnumbered and undecorated
        self.assertIn("Untitled prose\n", joined)
        # and it's not counted in the tables: listing either
        self.assertEqual(joined.count("  - "), 2)

    def test_falsy_title_emits_no_title_line_and_no_tables_listing(self):
        parts = rb.render_report("HEADER\n", [(None, "Body\n")])
        rendered = "".join(parts)
        self.assertEqual(rendered, "HEADER\nBody\n\n")
        self.assertNotIn("tables:", rendered)

    def test_empty_footer_emits_nothing_extra(self):
        parts = rb.render_report("HEADER\n", [(None, "Body\n")], "")
        self.assertEqual("".join(parts), "HEADER\nBody\n\n")

    def test_zero_sections_still_emits_header_and_footer(self):
        parts = rb.render_report("HEADER\n", [], "FOOTER\n")
        self.assertEqual("".join(parts), "HEADER\nFOOTER\n")


class CommandHeaderTests(unittest.TestCase):
    def test_short_command_stays_on_one_line(self):
        line = rb.command_header("tool.py", ["/some/dir"], width=1000)
        self.assertEqual(line.count("\n"), 1)
        self.assertTrue(line.startswith("command: "))
        self.assertIn(os.path.abspath("tool.py"), line)
        self.assertTrue(line.rstrip("\n").endswith("/some/dir"))

    def test_long_command_wraps_with_aligned_continuation(self):
        long_dir = "/a/very/long/path/" + "x" * 80
        line = rb.command_header("tool.py", [long_dir, "--top", "10"], width=60)
        lines = line.rstrip("\n").split("\n")
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].endswith(" \\"))
        self.assertTrue(lines[1].startswith(" " * len("command: ")))
        self.assertIn("--top 10", lines[-1])


class HelpRedirectTests(unittest.TestCase):
    def test_default_script_name_from_argv(self):
        line = rb.help_redirect("some topics", script_name="extract_calltree.py")
        self.assertEqual(line, "For details on some topics, see extract_calltree.py --help.\n")


class StandardHeaderTests(unittest.TestCase):
    def test_first_line_names_tool_and_timestamp(self):
        header = rb.standard_header("extract_CPU_hotspots.py", "Short description.\n", [
            {"directories": [("source directory", "/x")]},
        ])
        first_line = header.splitlines()[0]
        self.assertTrue(first_line.startswith('"extract_CPU_hotspots.py" report generated "'))
        self.assertTrue(first_line.endswith('".'))
        self.assertIn("Short description.\n", header)

    def test_single_run_all_fields_present(self):
        header = rb.standard_header("extract_CPU_hotspots.py", "Desc.\n", [
            {"directories": [("source directory", "/x")], "executable": "app",
             "run_datetime": "2026-01-01", "runtime": "1.0 sec", "num_ranks": 2,
             "scanned_files": ["/x/wall_clock-1.txt", "/x/wall_clock-2.txt"]},
        ])
        self.assertIn(f"source directory: {os.path.abspath('/x')}\n", header)
        self.assertIn("  executable: app\n", header)
        self.assertIn("  run date/time: 2026-01-01\n", header)
        self.assertIn("  runtime: 1.0 sec\n", header)
        self.assertIn("  MPI ranks: 2\n", header)
        self.assertIn("  files scanned:\n    - wall_clock-1.txt\n    - wall_clock-2.txt\n", header)
        self.assertNotIn("note:", header)

    def test_single_run_all_fields_blank(self):
        header = rb.standard_header("extract_GPU_hotspots.py", "Desc.\n", [
            {"directories": [("source directory", "/x")]},
        ])
        self.assertIn("  executable: \n", header)
        self.assertIn("  run date/time: \n", header)
        self.assertIn("  runtime: \n", header)
        self.assertIn("  MPI ranks: \n", header)
        self.assertNotIn("files scanned:", header)

    def test_two_directories_in_one_run_get_one_shared_metadata_block_and_caveat(self):
        header = rb.standard_header("extract_hotspots.py", "Desc.\n", [
            {"directories": [("CPU run directory", "/cpu"), ("GPU run directory", "/gpu")],
             "executable": "app", "num_ranks": 4},
        ])
        self.assertIn(f"CPU run directory: {os.path.abspath('/cpu')}\n", header)
        self.assertIn(f"GPU run directory: {os.path.abspath('/gpu')}\n", header)
        # exactly one metadata block, not two
        self.assertEqual(header.count("executable:"), 1)
        self.assertEqual(header.count("MPI ranks:"), 1)
        self.assertIn("note: the 2 directories above are not checked against each other", header)

    def test_multiple_independent_runs_get_cross_check_caveat(self):
        header = rb.standard_header("extract_pop_metrics.py", "Desc.\n", [
            {"directories": [("reference run", "/a")]},
            {"directories": [("scaling run 2", "/b")]},
        ])
        self.assertIn("note: the 2 directories above are not checked against each other", header)

    def test_extra_lines_appended_verbatim(self):
        header = rb.standard_header("extract_calltree.py", "Desc.\n", [
            {"directories": [("source directory", "/x")],
             "extra_lines": ["  extra: yes\n"]},
        ])
        self.assertIn("  extra: yes\n", header)


if __name__ == "__main__":
    unittest.main()
