import importlib.util
import os
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location(
    "stage6_report_builder", os.path.join(POSTPROCESS_DIR, "stage6_report_builder.py")
)
rb = importlib.util.module_from_spec(spec)
sys.modules["stage6_report_builder"] = rb
spec.loader.exec_module(rb)


class WriteReportFileTests(unittest.TestCase):
    def test_joins_parts_writes_and_returns_same_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = rb.write_report_file(dest, ["a\n", "b\n", "c\n"])
            self.assertEqual(report, "a\nb\nc\n")
            with open(dest) as f:
                self.assertEqual(f.read(), report)


class RenderReportTests(unittest.TestCase):
    def test_header_sections_and_footer_assemble_in_order(self):
        parts = rb.render_report(
            "HEADER\n",
            [("Title 1\n", "Body 1\n"), ("Title 2\n", "Body 2\n")],
            "FOOTER\n",
        )
        self.assertEqual("".join(parts), "HEADER\nTitle 1\nBody 1\n\nTitle 2\nBody 2\n\nFOOTER\n")

    def test_falsy_title_emits_no_title_line(self):
        parts = rb.render_report("HEADER\n", [(None, "Body\n")])
        self.assertEqual("".join(parts), "HEADER\nBody\n\n")

    def test_empty_footer_emits_nothing_extra(self):
        parts = rb.render_report("HEADER\n", [(None, "Body\n")], "")
        self.assertEqual("".join(parts), "HEADER\nBody\n\n")

    def test_zero_sections_still_emits_header_and_footer(self):
        parts = rb.render_report("HEADER\n", [], "FOOTER\n")
        self.assertEqual("".join(parts), "HEADER\nFOOTER\n")


if __name__ == "__main__":
    unittest.main()
