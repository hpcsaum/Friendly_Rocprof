"""Tests for stage6_cli_common.py's shared argparse/validation helpers used across the CLI tools.

RequireDirectoryTests   -- require_directory() pass/raise on a single path
RequireDirectoriesTests -- require_directories() over multiple paths, None-skipping, first-missing reporting
ResolveDestTests        -- resolve_dest()'s explicit-vs-default-path resolution
AddSelectionArgsTests   -- add_selection_args()'s -n/--top/--threshold/--all group and its help-text substitution
AddMaxDepthArgTests     -- add_max_depth_arg()'s --max-depth flag and its default/overridden help text
AddNoiseTierArgsTests   -- add_noise_tier_args()'s per-tier --show-* flags and the --show-all-internals shorthand
"""

import argparse
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

cc = load_module_by_path("stage6_cli_common", "stage6", "stage6_cli_common.py")


class RequireDirectoryTests(unittest.TestCase):
    def test_existing_directory_passes_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            cc.require_directory(tmp)  # must not raise

    def test_missing_directory_raises_standard_message(self):
        with self.assertRaises(SystemExit) as ctx:
            cc.require_directory("/no/such/directory")
        self.assertIn("no such directory: '/no/such/directory'", str(ctx.exception))


class RequireDirectoriesTests(unittest.TestCase):
    def test_all_existing_passes_silently(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            cc.require_directories([tmp1, tmp2])  # must not raise

    def test_one_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cc.require_directories([tmp, "/no/such/directory"])

    def test_none_entries_are_skipped_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cc.require_directories([tmp, None])  # must not raise

    def test_first_missing_reported_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                cc.require_directories(["/no/such/directory", tmp])
            self.assertIn("/no/such/directory", str(ctx.exception))


class ResolveDestTests(unittest.TestCase):
    def test_explicit_dest_wins(self):
        self.assertEqual(cc.resolve_dest("/explicit/path.txt", "/default/dir", "default.txt"), "/explicit/path.txt")

    def test_falls_back_to_default(self):
        self.assertEqual(cc.resolve_dest(None, "/default/dir", "default.txt"), os.path.join("/default/dir", "default.txt"))

    def test_empty_string_dest_also_falls_back(self):
        self.assertEqual(cc.resolve_dest("", "/default/dir", "default.txt"), os.path.join("/default/dir", "default.txt"))


class AddSelectionArgsTests(unittest.TestCase):
    def _parser(self, **kwargs):
        parser = argparse.ArgumentParser()
        cc.add_selection_args(parser, **kwargs)
        return parser

    def test_mutually_exclusive(self):
        parser = self._parser(plural_noun="entries", threshold_unit_help="of total runtime")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--top", "5", "--all"])

    # (case, add_selection_args kwargs, dest of the argparse action to inspect, substring its
    # .help text must contain) -- every case here builds a parser with one varied kwarg and
    # checks one substituted phrase, the same shape each time.
    HELP_TEXT_CASES = [
        ("default_verb_is_list_top",
         {"plural_noun": "kernels", "threshold_unit_help": "of total runtime"}, "top", "to list"),
        ("default_verb_is_list_all",
         {"plural_noun": "kernels", "threshold_unit_help": "of total runtime"}, "show_all", "list every kernel"),
        ("verb_override",
         {"plural_noun": "functions", "threshold_unit_help": "of total runtime", "verb": "select"},
         "top", "to select"),
        ("top_noun_defaults_to_plural_noun",
         {"plural_noun": "kernels", "threshold_unit_help": "of total device time"},
         "top", "number of kernels to list"),
        ("top_noun_override",
         {"plural_noun": "kernels", "threshold_unit_help": "of total device time", "top_noun": "hotspot kernels"},
         "top", "number of hotspot kernels to list"),
        ("singular_noun_defaults_by_stripping_trailing_s",
         {"plural_noun": "kernels", "threshold_unit_help": "of total device time"}, "show_all", "every kernel,"),
        ("singular_noun_override_for_irregular_plural",
         {"plural_noun": "entries", "threshold_unit_help": "of total runtime", "singular_noun": "entry"},
         "show_all", "every entry,"),
        ("threshold_unit_help_substituted",
         {"plural_noun": "entries", "threshold_unit_help": "of their table's total"},
         "threshold", "of their table's total"),
    ]

    def test_help_text_substitution(self):
        for name, kwargs, dest, expected_substring in self.HELP_TEXT_CASES:
            with self.subTest(case=name):
                parser = self._parser(**kwargs)
                action = next(a for a in parser._actions if a.dest == dest)
                self.assertIn(expected_substring, action.help)

    def test_returns_the_group_for_further_extension(self):
        parser = argparse.ArgumentParser()
        group = cc.add_selection_args(parser, "entries", "of total runtime")
        group.add_argument("--extra-flag", action="store_true")
        args = parser.parse_args(["--extra-flag"])
        self.assertTrue(args.extra_flag)


class AddMaxDepthArgTests(unittest.TestCase):
    def test_parses_to_int_dest(self):
        parser = argparse.ArgumentParser()
        cc.add_max_depth_arg(parser)
        args = parser.parse_args(["--max-depth", "3"])
        self.assertEqual(args.max_depth, 3)

    def test_defaults_to_none(self):
        parser = argparse.ArgumentParser()
        cc.add_max_depth_arg(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.max_depth)

    def test_default_help_text_is_root_relative(self):
        parser = argparse.ArgumentParser()
        cc.add_max_depth_arg(parser)
        action = next(a for a in parser._actions if a.dest == "max_depth")
        self.assertIn("truncate the tree at this depth", action.help)

    def test_custom_help_text_overrides_default(self):
        parser = argparse.ArgumentParser()
        cc.add_max_depth_arg(parser, help_text="truncate upward from a known target instead")
        action = next(a for a in parser._actions if a.dest == "max_depth")
        self.assertEqual(action.help, "truncate upward from a known target instead")


class AddNoiseTierArgsTests(unittest.TestCase):
    def test_subset_adds_only_those_flags(self):
        parser = argparse.ArgumentParser()
        cc.add_noise_tier_args(parser, ["gpu_api"])
        args = parser.parse_args(["--show-gpu-api"])
        self.assertTrue(args.show_gpu_api)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--show-mpi-internals"])

    def test_show_all_internals_absent_when_all_shorthand_false(self):
        parser = argparse.ArgumentParser()
        cc.add_noise_tier_args(parser, ["gpu_api", "mpi_internals"], all_shorthand=False)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--show-all-internals"])

    def test_show_all_internals_absent_for_a_single_tier_even_with_shorthand_true(self):
        parser = argparse.ArgumentParser()
        cc.add_noise_tier_args(parser, ["gpu_api"], all_shorthand=True)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--show-all-internals"])

    def test_show_all_internals_present_for_multiple_tiers_with_shorthand_true(self):
        parser = argparse.ArgumentParser()
        cc.add_noise_tier_args(
            parser, ["gpu_api", "rocprofsys_internals", "mpi_internals", "compiler_runtime"],
            all_shorthand=True,
        )
        args = parser.parse_args(["--show-all-internals"])
        self.assertTrue(args.show_all_internals)

    def test_all_four_tiers_get_distinct_dests(self):
        parser = argparse.ArgumentParser()
        cc.add_noise_tier_args(
            parser, ["gpu_api", "rocprofsys_internals", "mpi_internals", "compiler_runtime"],
        )
        args = parser.parse_args([
            "--show-gpu-api", "--show-rocprofsys-internals", "--show-mpi-internals", "--show-compiler-runtime",
        ])
        self.assertTrue(args.show_gpu_api)
        self.assertTrue(args.show_rocprofsys_internals)
        self.assertTrue(args.show_mpi_internals)
        self.assertTrue(args.show_compiler_runtime)


if __name__ == "__main__":
    unittest.main()
