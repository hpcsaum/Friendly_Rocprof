import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

rm = load_module_by_path("stage6_run_metadata", "stage6", "stage6_run_metadata.py")

PID_SUFFIX_RE = re.compile(r"(\d+)\.txt$")
DIR_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}")


class LoadJsonFileTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rm.load_json_file(tmp, "metadata.json"), {})

    def test_found_file_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                f.write('{"executable": "app"}')
            self.assertEqual(rm.load_json_file(tmp, "metadata.json"), {"executable": "app"})

    def test_corrupt_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                f.write("not json")
            self.assertEqual(rm.load_json_file(tmp, "metadata.json"), {})

    def test_non_dict_content_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                f.write("[1, 2, 3]")
            self.assertEqual(rm.load_json_file(tmp, "metadata.json"), {})


class FindFirstKeyTests(unittest.TestCase):
    def test_flat_case_insensitive_match(self):
        self.assertEqual(rm.find_first_key({"Command": "app"}, ["command"]), "app")

    def test_one_level_nested_match(self):
        self.assertEqual(rm.find_first_key({"outer": {"command": "app"}}, ["command"]), "app")

    def test_no_match_returns_none(self):
        self.assertIsNone(rm.find_first_key({"other": "x"}, ["command"]))


class GuessExecutableTests(unittest.TestCase):
    def test_value_found_basename_of_first_token(self):
        self.assertEqual(rm.guess_executable({"command": "/path/to/app --flag"}, ["command"]), "app")

    def test_list_unwrapped(self):
        self.assertEqual(rm.guess_executable({"argv": ["/path/to/app", "--flag"]}, ["argv"]), "app")

    def test_missing_returns_none(self):
        self.assertIsNone(rm.guess_executable({}, ["command"]))


class GuessTotalRuntimeTests(unittest.TestCase):
    def test_numeric_value_formatted(self):
        self.assertEqual(rm.guess_total_runtime({"elapsed": 1.5}, ["elapsed"]), "1.500000 sec")

    def test_string_value_stripped(self):
        self.assertEqual(rm.guess_total_runtime({"elapsed": " 2 sec "}, ["elapsed"]), "2 sec")

    def test_missing_returns_none(self):
        self.assertIsNone(rm.guess_total_runtime({}, ["elapsed"]))


class GuessRunDatetimeTests(unittest.TestCase):
    def test_key_hit(self):
        self.assertEqual(rm.guess_run_datetime({"timestamp": "2025-01-01"}, ["timestamp"]), "2025-01-01")

    def test_key_miss_falls_back_to_output_dir(self):
        result = rm.guess_run_datetime(
            {}, ["timestamp"], output_dir="/runs/2025-01-21_07.40/out", dir_pattern=DIR_PATTERN,
        )
        self.assertEqual(result, "2025-01-21_07.40")

    def test_key_miss_falls_back_to_scanned_files(self):
        result = rm.guess_run_datetime(
            {}, ["timestamp"], output_dir="/runs/out",
            scanned_files=["/runs/2025-01-21_07.40/wall_clock-1.txt"], dir_pattern=DIR_PATTERN,
        )
        self.assertEqual(result, "2025-01-21_07.40")

    def test_key_miss_no_dir_pattern_returns_none_immediately(self):
        # Matches extract_GPU_hotspots.py's simpler call shape: no directory-name
        # fallback convention to try at all.
        self.assertIsNone(rm.guess_run_datetime({}, ["timestamp"]))


class GuessNumRanksTests(unittest.TestCase):
    def test_keys_given_and_hit(self):
        self.assertEqual(rm.guess_num_ranks({"num_ranks": 4}, PID_SUFFIX_RE, [], keys=["num_ranks"]), 4)

    def test_keys_given_but_missed_falls_back_to_pid_counting(self):
        files = ["wall_clock-1.txt", "wall_clock-2.txt"]
        self.assertEqual(rm.guess_num_ranks({}, PID_SUFFIX_RE, files, keys=["num_ranks"]), 2)

    def test_no_keys_goes_straight_to_pid_counting(self):
        # Matches extract_GPU_hotspots.py's call shape: no metadata key list at all.
        files = ["wall_clock-1.txt", "wall_clock-2.txt", "wall_clock-1.txt"]
        self.assertEqual(rm.guess_num_ranks({}, PID_SUFFIX_RE, files), 2)

    def test_no_matching_files_returns_none(self):
        self.assertIsNone(rm.guess_num_ranks({}, PID_SUFFIX_RE, []))


if __name__ == "__main__":
    unittest.main()
