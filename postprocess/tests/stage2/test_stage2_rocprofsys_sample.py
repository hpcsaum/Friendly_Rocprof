import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

stage2 = load_module_by_path("stage2_rocprofsys_sample", "stage2", "stage2_rocprofsys_sample.py")


class AttachAncestryTests(unittest.TestCase):
    def test_chain_within_one_thread(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 1, "thread_id": "0"},
            {"label": "c", "depth": 2, "thread_id": "0"},
        ]
        stage2.attach_ancestry(rows)
        self.assertIsNone(rows[0]["parent"])
        self.assertFalse(rows[0]["is_thread_root"])
        self.assertIs(rows[1]["parent"], rows[0])
        self.assertFalse(rows[1]["is_thread_root"])
        self.assertIs(rows[2]["parent"], rows[1])
        # contrast with test_thread_change_relative_to_parent_flags_thread_root below: same
        # thread_id as its parent must not flag as a thread root.
        self.assertFalse(rows[2]["is_thread_root"])

    def test_sibling_depth_zero_rows_have_no_parent(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 0, "thread_id": "0"},
        ]
        stage2.attach_ancestry(rows)
        self.assertIsNone(rows[1]["parent"])
        self.assertFalse(rows[1]["is_thread_root"])

    def test_thread_change_relative_to_parent_flags_thread_root(self):
        rows = [
            {"label": "a", "depth": 0, "thread_id": "0"},
            {"label": "b", "depth": 1, "thread_id": "0"},
            {"label": "c", "depth": 2, "thread_id": "1"},
        ]
        stage2.attach_ancestry(rows)
        self.assertIs(rows[2]["parent"], rows[1])
        self.assertTrue(rows[2]["is_thread_root"])


if __name__ == "__main__":
    unittest.main()
