import importlib.util
import json
import os
import sys
import tempfile
import unittest

POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location(
    "stage6_noise_config", os.path.join(POSTPROCESS_DIR, "stage6_noise_config.py")
)
nc = importlib.util.module_from_spec(spec)
sys.modules["stage6_noise_config"] = nc
spec.loader.exec_module(nc)


def _write_diff(tmp, diff):
    path = os.path.join(tmp, "noise_config.json")
    with open(path, "w") as f:
        json.dump(diff, f)
    return path


class ConfigureAndTagDefsTests(unittest.TestCase):
    def tearDown(self):
        nc.configure(None)  # never let one test's active config leak into the next

    def test_no_config_returns_bundled_defaults_plus_empty_other(self):
        nc.configure(None)
        defs = nc.tag_defs()
        self.assertEqual(
            set(defs.keys()),
            {"gpu_api", "wrapper_noise", "mpi_territory", "compiler_runtime_noise",
             "wrapper_branch_noise", "other"},
        )
        self.assertEqual(defs["other"], {"substrings": []})

    def test_lazy_configure_on_first_use(self):
        # never explicitly configure()d -- tag_defs() should still resolve to the bundled
        # defaults, not raise or return None
        defs = nc.tag_defs()
        self.assertIn("gpu_api", defs)

    def test_add_extends_an_existing_tags_substrings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"add": {"wrapper_noise": ["my_site_gotcha_variant"]}})
            nc.configure(path)
        substrings = nc.tag_defs()["wrapper_noise"]["substrings"]
        self.assertIn("my_site_gotcha_variant", substrings)
        self.assertIn("gotcha", substrings)  # bundled substrings survive -- add, not replace

    def test_add_lowercases_mixed_case_input_to_match_case_insensitively(self):
        # _label_matches() lowercases the row's label but not the pattern, so a mixed-case
        # substring must be normalized here or it would silently never match anything.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"add": {"other": ["PMPI_Allreduce"]}})
            nc.configure(path)
        self.assertIn("pmpi_allreduce", nc.tag_defs()["other"]["substrings"])

    def test_remove_lowercases_mixed_case_input_to_match_bundled_lowercase_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"remove": {"wrapper_noise": ["GOTCHA"]}})
            nc.configure(path)
        self.assertNotIn("gotcha", nc.tag_defs()["wrapper_noise"]["substrings"])

    def test_add_extends_the_other_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"add": {"other": ["some_vendor_specific_noise_"]}})
            nc.configure(path)
        self.assertEqual(nc.tag_defs()["other"]["substrings"], ["some_vendor_specific_noise_"])

    def test_remove_excludes_named_substrings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"remove": {"wrapper_noise": ["gotcha"]}})
            nc.configure(path)
        self.assertNotIn("gotcha", nc.tag_defs()["wrapper_noise"]["substrings"])

    def test_disable_drops_the_tag_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"disable": ["compiler_runtime_noise"]})
            nc.configure(path)
        self.assertNotIn("compiler_runtime_noise", nc.tag_defs())

    def test_disable_makes_add_remove_on_same_tag_a_no_op_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {
                "disable": ["compiler_runtime_noise"],
                "add": {"compiler_runtime_noise": ["should_be_ignored"]},
                "remove": {"compiler_runtime_noise": ["_f90_"]},
            })
            nc.configure(path)  # must not raise
        self.assertNotIn("compiler_runtime_noise", nc.tag_defs())

    def test_unknown_tag_in_disable_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"disable": ["not_a_real_tag"]})
            with self.assertRaises(SystemExit):
                nc.configure(path)

    def test_unknown_tag_in_add_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"add": {"not_a_real_tag": ["x"]}})
            with self.assertRaises(SystemExit):
                nc.configure(path)

    def test_unknown_tag_in_remove_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"remove": {"not_a_real_tag": ["x"]}})
            with self.assertRaises(SystemExit):
                nc.configure(path)

    def test_add_targeting_derived_tag_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"add": {"wrapper_branch_noise": ["x"]}})
            with self.assertRaises(SystemExit):
                nc.configure(path)

    def test_remove_targeting_derived_tag_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"remove": {"wrapper_branch_noise": ["x"]}})
            with self.assertRaises(SystemExit):
                nc.configure(path)

    def test_disable_targeting_derived_tag_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_diff(tmp, {"disable": ["wrapper_branch_noise"]})
            nc.configure(path)  # must not raise
        self.assertNotIn("wrapper_branch_noise", nc.tag_defs())

    def test_configure_again_fully_replaces_prior_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1 = _write_diff(tmp, {"add": {"other": ["first"]}})
            nc.configure(path1)
            self.assertEqual(nc.tag_defs()["other"]["substrings"], ["first"])
            path2 = _write_diff(tmp, {"add": {"other": ["second"]}})
            nc.configure(path2)
        # ["second"], not ["first", "second"] -- configure() never merges with a prior config
        self.assertEqual(nc.tag_defs()["other"]["substrings"], ["second"])


if __name__ == "__main__":
    unittest.main()
