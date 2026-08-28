"""Tests for debug_hang.py's pure logic: scheduler/node discovery, environment
capture/replay, and CLI parsing. Deliberately does not touch anything that
needs real HPC hardware (rocgdb, rocm-smi, pbsdsh, srun) -- this dev
environment has none of those, per CLAUDE.md's environment-constraints note.

PbsNodeIndicesTests   -- PBS_NODEFILE dedup/index-mapping
LoadEnvTests          -- NUL-separated env dump round-trip
DetectSchedulerTests  -- env-var and PATH-based scheduler detection
ParseArgsTests        -- --mpi prefixing, leading `--` stripping, --log defaulting
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..")


def _load_debug_hang():
    file_path = os.path.join(SCRIPTS_DIR, "debug_hang.py")
    spec = importlib.util.spec_from_file_location("debug_hang", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["debug_hang"] = module
    spec.loader.exec_module(module)
    return module


debug_hang = _load_debug_hang()


class PbsNodeIndicesTests(unittest.TestCase):
    def test_maps_each_host_to_its_first_line_in_the_nodefile(self):
        # node02 appears on lines 2 and 3 (0-indexed); its first line is 2.
        lines = ["node01", "node02", "node02", "node03"]
        with tempfile.TemporaryDirectory() as tmp:
            nodefile = os.path.join(tmp, "nodefile")
            with open(nodefile, "w") as f:
                f.write("\n".join(lines) + "\n")
            with mock.patch.dict(os.environ, {"PBS_NODEFILE": nodefile}):
                indices = debug_hang.pbs_node_indices(["node01", "node02", "node03"])
        self.assertEqual(indices, [0, 1, 3])

    def test_falls_back_to_positional_index_for_a_host_missing_from_the_nodefile(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodefile = os.path.join(tmp, "nodefile")
            with open(nodefile, "w") as f:
                f.write("node01\n")
            with mock.patch.dict(os.environ, {"PBS_NODEFILE": nodefile}):
                indices = debug_hang.pbs_node_indices(["node01", "node-not-in-file"])
        self.assertEqual(indices, [0, 1])

    def test_no_nodefile_set_falls_back_to_positional_indices_for_every_host(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PBS_NODEFILE", None)
            indices = debug_hang.pbs_node_indices(["a", "b", "c"])
        self.assertEqual(indices, [0, 1, 2])


class LoadEnvTests(unittest.TestCase):
    def test_round_trips_a_nul_separated_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env")
            with open(path, "wb") as f:
                f.write(b"FOO=bar\0BAZ=qux\0")
            self.assertEqual(debug_hang.load_env(path), {"FOO": "bar", "BAZ": "qux"})

    def test_value_containing_an_equals_sign_is_split_only_on_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env")
            with open(path, "wb") as f:
                f.write(b"EQ=a=b=c\0")
            self.assertEqual(debug_hang.load_env(path), {"EQ": "a=b=c"})

    def test_missing_path_returns_none(self):
        self.assertIsNone(debug_hang.load_env("/nonexistent/path/should/not/exist"))

    def test_falsy_path_returns_none_without_touching_the_filesystem(self):
        self.assertIsNone(debug_hang.load_env(None))
        self.assertIsNone(debug_hang.load_env(""))


class DetectSchedulerTests(unittest.TestCase):
    def _no_scheduler_env(self):
        return mock.patch.dict(os.environ, {"SLURM_JOB_ID": "", "PBS_JOBID": ""})

    def test_slurm_job_id_env_var_wins(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "PBS_JOBID": ""}):
            self.assertEqual(debug_hang.detect_scheduler(), "slurm")

    def test_pbs_jobid_env_var_wins_when_slurm_is_absent(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "", "PBS_JOBID": "7.pbs"}):
            self.assertEqual(debug_hang.detect_scheduler(), "pbs")

    def test_falls_back_to_scancel_on_path_when_no_job_env_vars_are_set(self):
        with self._no_scheduler_env():
            with mock.patch("debug_hang.shutil.which",
                            side_effect=lambda c: "/usr/bin/scancel" if c == "scancel" else None):
                self.assertEqual(debug_hang.detect_scheduler(), "slurm")

    def test_falls_back_to_qdel_on_path_when_scancel_is_absent(self):
        with self._no_scheduler_env():
            with mock.patch("debug_hang.shutil.which",
                            side_effect=lambda c: "/usr/bin/qdel" if c == "qdel" else None):
                self.assertEqual(debug_hang.detect_scheduler(), "pbs")

    def test_local_when_nothing_indicates_a_scheduler(self):
        with self._no_scheduler_env():
            with mock.patch("debug_hang.shutil.which", return_value=None):
                self.assertEqual(debug_hang.detect_scheduler(), "local")


class ParseArgsTests(unittest.TestCase):
    def test_mpi_flag_is_captured_separately_from_the_trailing_command(self):
        args = debug_hang.parse_args(
            ["--mpi", "mpirun -np 4", "--timeout", "42", "--", "./app", "arg1"])
        self.assertEqual(args.mpi, "mpirun -np 4")
        self.assertEqual(args.command, ["./app", "arg1"])
        self.assertEqual(args.timeout, 42)

    def test_leading_double_dash_is_stripped_from_command(self):
        args = debug_hang.parse_args(["--", "./app"])
        self.assertEqual(args.command, ["./app"])

    def test_command_given_without_log_gets_a_pid_based_default_log(self):
        args = debug_hang.parse_args(["--", "./app"])
        self.assertEqual(args.log, f"debug_hang-{os.getpid()}.log")

    def test_explicit_log_is_not_overridden_when_a_command_is_also_given(self):
        args = debug_hang.parse_args(["--log", "custom.log", "--", "./app"])
        self.assertEqual(args.log, "custom.log")

    def test_log_only_mode_needs_no_command(self):
        args = debug_hang.parse_args(["--log", "existing.log"])
        self.assertEqual(args.command, [])
        self.assertEqual(args.log, "existing.log")

    def test_no_command_and_no_log_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            debug_hang.parse_args([])

    def test_mpi_flag_defaults_to_none_for_a_single_rank_run(self):
        args = debug_hang.parse_args(["--", "./app"])
        self.assertIsNone(args.mpi)


if __name__ == "__main__":
    unittest.main()
