import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

trace = load_module_by_path("stage3_rocprofsys_trace", "stage3", "stage3_rocprofsys_trace.py")

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)


def make_category_row(name, category, parent=None):
    return {"name": name, "category": category, "parent": parent}


class TagForCategoryTests(unittest.TestCase):
    # (category, expected_tag) -- the full mapping table, one input to one output each, plus the
    # case-insensitivity and unknown/None fallback rules. Table-driven since every case is the
    # same shape.
    CASES = [
        ("rocm_hip_api", "gpu_api"),
        ("rocm_rccl", "gpu_kernel"),
        ("rocm_rccl_api", "gpu_api"),  # distinct from the plain rccl (collective op) mapping above
        ("rocm_kernel_dispatch", "gpu_kernel"),
        ("rocm_memory_copy", "gpu_memcpy"),
        ("mpi", "mpi_territory"),
        ("numa", "other"),
        ("amd_smi_power", "other"),  # amd_smi_* prefix
        ("rocm_totally_new_category_v99", "other"),  # unrecognized future category falls back
        (None, "other"),
        ("ROCM_HIP_API", "gpu_api"),  # matching is case-insensitive
        ("host", None), ("ompt", None), ("pthread", None), ("sampling", None),
        ("python", None), ("user", None), ("kokkos", None), ("none", None),  # cpu-ish: no tag
    ]

    def test_category_maps_to_expected_tag(self):
        for category, expected in self.CASES:
            with self.subTest(category=category):
                self.assertEqual(trace.tag_for_category(category), expected)


class TagRowsTests(unittest.TestCase):
    def setUp(self):
        stage6_noise_config.configure(None)

    def test_gpu_api_category_tagged(self):
        row = make_category_row("hipLaunchKernel", "rocm_hip_api")
        trace.tag_rows([row])
        self.assertIn("gpu_api", row["tags"])

    def test_mpi_category_tagged(self):
        row = make_category_row("MPI_Barrier", "mpi")
        trace.tag_rows([row])
        self.assertIn("mpi_territory", row["tags"])

    def test_kernel_dispatch_category_tagged(self):
        row = make_category_row("jacobi_kernel.kd", "rocm_kernel_dispatch")
        trace.tag_rows([row])
        self.assertIn("gpu_kernel", row["tags"])

    def test_memcpy_category_tagged(self):
        row = make_category_row("hipMemcpyAsync", "rocm_memory_copy")
        trace.tag_rows([row])
        self.assertIn("gpu_memcpy", row["tags"])

    def test_wrapper_noise_by_name_under_host_category(self):
        row = make_category_row("gotcha_call", "host")
        trace.tag_rows([row])
        self.assertIn("wrapper_noise", row["tags"])

    def test_ordinary_host_row_gets_no_tag(self):
        row = make_category_row("run_simulation", "host")
        trace.tag_rows([row])
        self.assertEqual(row["tags"], set())

    def test_numa_category_tagged_other(self):
        row = make_category_row("numa_migration_event", "numa")
        trace.tag_rows([row])
        self.assertIn("other", row["tags"])

    def test_unmapped_category_tagged_other(self):
        row = make_category_row("mystery_event", "rocm_totally_new_category_v99")
        trace.tag_rows([row])
        self.assertIn("other", row["tags"])

    def test_category_tag_and_name_pattern_tag_both_survive(self):
        # "mpi" gives an exact category tag; the name also matches wrapper_noise by substring --
        # the category-derived tag must not clobber (or be clobbered by) the name-derived one.
        row = make_category_row("gotcha_mpi_helper", "mpi")
        trace.tag_rows([row])
        self.assertIn("mpi_territory", row["tags"])
        self.assertIn("wrapper_noise", row["tags"])


if __name__ == "__main__":
    unittest.main()
