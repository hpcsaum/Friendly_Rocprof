import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

trace = load_module_by_path("stage3_rocprofsys_trace", "stage3", "stage3_rocprofsys_trace.py")

import stage3_rocprofsys_common  # noqa: E402  (needs sys.path insert above first)
import stage6_noise_config  # noqa: E402


def make_category_row(name, category, parent=None):
    return {"name": name, "category": category, "parent": parent}


class TagForCategoryTests(unittest.TestCase):
    def test_hip_api_maps_to_gpu_api(self):
        self.assertEqual(trace.tag_for_category("rocm_hip_api"), "gpu_api")

    def test_rccl_collective_op_maps_to_gpu_kernel(self):
        self.assertEqual(trace.tag_for_category("rocm_rccl"), "gpu_kernel")

    def test_rccl_api_maps_to_gpu_api_distinct_from_rccl_op(self):
        self.assertEqual(trace.tag_for_category("rocm_rccl_api"), "gpu_api")

    def test_kernel_dispatch_maps_to_gpu_kernel(self):
        self.assertEqual(trace.tag_for_category("rocm_kernel_dispatch"), "gpu_kernel")

    def test_memory_copy_maps_to_gpu_memcpy(self):
        self.assertEqual(trace.tag_for_category("rocm_memory_copy"), "gpu_memcpy")

    def test_mpi_maps_to_mpi_territory(self):
        self.assertEqual(trace.tag_for_category("mpi"), "mpi_territory")

    def test_numa_maps_to_other(self):
        self.assertEqual(trace.tag_for_category("numa"), "other")

    def test_amd_smi_prefix_maps_to_other(self):
        self.assertEqual(trace.tag_for_category("amd_smi_power"), "other")

    def test_cpu_ish_categories_get_no_category_tag(self):
        for category in ("host", "ompt", "pthread", "sampling", "python", "user", "kokkos", "none"):
            with self.subTest(category=category):
                self.assertIsNone(trace.tag_for_category(category))

    def test_unrecognized_future_category_falls_back_to_other(self):
        self.assertEqual(trace.tag_for_category("rocm_totally_new_category_v99"), "other")

    def test_none_category_falls_back_to_other(self):
        self.assertEqual(trace.tag_for_category(None), "other")

    def test_matching_is_case_insensitive(self):
        self.assertEqual(trace.tag_for_category("ROCM_HIP_API"), "gpu_api")


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

    def test_wrapper_branch_noise_sibling_derivation_still_works(self):
        root = make_category_row("main", "host")
        contaminated_top = make_category_row("std::pair<...>", "host", parent=root)
        buried = make_category_row("gotcha_call", "host", parent=contaminated_top)
        clean_sibling = make_category_row("run_simulation", "host", parent=root)
        rows = [root, contaminated_top, buried, clean_sibling]
        trace.tag_rows(rows)
        self.assertIn("wrapper_branch_noise", contaminated_top["structural_drop_tags"])
        self.assertEqual(clean_sibling["structural_drop_tags"], set())

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


class ReexportedPrimitivesTests(unittest.TestCase):
    def test_primitives_are_the_same_objects_as_stage3_rocprofsys_common(self):
        self.assertIs(trace.remove_tagged_subtrees, stage3_rocprofsys_common.remove_tagged_subtrees)
        self.assertIs(trace.splice_by_tag, stage3_rocprofsys_common.splice_by_tag)
        self.assertIs(trace.make_collapses_children, stage3_rocprofsys_common.make_collapses_children)
        self.assertIs(trace.make_is_pruned, stage3_rocprofsys_common.make_is_pruned)


if __name__ == "__main__":
    unittest.main()
