import os
import sys
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _test_helpers import load_module_by_path  # noqa: E402

fused = load_module_by_path("stage5_fused_hotspots_table", "stage5", "stage5_fused_hotspots_table.py")

import stage4_rocprofsys_sample_flat  # noqa: E402  (needs sys.path insert above first)
import stage4_rocprofv3  # noqa: E402
from stage5_table_render import render_table  # noqa: E402

CPU_DIR = os.path.join(FIXTURES, "mpi_2rank")
GPU_DIR = os.path.join(FIXTURES, "rocprofv3_mpi_2rank")
CPU_DIR_SINGLE = os.path.join(FIXTURES, "single_rank")
GPU_DIR_SINGLE = os.path.join(FIXTURES, "rocprofv3_single_rank")
CPU_DIR_GPU_API_NESTED_CHAIN = os.path.join(FIXTURES, "gpu_api_nested_chain")
CPU_DIR_GPU_SYNC_WAIT = os.path.join(FIXTURES, "gpu_sync_wait")


class BuildCombinedViewTests(unittest.TestCase):
    def test_subtraction_arithmetic_matches_documented_formula(self):
        _fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = fused.build_combined_view(
            CPU_DIR_GPU_SYNC_WAIT, GPU_DIR_SINGLE
        )

        # Independently recompute expected numbers straight from the sibling
        # stage4 modules' own aggregate() on the same fixtures, rather than hand-typing
        # decimals -- this is the actual documented formula, not a guess.
        exp_cpu_entries, exp_gpu_api_entries, exp_cpu_scanned, exp_cpu_total_raw = stage4_rocprofsys_sample_flat.aggregate(
            CPU_DIR_GPU_SYNC_WAIT
        )
        exp_gpu_entries, exp_gpu_scanned, exp_gpu_total_ns = stage4_rocprofv3.aggregate(GPU_DIR_SINGLE)
        # self_sum, not inclusive sum, and ONLY the two sync-wait labels -- see
        # build_combined_view()'s own docstring for why the rest of the GPU-API
        # bucket (which can include multi-thread-inflated self-time sums) is excluded.
        exp_overhead = sum(
            e["self_sum"] for e in exp_gpu_api_entries if e["label"] in fused.SYNC_WAIT_LABELS
        )
        exp_cpu_pure = max(0.0, exp_cpu_total_raw - exp_overhead)
        exp_gpu_total_sec = exp_gpu_total_ns / 1e9
        exp_combined_total = exp_cpu_pure + exp_gpu_total_sec

        self.assertAlmostEqual(info["cpu_total_raw"], exp_cpu_total_raw)
        self.assertAlmostEqual(info["gpu_api_overhead_sec"], exp_overhead)
        self.assertAlmostEqual(info["gpu_api_overhead_sec"], 3.5)  # hipStreamSynchronize(2.0) + hipDeviceSynchronize(1.5)
        self.assertAlmostEqual(info["cpu_pure_total_sec"], exp_cpu_pure)
        self.assertAlmostEqual(info["gpu_total_sec"], exp_gpu_total_sec)
        self.assertAlmostEqual(info["combined_total_sec"], exp_combined_total)
        # the subtraction must have actually removed something, not be a no-op
        self.assertLess(info["cpu_pure_total_sec"], info["cpu_total_raw"])

    def test_gpu_api_overhead_excludes_non_sync_wait_calls(self):
        # hipLaunchKernel is a real GPU-API entry (table 4 will still show it)
        # but it isn't a blocking sync call -- it must not feed the subtraction.
        _fused, _cpu_entries, cpu_gpu_api_entries, _gpu_entries, info = fused.build_combined_view(
            CPU_DIR_GPU_SYNC_WAIT, GPU_DIR_SINGLE
        )
        gpu_api_labels = {e["label"] for e in cpu_gpu_api_entries}
        self.assertIn("hipLaunchKernel", gpu_api_labels)  # still in table 4's source data
        self.assertAlmostEqual(info["gpu_api_overhead_sec"], 3.5)  # NOT 3.5 + hipLaunchKernel's 0.5

    def test_fused_pct_total_differs_from_each_sides_own_standalone_pct(self):
        fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = fused.build_combined_view(CPU_DIR, GPU_DIR)
        fused_by_label = {(e["label"], e["domain"]): e for e in fused_entries}
        cpu_by_label = {e["label"]: e for e in cpu_entries}
        gpu_by_label = {e["label"]: e for e in gpu_entries}

        # same absolute "sum" as the standalone tools...
        self.assertAlmostEqual(fused_by_label[("compute_stencil", "CPU")]["sum"], cpu_by_label["compute_stencil"]["sum"])
        self.assertAlmostEqual(fused_by_label[("JacobiIterationKernel", "GPU")]["sum"], gpu_by_label["JacobiIterationKernel"]["sum"])
        # ...but a DIFFERENT pct_total than each side's own standalone number,
        # proving genuine recombination happened (not the rejected no-op rescale).
        self.assertNotAlmostEqual(
            fused_by_label[("compute_stencil", "CPU")]["pct_total"],
            cpu_by_label["compute_stencil"]["pct_total"],
        )
        self.assertNotAlmostEqual(
            fused_by_label[("JacobiIterationKernel", "GPU")]["pct_total"],
            gpu_by_label["JacobiIterationKernel"]["pct_total"],
        )

    def test_fused_list_excludes_gpu_api_overhead_bucket(self):
        fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = fused.build_combined_view(CPU_DIR, GPU_DIR)
        fused_labels = {e["label"] for e in fused_entries}
        self.assertNotIn("hipMemcpy", fused_labels)
        self.assertIn("hipMemcpy", {e["label"] for e in cpu_gpu_api_entries})

    def test_mismatched_pairing_combines_without_error_gigo(self):
        # Deliberately mismatched: single_rank (CPU) with the 2-rank GPU fixture.
        # No cross-validation should happen -- this must not raise.
        fused_entries, cpu_entries, cpu_gpu_api_entries, gpu_entries, info = fused.build_combined_view(CPU_DIR_SINGLE, GPU_DIR)
        self.assertTrue(fused_entries)
        self.assertGreater(info["combined_total_sec"], 0)

    def test_gpu_api_overhead_does_not_overcount_a_nested_call_chain(self):
        # gpu_api_nested_chain fixture: hipStreamCreate -> hip::hipStreamCreate(...) ->
        # hip::ihipStreamCreate(...) -- none of these three labels is
        # hipStreamSynchronize/hipDeviceSynchronize, so this whole chain (a real
        # GPU-API cost, just not a blocking sync wait) contributes NOTHING to
        # the table-1 subtraction, even though table 4 still shows all three.
        _fused, _cpu_entries, cpu_gpu_api_entries, _gpu_entries, info = fused.build_combined_view(
            CPU_DIR_GPU_API_NESTED_CHAIN, GPU_DIR_SINGLE
        )
        self.assertEqual(len(cpu_gpu_api_entries), 3)
        self.assertAlmostEqual(info["gpu_api_overhead_sec"], 0.0)


class FusedHotspotsColumnsTests(unittest.TestCase):
    # Thin column-spec sanity check, not a render_table() test -- see
    # test_stage5_cpu_hotspots_table.py's CpuHotspotsColumnsTests for why.
    CASES = [
        ("includes_domain_column",
         [{"label": "k", "domain": "GPU", "count": 1, "sum": 1.0, "self_sum": 1.0, "pct_total": 50.0}],
         ["dom", "GPU"]),
        ("empty_entries", [], ["none found"]),
    ]

    def test_columns(self):
        for name, entries, expect_substrings in self.CASES:
            with self.subTest(case=name):
                table = render_table(fused.FUSED_HOTSPOTS_COLUMNS, entries)
                for substring in expect_substrings:
                    self.assertIn(substring, table)


if __name__ == "__main__":
    unittest.main()
