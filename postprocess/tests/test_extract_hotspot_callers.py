import importlib.util
import json
import os
import sys
import tempfile
import unittest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSTPROCESS_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_PATH = os.path.join(POSTPROCESS_DIR, "extract_hotspot_callers.py")

# extract_hotspot_callers.py does a plain top-level "from stage4_rocprofsys_flat import ...",
# relying on its own directory being on sys.path -- true automatically when run directly, but not
# when loaded here by explicit file path, so replicate that manually (same as the other tool tests).
sys.path.insert(0, os.path.abspath(POSTPROCESS_DIR))

spec = importlib.util.spec_from_file_location("extract_hotspot_callers", MODULE_PATH)
hc_tool = importlib.util.module_from_spec(spec)
sys.modules["extract_hotspot_callers"] = hc_tool
spec.loader.exec_module(hc_tool)

import stage6_noise_config  # noqa: E402  (needs sys.path insert above first)

MPI_2RANK_DIR = os.path.join(FIXTURES, "mpi_2rank")
SINGLE_RANK_DIR = os.path.join(FIXTURES, "single_rank")
# main -> PMPI_Waitall -> MPIR_Typerep_icopy is a real 3-level, all-CPU (no gpu_api tag) chain --
# calltree_kernel_anchor's own 3rd level is hipLaunchKernel, which aggregate() classifies as GPU
# API overhead, not a CPU hotspot, so it can never be this tool's own top-N target.
POP_REF_2RANK_DIR = os.path.join(FIXTURES, "pop_ref_2rank")
EMPTY_DIR = os.path.join(FIXTURES, "no_timing_data")


def _stub_node_values(node):
    return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class RenderChainTests(unittest.TestCase):
    """_render_chain()'s own truncation math, isolated from real tree data -- max_depth counts
    UPWARD from the target (last element of chain), so a 4-element chain with max_depth=1 keeps
    only the last 2 elements (the target plus its one nearest caller)."""

    CHAIN = [{"label": "root"}, {"label": "mid"}, {"label": "near"}, {"label": "target"}]

    def test_no_max_depth_keeps_whole_chain_root_flush(self):
        rows = hc_tool._render_chain(self.CHAIN, _stub_node_values, None)
        self.assertEqual([text for text, _values in rows], [
            "root", "    └── mid", "        └── near", "            └── target",
        ])

    def test_max_depth_truncates_upward_from_target(self):
        rows = hc_tool._render_chain(self.CHAIN, _stub_node_values, 1)
        texts = [text for text, _values in rows]
        self.assertEqual(len(texts), 3)  # 1 marker + 2 visible nodes (near, target)
        self.assertIn("2 more ancestor(s) hidden above this point", texts[0])
        self.assertIsNone(rows[0][1])
        self.assertEqual(texts[1], "└── near")
        self.assertEqual(texts[2], "    └── target")

    def test_max_depth_covering_whole_chain_adds_no_marker(self):
        rows = hc_tool._render_chain(self.CHAIN, _stub_node_values, 3)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0][0], "root")

    def test_single_node_chain_unaffected_by_max_depth(self):
        rows = hc_tool._render_chain([{"label": "root"}], _stub_node_values, 0)
        self.assertEqual([text for text, _values in rows], ["root"])


class WriteReportTests(unittest.TestCase):
    def test_end_to_end_on_mpi_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = hc_tool.write_report(MPI_2RANK_DIR, dest, top=3)
            self.assertTrue(os.path.isfile(dest))
            self.assertIn("Top CPU hotspots", report)
            self.assertIn("Caller chain(s) for 'compute_stencil'", report)
            chain_section = report[report.index("Caller chain(s) for 'compute_stencil'"):]
            self.assertIn("main", chain_section)
            self.assertIn("└── compute_stencil", chain_section)

    def test_root_only_function_renders_as_single_line_chain(self):
        # single_rank's own timemory table is flat (every row at DEPTH 0) -- each label is its
        # own root, so its "caller chain" is just itself, not an error.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = hc_tool.write_report(SINGLE_RANK_DIR, dest, top=1)
            self.assertIn("Caller chain(s) for 'compute_stencil'", report)
            chain_section = report[report.index("Caller chain(s) for 'compute_stencil'"):]
            self.assertIn("compute_stencil", chain_section)
            self.assertNotIn("└──", chain_section)

    def test_max_depth_truncates_a_real_three_level_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = hc_tool.write_report(POP_REF_2RANK_DIR, dest, show_all=True, max_depth=1)
            start = report.index("=== 4. Caller chain(s) for 'MPIR_Typerep_icopy'")
            chain_section = report[start:report.index("=== 5.")]
            self.assertIn("more ancestor(s) hidden above this point", chain_section)
            self.assertNotIn("main", chain_section)
            self.assertIn("PMPI_Waitall", chain_section)
            self.assertIn("MPIR_Typerep_icopy", chain_section)

    def test_no_max_depth_shows_the_whole_three_level_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = hc_tool.write_report(POP_REF_2RANK_DIR, dest, show_all=True)
            chain_section = report[report.index("Caller chain(s) for 'MPIR_Typerep_icopy'"):]
            self.assertNotIn("hidden above this point", chain_section)
            self.assertIn("main", chain_section)
            self.assertIn("PMPI_Waitall", chain_section)

    def test_raises_when_no_timing_data_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            with self.assertRaises(SystemExit):
                hc_tool.write_report(EMPTY_DIR, dest)

    def test_unfiltered_ranks_by_inclusive_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            report = hc_tool.write_report(MPI_2RANK_DIR, dest, show_all=True, unfiltered=True)
            self.assertIn("Ranked by inclusive (total) time", report)
            hotspots_section = report[report.index("=== 1."):report.index("=== 2.")]
            self.assertLess(hotspots_section.index("main"), hotspots_section.index("compute_stencil"))


class MainCliTests(unittest.TestCase):
    def tearDown(self):
        stage6_noise_config.configure(None)

    def test_missing_directory_raises_clear_error(self):
        with self.assertRaises(SystemExit):
            hc_tool.main(["/no/such/directory"])

    def test_end_to_end_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            hc_tool.main([MPI_2RANK_DIR, "-o", dest, "-n", "2"])
            self.assertTrue(os.path.isfile(dest))

    def test_default_top_is_10_when_no_selection_flag_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            hc_tool.main([MPI_2RANK_DIR, "-o", dest])
            with open(dest) as f:
                report = f.read()
        self.assertIn("--top 10", report)

    def test_extra_noise_config_flag_excludes_a_configured_hotspot_entirely(self):
        # apply_boundary is a real, otherwise-untagged row in single_rank (same fixture
        # extract_CPU_hotspots.py's own equivalent test uses) -- configuring it as "other" drops
        # it from aggregate()'s cpu_entries, so it never becomes a hotspot or gets its own
        # caller-chain section, proving --extra-noise-config reaches this tool's ranking too.
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.txt")
            config_path = os.path.join(tmp, "noise_config.json")
            with open(config_path, "w") as f:
                json.dump({"add": {"other": ["apply_boundary"]}}, f)
            hc_tool.main([SINGLE_RANK_DIR, "-o", dest, "--all", "--extra-noise-config", config_path])
            with open(dest) as f:
                report = f.read()
        self.assertNotIn("apply_boundary", report)
        self.assertIn("compute_stencil", report)


if __name__ == "__main__":
    unittest.main()
