"""Stage 5 column spec for the CPU hotspots table.

Scope: just the column layout for stage4_rocprofsys_sample_flat.aggregate()'s CPU-side entries -- self
time, inclusive time, call count, and self-vs-inclusive percentage. Ranking/filtering/rendering
themselves are generic (see stage5_table_render.py); this file only decides what the table looks
like.
"""

CPU_HOTSPOTS_COLUMNS = [
    {"header": "#", "width": 3, "value": lambda e, i: str(i)},
    {"header": "self(s)", "width": 12, "value": lambda e, i: f"{e['self_sum']:.6f}"},
    {"header": "%total", "width": 7,
     "value": lambda e, i: f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"},
    {"header": "total(s)", "width": 12, "value": lambda e, i: f"{e['sum']:.6f}"},
    {"header": "calls", "width": 10, "value": lambda e, i: str(e["count"])},
    {"header": "%self", "width": 7,
     "value": lambda e, i: f"{e['pct_self']:.1f}" if e.get("pct_self") is not None else "n/a"},
    {"header": "function", "width": None, "value": lambda e, i: e["label"]},
]
