"""Stage 5 column spec for the GPU kernel hotspots table.

Scope: just the column layout for stage4_rocprofv3.aggregate()'s kernel entries -- total device
time, call count, and per-call average. No self-vs-inclusive split, unlike the CPU table: a GPU
kernel is already a leaf event. Ranking/filtering/rendering themselves are generic (see
stage5_table_render.py); this file only decides what the table looks like.
"""

GPU_HOTSPOTS_COLUMNS = [
    {"header": "#", "width": 3, "value": lambda e, i: str(i)},
    {"header": "total(s)", "width": 12, "value": lambda e, i: f"{e['sum']:.6f}"},
    {"header": "%total", "width": 7,
     "value": lambda e, i: f"{e['pct_total']:.1f}" if e["pct_total"] is not None else "n/a"},
    {"header": "calls", "width": 10, "value": lambda e, i: str(e["count"])},
    {"header": "avg(us)", "width": 10,
     "value": lambda e, i: f"{e['avg_us']:.2f}" if e["avg_us"] is not None else "n/a"},
    {"header": "kernel", "width": None, "value": lambda e, i: e["label"]},
]
