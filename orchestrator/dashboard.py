"""Comparison dashboard: one self-contained HTML file (inline SVG, no scripts, no network) generated from the benchmark JSON.

Colours are the Okabe-Ito colour-blind-safe set; every bar is also labelled with its policy name and value, so identity never depends on colour alone.
"""
from __future__ import annotations

import html
import os

COLORS = {"lru_demand": "#6b7280", "always_prefetch": "#E69F00", "static_threshold": "#56B4E9", "adaptive_gate": "#0072B2", "adaptive_gate_v2": "#F0E442", "adaptive": "#009E73", "adaptive_v2": "#D55E00", "oracle_prefetch": "#CC79A7"}
NOTE = {"oracle_prefetch": "upper bound (clairvoyant)", "lru_demand": "baseline"}


def _cell_svg(cell: dict) -> str:
    pols = list(cell["policies"])
    vals = [(p, cell["policies"][p]["p99_ms"]["mean"], cell["policies"][p]["p99_ms"]["ci95"]) for p in pols]
    top = max(v + (c if c == c else 0) for _, v, c in vals) * 1.08
    row, left, width = 26, 150, 420
    h = row * len(vals) + 22
    out = [f'<svg viewBox="0 0 {left + width + 90} {h}" role="img" aria-label="P99 latency by policy, {html.escape(cell["scenario"])}, {html.escape(cell["background"])}">']
    for i, (p, v, c) in enumerate(vals):
        y = 8 + i * row
        w = v / top * width
        ci = (c if c == c else 0) / top * width
        dash = ' stroke-dasharray="4 3" fill-opacity="0.45"' if p == "oracle_prefetch" else ""
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" class="lab">{p}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{max(1.0, w):.1f}" height="16" rx="3" fill="{COLORS[p]}" stroke="{COLORS[p]}"{dash}/>')
        out.append(f'<line x1="{left + w - ci:.1f}" x2="{left + w + ci:.1f}" y1="{y + 8}" y2="{y + 8}" class="ci"/>')
        out.append(f'<text x="{left + w + ci + 6:.1f}" y="{y + 13}" class="val">{v:.1f} ms</text>')
    out.append("</svg>")
    return "".join(out)


def _table(cell: dict) -> str:
    rows = []
    for p, e in cell["policies"].items():
        d = e.get("paired_vs_lru_demand_p99_ms")
        dm = e.get("paired_vs_lru_demand_mean_latency_ms")
        fmt = lambda x: "—" if not x else f'{x["pct"]:+.1f}%{" ✱" if x["significant"] else ""}'  # noqa: E731
        rows.append(f'<tr><th scope="row">{p}<small> {NOTE.get(p, "")}</small></th><td>{e["mean_latency_ms"]["mean"]:.1f} ± {e["mean_latency_ms"]["ci95"]:.1f}</td>'
                    f'<td>{e["p99_ms"]["mean"]:.1f} ± {e["p99_ms"]["ci95"]:.1f}</td><td>{fmt(dm)}</td><td>{fmt(d)}</td><td>{e["hbm_block_hit_rate"]["mean"]:.2f}</td>'
                    f'<td>{e["wasted_prefetch_gib"]["mean"]:.1f}</td></tr>')
    return ('<table><thead><tr><th></th><th>mean ms ±CI</th><th>P99 ms ±CI</th><th>mean vs baseline</th><th>P99 vs baseline</th><th>HBM hit</th><th>wasted prefetch GiB</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def write_dashboard(payload: dict, path: str) -> None:
    m = payload["metadata"]
    sections = []
    for c in payload["cells"]:
        sections.append(f'<section><h2>{html.escape(c["scenario"])} <small>· {html.escape(c["background"])}</small></h2>'
                        f'<p class="desc">{html.escape(m["scenarios"][c["scenario"]])}</p>{_cell_svg(c)}{_table(c)}</section>')
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive memory orchestrator: comparison dashboard</title>
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#4b5563;--line:#d1d5db;--warn:#fef3c7;--warnfg:#78350f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1020;--fg:#e5e7eb;--mut:#9ca3af;--line:#374151;--warn:#3b2f0b;--warnfg:#fde68a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:20px 16px 60px}} h1{{font-size:1.5rem;margin:.2em 0}} h2{{font-size:1.1rem;margin:1.8em 0 .2em}} h2 small{{color:var(--mut);font-weight:400}}
.banner{{background:var(--warn);color:var(--warnfg);border-radius:8px;padding:.7em 1em;margin:1em 0}} .desc{{color:var(--mut);margin:.2em 0 .6em}}
svg{{width:100%;height:auto;max-width:700px}} .lab{{font-size:12px;fill:currentColor}} .val{{font-size:12px;fill:currentColor}} .ci{{stroke:currentColor;stroke-width:1.5}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:.6em}} th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}} th:first-child,th[scope=row]{{text-align:left}} thead th{{color:var(--mut);font-weight:600}}
small{{color:var(--mut)}} .wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Adaptive memory orchestrator: policy comparison</h1>
<div class="banner"><b>Simulated, not measured.</b> Tier parameters are assumptions (see the assumptions table in the JSON). CXL is a model only: no CXL hardware was used. A working software prototype is not evidence of physical performance.
<code>oracle_prefetch</code> is a clairvoyant upper bound, not an implementable policy.</div>
<p>{len(m["seeds"])} seeds per cell · mean ± 95% CI (t) · ✱ = paired difference vs <code>{m["baseline"]}</code> has a 95% CI that excludes 0 (no multiple-comparison correction) · lower is better for latency.
Generated {m["generated_utc"]} from repo commit {m["repo_head"]}. Regenerate: <code>{html.escape(m["regenerate"])}</code>.</p>
<div class="wrap">{"".join(sections)}</div>
</main></body></html>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
