"""Regenerate the active figures from stored data only (no simulation is run here).

    python -X utf8 -m research.figures.make_figures
Inputs: results_summary.json, experiments/results/h13_sim_vs_formula.json, experiments/results/measured_gpu_tesla_t4.json
Outputs (SVG, no dependencies): research/figures/{controller_regret,h13_breakeven,t4_roofline_ratio}.svg
Each figure states its evidence class in its title. Nothing here reads the archive.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "figures")
INK, MUT, GRID = "#111827", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#6b7280"


def load(rel: str):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def svg(w, h, title, body, sub=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{title}" font-family="Arial,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="#fff"/><text x="16" y="24" font-size="15" font-weight="700" fill="{INK}">{title}</text>'
            f'<text x="16" y="42" font-size="11" fill="{MUT}">{sub}</text>{body}</svg>')


def t975(n):
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 20: 2.093, 60: 2.0}.get(n, 2.0 if n > 20 else 2.5)


def controller_regret():
    d = load("results_summary.json")
    by = defaultdict(list)
    for r in d["summary"]:
        by[r["controller"]].append(r["mean_regret"])
    rows = sorted(((c, sum(v) / len(v), t975(len(v)) * math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))) for c, v in by.items()), key=lambda r: r[1])
    top = max(m + ci for _, m, ci in rows) * 1.1
    body = []
    for i, (c, m, ci) in enumerate(rows):
        y = 70 + i * 34
        w, cw = m / top * 420, ci / top * 420
        col = GRAY if c == "Oracle" else (ORANGE if c == "Predictive" else BLUE)
        body += [f'<text x="190" y="{y + 14}" text-anchor="end" font-size="12" fill="{INK}">{c}</text>',
                 f'<rect x="200" y="{y}" width="{max(1, w):.1f}" height="20" rx="3" fill="{col}"/>',
                 f'<line x1="{200 + w - cw:.1f}" x2="{200 + w + cw:.1f}" y1="{y + 10}" y2="{y + 10}" stroke="{INK}" stroke-width="1.5"/>',
                 f'<text x="{200 + w + cw + 6:.1f}" y="{y + 14}" font-size="12" fill="{INK}">{m:.3f} ms</text>']
    meta = d["metadata"]
    return svg(700, 80 + 34 * len(rows), "Mean regret by controller (SIMULATED)", "".join(body),
               f'{meta["experiment_id"]} · {meta["runs"]} runs, 5 seeds · mean ± 95% CI · regret vs min-cost placement · lower is better')


def h13_breakeven():
    rows = [r for r in load("experiments/results/h13_sim_vs_formula.json")["breakeven"] if r["gamma"] == 0.0 and r["lead"] == 1.5]
    rows.sort(key=lambda r: r["u"])
    x0, y0, W, H = 70, 70, 560, 280
    def px(u): return x0 + u / max(r["u"] for r in rows) * W
    def py(p): return y0 + H - (p if p is not None else 1.0) * H
    body = [f'<line x1="{x0}" y1="{y0 + H}" x2="{x0 + W}" y2="{y0 + H}" stroke="{INK}"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + H}" stroke="{INK}"/>']
    for g in (0, .25, .5, .75, 1):
        body.append(f'<line x1="{x0}" y1="{py(g):.1f}" x2="{x0 + W}" y2="{py(g):.1f}" stroke="{GRID}"/><text x="{x0 - 8}" y="{py(g) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{g}</text>')
    for key, col, lab, dash in (("formula", GRAY, "closed form (u+γ)/(1+u)", "5 4"), ("simulated", BLUE, "queueing simulation", "")):
        pts = " ".join(f"{px(r['u']):.1f},{py(r[key]):.1f}" for r in rows if r[key] is not None)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        r = [r for r in rows if r[key] is not None][-1]
        body.append(f'<text x="{x0 + 14}" y="{y0 + 16 + (0 if key == "simulated" else 18)}" font-size="12" fill="{col}">{"━" if key == "simulated" else "╍"} {lab}</text>')
    for r in rows:
        body.append(f'<text x="{px(r["u"]):.1f}" y="{y0 + H + 16}" text-anchor="middle" font-size="11" fill="{MUT}">{r["u"]:g}</text>')
    body.append(f'<text x="{x0 + W / 2}" y="{y0 + H + 36}" text-anchor="middle" font-size="12" fill="{INK}">background link utilisation u</text>')
    body.append(f'<text x="18" y="{y0 + H / 2}" font-size="12" fill="{INK}" transform="rotate(-90 18 {y0 + H / 2})" text-anchor="middle">break-even predictor accuracy p*</text>')
    return svg(700, 420, "H13 break-even accuracy: formula vs simulation (SIMULATED)", "".join(body), "gamma = 0, prefetch lead 1.5 transfers · EXP-H13-FORM-001 · the formula holds at low load and breaks as u grows")


def t4_ratio():
    res = load("experiments/results/measured_gpu_tesla_t4.json")["sections"]["gemm_family"]["result"]
    rows = res["rows"]
    x0, y0, W, H = 70, 70, 560, 260
    lo, hi = 0.0, max(r["ratio"] for r in rows) * 1.15
    xs = [math.log2(r["batch"]) for r in rows]
    def px(b): return x0 + (math.log2(b) - min(xs)) / (max(xs) - min(xs)) * W
    def py(v): return y0 + H - (v - lo) / (hi - lo) * H
    body = [f'<line x1="{x0}" y1="{y0 + H}" x2="{x0 + W}" y2="{y0 + H}" stroke="{INK}"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + H}" stroke="{INK}"/>',
            f'<line x1="{x0}" y1="{py(1):.1f}" x2="{x0 + W}" y2="{py(1):.1f}" stroke="{GRAY}" stroke-dasharray="5 4"/><text x="{x0 + 8}" y="{py(1) + 16:.1f}" font-size="11" fill="{GRAY}">- - ratio 1 = roofline exact</text>']
    pts = " ".join(f"{px(r['batch']):.1f},{py(r['ratio']):.1f}" for r in rows)
    body.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for r in rows:
        body.append(f'<circle cx="{px(r["batch"]):.1f}" cy="{py(r["ratio"]):.1f}" r="4" fill="{BLUE}"/><text x="{px(r["batch"]):.1f}" y="{y0 + H + 16}" text-anchor="middle" font-size="11" fill="{MUT}">{r["batch"]}</text>')
    for v in (0, 0.5, 1.0, 1.5, 2.0):
        if v <= hi:
            body.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{v}</text>')
    body.append(f'<text x="{x0 + W / 2}" y="{y0 + H + 36}" text-anchor="middle" font-size="12" fill="{INK}">batch size (log scale)</text>')
    body.append(f'<text x="18" y="{y0 + H / 2}" font-size="12" fill="{INK}" transform="rotate(-90 18 {y0 + H / 2})" text-anchor="middle">measured time / predicted time</text>')
    return svg(700, 400, "Roofline prediction vs measurement, Tesla T4 (PRELIMINARY HARDWARE, one run)", "".join(body),
               f'EXP-HW-T4-001 · fp16 W@X, {res["m"]}x{res["k"]} · predicted balance batch {res["predicted_balance_batch"]:.0f}, measured 50%-of-peak batch {res["measured_batch_at_half_peak"]} · a T4 is not an H100')


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("controller_regret", controller_regret), ("h13_breakeven", h13_breakeven), ("t4_roofline_ratio", t4_ratio)):
        with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(fn())
        print("wrote", name + ".svg")


if __name__ == "__main__":
    main()
