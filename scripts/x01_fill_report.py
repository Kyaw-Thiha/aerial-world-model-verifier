#!/usr/bin/env python3
"""Fill the machine-copied numbers in reports/X1_generator_breadth.md from results/x01/.

Every table row is read from ``results/x01/score.json`` (produced by ``experiments/x01_score.py``)
and from the per-clip run manifests in ``results/x01/runs/``; nothing is typed by hand.  The
script rewrites the marked blocks in place and is idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "results/x01/score.json"
REPORT = ROOT / "reports/X1_generator_breadth.md"

BEGIN = "<!-- AUTO:BEGIN -->"
END = "<!-- AUTO:END -->"


def fmt(x, nd=2):
    if x is None:
        return "—"
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def cost_table() -> list[str]:
    """Per-generator cost per clip from the run manifests on the volume (pulled small JSONs)."""
    prices = {"A10": 1.10, "A10G": 1.10, "A100-80GB": 2.50}
    rows = {}
    for d in sorted((ROOT / "results/x01/runs").rglob("manifest.json")):
        try:
            m = json.loads(d.read_text())
        except Exception:  # noqa: BLE001
            continue
        secs = m.get("seconds")
        gpu = m.get("gpu", "")
        if secs is None:
            continue
        price = next((v for k, v in prices.items() if k in gpu), None)
        if price is None:
            continue
        rows.setdefault(str(d.parent.parent.name), []).append((secs, price * secs / 3600))
    out = []
    for k, v in rows.items():
        med = sorted(x[1] for x in v)[len(v) // 2]
        out.append(f"| {k} | {len(v)} | {med:.2f} |")
    return out


def main() -> int:
    s = json.loads(SCORE.read_text())
    models = sorted(s["models"])
    lines = ["| model | clips | testable | in-envelope | median gain | 95 % CI | paths | gain in "
             "[0.5, 1.5] | null | median ratio | R² |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    order = []
    for m in models:
        r = s["models"][m]
        tf = r["tilt_fidelity"]
        ci = ("descriptive" if tf.get("descriptive") or tf.get("lo") is None
              else f"{tf['lo']:+.2f} to {tf['hi']:+.2f}")
        lb = r["frac_in_band"]
        lines.append(f"| {m} | {r['n_clips_produced']} | {r['n_clips_testable']} | "
                     f"{r['n_clips_in_envelope']} | {fmt(tf['median'])} | {ci} | "
                     f"{tf['n_clusters']} | {lb['k']}/{lb['n']} | "
                     f"{fmt(r['null_gain']['median'])} | {fmt(tf['median_ratio'])} | "
                     f"{fmt(tf['median_r2'])} |")
        order.append((m, tf["median"]))
    cl = ["| model | path class | paths | median gain | 95 % CI |",
          "| --- | --- | --- | --- | --- |"]
    for m in models:
        for fam, v in sorted(s["models"][m]["by_path_class"].items()):
            ci = ("descriptive" if v.get("descriptive") or v.get("lo") is None
                  else f"{v['lo']:+.2f} to {v['hi']:+.2f}")
            cl.append(f"| {m} | {fam} | {v['n_paths']} | {fmt(v['median'])} | {ci} |")
    pw = ["| pair | shared paths | median difference | 95 % CI |", "| --- | --- | --- | --- |"]
    for k, v in s["pairwise"].items():
        pw.append(f"| {k} | {v['n_paths']} | {v['median_diff']:+.2f} | "
                  f"{v['lo']:+.2f} to {v['hi']:+.2f} |")
    cost = cost_table()
    block = "\n".join([
        "### Tilt fidelity (new, this stage)", "",
        *lines, "",
        "Cluster bootstrap over paths, 5,000 replicates, `default_rng(1)`; fewer than 10 clusters "
        "is labelled descriptive.  The null pairs one clip's depicted lean with another path's "
        "required lean.", "",
        "### By path class", "", *cl, "",
        "### Pairwise differences on shared in-envelope paths", "", *pw, "",
        "### Cost per clip (measured, GPU list price)", "",
        "| generator | clips with a manifest | median USD per clip |", "| --- | --- | --- |",
        *cost, "",
    ])
    txt = REPORT.read_text()
    if BEGIN in txt and END in txt:
        head, rest = txt.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        txt = head + BEGIN + "\n" + block + END + tail
    else:
        txt = txt.rstrip() + "\n\n" + BEGIN + "\n" + block + END + "\n"
    REPORT.write_text(txt)
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
