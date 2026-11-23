"""H03: reader validity on generated frames — the pre-registered analysis.

Reads the 150-frame manifest, the reader / Perspective Fields rows (Modal) and the classical
rows, and applies the rule in ``configs/h03_reader_validity_prereg.yaml`` (amended 1).

Primary metric: roll error, degrees, wrapped to [0, 90].  E_control is the reader's median
roll error over the 60 controls; a generated source passes if, for every independent estimator
that reads at least 70 % of its frames, the median reader-versus-estimator disagreement is at
most 1.5 E_control.

    PYTHONPATH=. pixi run -e gpu python experiments/h03_analyze.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/h03"
PF_ROLL_SIGN = -1.0          # from the rotation probe: d(pf_roll)/d(image rotation) = +0.95
BAR = 1.5
MIN_COVERAGE = 0.70


def wrap90(x):
    x = np.asarray(x, dtype=float)
    return (x + 90.0) % 180.0 - 90.0


def roll_of(ux, uy):
    return np.degrees(np.arctan2(ux, -uy))


def boot_median(v, n=5000, seed=1):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    med = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def load_rows(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["frame_id"]] = r
    return out


def main() -> int:
    man = list(csv.DictReader((R / "frames_manifest.csv").open()))
    readers = load_rows(R / "readers/depicted_main.jsonl")
    classical = load_rows(R / "classical.jsonl")
    rows = []
    for m in man:
        fid = m["frame_id"]
        rd = readers.get(fid, {})
        cl = classical.get(fid, {})
        up = [float(m[k]) if m[k] else np.nan for k in ("up_cam_x", "up_cam_y", "up_cam_z")]
        r = {"frame_id": fid, "source": m["source"], "clip_id": m["clip_id"],
             "frame_index": int(m["frame_index"]), "scene": m["scene"],
             "truth_roll": (roll_of(up[0], up[1]) if np.isfinite(up[0]) else np.nan),
             "truth_available": bool(np.isfinite(up[0])),
             "reader_roll": rd.get("reader_roll_deg", np.nan),
             "pf_roll": (PF_ROLL_SIGN * rd["pf_roll_deg"]
                         if "pf_roll_deg" in rd else np.nan),
             "pf_field_roll": (PF_ROLL_SIGN * rd["pf_field_roll_deg"]
                               if "pf_field_roll_deg" in rd else np.nan),
             "pf_pitch": rd.get("pf_pitch_deg", np.nan),
             "pf_vfov": rd.get("pf_vfov_deg", np.nan),
             "classical_roll": (cl.get("roll_deg") if cl.get("roll_deg") is not None
                                else np.nan),
             "classical_method": cl.get("method"),
             "reader_fisheye_roll": rd.get("reader_fisheye_roll_deg", np.nan)}
        if r["truth_available"]:
            r["reader_err"] = abs(wrap90(r["reader_roll"] - r["truth_roll"]))
            r["pf_err"] = abs(wrap90(r["pf_roll"] - r["truth_roll"]))
            r["classical_err"] = abs(wrap90(r["classical_roll"] - r["truth_roll"]))
        rows.append(r)

    def sel(pred):
        return [r for r in rows if pred(r)]

    def summ(vals):
        v = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        med, lo, hi = boot_median(v)
        return {"n": int(len(v)), "median": med, "ci": [lo, hi],
                "iqr": ([float(np.percentile(v, 25)), float(np.percentile(v, 75))]
                        if len(v) else [float("nan")] * 2)}

    controls = sel(lambda r: r["source"] in ("sim_control", "real_control"))
    E_control = summ([r.get("reader_err", np.nan) for r in controls])
    E_sim = summ([r["reader_err"] for r in sel(lambda r: r["source"] == "sim_control")])
    E_real = summ([r["reader_err"] for r in sel(lambda r: r["source"] == "real_control")])
    # same real controls under the reader's native fisheye configuration (context)
    E_real_fisheye = summ([abs(wrap90(r["reader_fisheye_roll"] - r["truth_roll"]))
                           for r in sel(lambda r: r["source"] == "real_control")])
    bar = BAR * E_control["median"]
    E_indep = {
        "pf": summ([r["pf_err"] for r in controls]),
        "classical": summ([r["classical_err"] for r in controls]),
        "pf_sim": summ([r["pf_err"] for r in sel(lambda r: r["source"] == "sim_control")]),
        "pf_real": summ([r["pf_err"] for r in sel(lambda r: r["source"] == "real_control")]),
        "classical_sim": summ([r["classical_err"] for r in sel(lambda r: r["source"] == "sim_control")]),
        "classical_real": summ([r["classical_err"] for r in sel(lambda r: r["source"] == "real_control")]),
        "coverage_sim": {x: float(np.mean([np.isfinite(r[x + "_roll"]) for r in sel(lambda r: r["source"] == "sim_control")]))
                         for x in ("pf", "classical")},
        "coverage_real": {x: float(np.mean([np.isfinite(r[x + "_roll"]) for r in sel(lambda r: r["source"] == "real_control")]))
                          for x in ("pf", "classical")},
    }

    sources = ["anwm", "lingbot_nf4", "lingbot_bf16", "solarwm"]
    per_source = {}
    for s in sources:
        sub = sel(lambda r, s=s: r["source"] == s)
        cov = {}
        for x, key in (("classical", "classical_roll"), ("pf", "pf_roll")):
            cov[x] = float(np.mean([np.isfinite(r[key]) for r in sub])) if sub else 0.0
        dis = {"classical": summ([abs(wrap90(r["reader_roll"] - r["classical_roll"]))
                                  for r in sub]),
               "pf": summ([abs(wrap90(r["reader_roll"] - r["pf_roll"])) for r in sub])}
        pair = summ([abs(wrap90(r["classical_roll"] - r["pf_roll"])) for r in sub])
        tested = {x: dis[x] for x in dis if cov[x] >= MIN_COVERAGE}
        status = "not_assessable"
        if tested:
            ok = all(dis[x]["median"] <= bar for x in tested)
            status = "pass" if ok else "fail"
        per_source[s] = {"n": len(sub), "coverage": cov, "disagreement": dis,
                         "independent_pair_agreement": pair, "tested": list(tested),
                         "status": status,
                         "reader_roll_summary": summ([r["reader_roll"] for r in sub]),
                         "classical_roll_summary": summ([r["classical_roll"] for r in sub]),
                         "pf_roll_summary": summ([r["pf_roll"] for r in sub])}
    overall = ("valid_on_generated_frames"
               if all(v["status"] == "pass" for v in per_source.values())
               else "not_valid_on_generated_frames")
    # per-clip detail for the LingBot eye-versus-reader question
    lingbot = {}
    for s in ("lingbot_nf4", "lingbot_bf16"):
        for clip in sorted({r["clip_id"] for r in sel(lambda r, s=s: r["source"] == s)}):
            sub = sel(lambda r, s=s, c=clip: r["source"] == s and r["clip_id"] == c)
            lingbot[f"{s}/{clip}"] = {
                "reader": summ([r["reader_roll"] for r in sub]),
                "pf": summ([r["pf_roll"] for r in sub]),
                "classical": summ([r["classical_roll"] for r in sub]),
                "classical_methods": sorted({r["classical_method"] for r in sub
                                             if r["classical_method"]})}

    out = {"E_control_deg": E_control, "E_independent_estimators_deg": E_indep, "E_sim_deg": E_sim, "E_real_deg": E_real,
           "E_real_native_fisheye_deg": E_real_fisheye,
           "bar_deg": bar, "bar_rule": f"{BAR} x E_control",
           "per_source": per_source, "overall": overall,
           "lingbot_by_clip": lingbot,
           "n_frames": len(rows),
           "reader_missing": int(sum(not np.isfinite(r["reader_roll"]) for r in rows))}
    (R / "analysis.json").write_text(json.dumps(out, indent=1))
    fields = ["frame_id", "source", "clip_id", "frame_index", "scene", "reader_roll",
              "pf_roll", "pf_field_roll", "pf_pitch", "pf_vfov", "classical_roll",
              "classical_method", "reader_fisheye_roll", "truth_roll", "truth_available",
              "reader_err", "pf_err", "classical_err"]
    with (R / "frames_scored.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: out[k] for k in ("E_control_deg", "E_sim_deg", "E_real_deg",
                                          "E_real_native_fisheye_deg", "bar_deg",
                                          "overall")}, indent=1))
    print(json.dumps(E_indep, indent=1))
    for s, v in per_source.items():
        print(f"{s:13s} status={v['status']:13s} cov={v['coverage']} "
              f"class={v['disagreement']['classical']['median']:.2f} "
              f"pf={v['disagreement']['pf']['median']:.2f} "
              f"pair={v['independent_pair_agreement']['median']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
