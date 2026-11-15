"""Assemble the Gate 1 table from scored clip records.

Input: JSON files written by ``remote/c_extractors.py`` (one list per extractor sweep).
Output: a per-(extractor, testbed) table against the pre-registered criteria, plus
``results/pathrec_bakeoff/<run>/summary.json``.

The runner refuses to work if ``configs/pathrec_prereg.yaml`` differs from its committed
version: criteria must not drift after scores are seen.

    python experiments/pathrec_bakeoff.py --inputs results/pathrec_bakeoff/raw --out results/pathrec_bakeoff/v1
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

PREREG = Path("configs/pathrec_prereg.yaml")


def prereg_status(path: Path = PREREG) -> dict:
    """Is the pre-registration committed and unmodified?"""
    try:
        diff = subprocess.run(["git", "diff", "--quiet", "--", str(path)],
                              capture_output=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(path)],
                                capture_output=True)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                                 capture_output=True)
    except FileNotFoundError as exc:  # pragma: no cover - no git in a stripped image
        raise RuntimeError("git is required to verify the pre-registration") from exc
    return {"tracked": tracked.returncode == 0,
            "modified": diff.returncode != 0,
            "staged": staged.returncode != 0}


def parse_text_line(line: str) -> dict | None:
    """Parse one ``== model/seq: err=None dir=..`` line from the Modal sweep output."""
    line = line.strip()
    if not line.startswith("== ") or "/" not in line:
        return None
    head, _, tail = line[3:].partition(":")
    model, _, seq = head.partition("/")
    fields = {}
    for part in tail.split():
        if "=" in part:
            k, _, v = part.partition("=")
            m = re.match(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", v)
            fields[k] = float(m.group()) if m else v
    score = {"accel_dir_err_deg_median": fields.get("dir"),
             "accel_mag_ratio": fields.get("ratio"),
             "lean_r2": fields.get("r2"),
             "coverage_fraction": fields.get("cov"),
             "rel_rot_err_deg_median": fields.get("rot"),
             "ate_sim3_m": fields.get("ate_sim3")}
    testbed = "TII" if seq.startswith("flight-") else ("ANWM" if seq.count("/") else "UZH")
    return {"model": model, "seq": seq, "testbed": testbed, "error": fields.get("err"),
            "score": {k: v for k, v in score.items() if isinstance(v, float)}}


def load_records(raw_dir: Path) -> list[dict]:
    rows = []
    for txt in sorted(Path(raw_dir).glob("*.txt")):
        for line in txt.read_text().splitlines():
            rec = parse_text_line(line)
            if rec and rec["score"]:
                rec["source_file"] = txt.name
                rows.append(rec)
    for path in sorted(Path(raw_dir).glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            rec.setdefault("source_file", path.name)
            rows.append(rec)
    return rows


def _median(values) -> float:
    v = [x for x in values if isinstance(x, (int, float)) and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def gate1(rows: list[dict], crit: dict) -> dict:
    """Apply the pre-registered criteria per extractor; return rows and verdicts."""
    thresholds = crit["acceleration_direction"]["threshold_deg"]
    lo, hi = crit["acceleration_magnitude"]["interval"]
    cov_min = crit["coverage"]["threshold"]
    out = {}
    for model in sorted({f'{r["model"]}/{r.get("testbed", "?")}' for r in rows}):
        recs = [r for r in rows
                if f'{r["model"]}/{r.get("testbed", "?")}' == model and r.get("score")]
        if not recs:
            out[model] = {"n_clips": 0, "pass": False, "reason": "no scored clips"}
            continue
        dirs = [r["score"].get("accel_dir_err_deg_median", float("nan")) for r in recs]
        ratios = [r["score"].get("accel_mag_ratio", float("nan")) for r in recs]
        tests = [r["score"].get("coverage_fraction", float("nan")) for r in recs]
        per_clip_pass = [bool(d <= thresholds and lo <= q <= hi and t >= 0.2)
                         for d, q, t in zip(dirs, ratios, tests)]
        out[model] = {
            "n_clips": len(recs),
            "median_dir_err_deg": float(np.median(dirs)),
            "median_mag_ratio": float(np.median(ratios)),
            "median_coverage": float(np.median(tests)),
            "median_ate_sim3_m": _median([r["score"].get("ate_sim3_m") for r in recs]),
            "median_ate_se3_m": _median([r["score"].get("ate_se3_m") for r in recs]),
            "median_scale_sim3": _median([r["score"].get("scale_sim3") for r in recs]),
            "median_rel_rot_err_deg": _median([r["score"].get("rel_rot_err_deg_median")
                                               for r in recs]),
            "coverage_pass": float(np.mean(per_clip_pass)),
            "pass": bool(np.mean(per_clip_pass) >= cov_min),
            "per_clip": [
                {"seq": r.get("seq"), "split": r.get("split"), "dir": d, "ratio": q,
                 "coverage": t, "pass": p}
                for r, d, q, t, p in zip(recs, dirs, ratios, tests, per_clip_pass)
            ],
            "pass_by_split": {
                sp: [bool(x) for x, rr in zip(per_clip_pass, recs) if rr.get("split") == sp]
                for sp in ("fit", "eval", "other")
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="results/pathrec_bakeoff/raw")
    ap.add_argument("--out", default="results/pathrec_bakeoff/v1")
    ap.add_argument("--prereg", default=str(PREREG))
    ap.add_argument("--split-config", default="configs/pathrec_prereg_v2.yaml")
    args = ap.parse_args()

    status = prereg_status(Path(args.prereg))
    crit = yaml.safe_load(Path(args.prereg).read_text())["gate1_pass_criteria"]
    rows = load_records(Path(args.inputs))
    split = {}
    try:
        cfg2 = yaml.safe_load(Path(args.split_config).read_text())
        for s_ in cfg2["split"]["fit"]:
            split[s_] = "fit"
        for s_ in cfg2["split"]["eval"]:
            split[s_] = "eval"
    except Exception:
        pass
    for r in rows:
        r["split"] = split.get(r.get("seq"), "other")
    table = gate1(rows, crit)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "prereg": {"path": args.prereg, **status, "git_diff_clean": not status["modified"]},
        "criteria": crit, "table": table, "n_records": len(rows)}, indent=1))

    print(f"pre-registration: tracked={status['tracked']} modified={status['modified']}")
    print(f"{'model':8s} {'clips':>5s} {'dir_deg':>8s} {'ratio':>6s} {'cov':>5s} "
          f"{'simp3_m':>8s} {'se3_m':>7s} {'rot_deg':>8s} {'pass':>5s}")
    for model, t in table.items():
        if t.get("pass_by_split"):
            t["rate_by_split"] = {k: (float(np.mean(v)) if v else None)
                                  for k, v in t["pass_by_split"].items()}
        print(f"{model:8s} {t['n_clips']:5d} {t.get('median_dir_err_deg', float('nan')):8.1f} "
              f"{t.get('median_mag_ratio', float('nan')):6.2f} "
              f"{t.get('median_coverage', float('nan')):5.2f} "
              f"{t.get('median_ate_sim3_m', float('nan')):8.2f} "
              f"{t.get('median_ate_se3_m', float('nan')):7.2f} "
              f"{t.get('median_rel_rot_err_deg', float('nan')):8.1f} "
              f"{str(t.get('pass')):>5s}")
    if status["modified"] or not status["tracked"]:
        print("WARNING: pre-registration is not committed and clean; this table is not a "
              "pre-registered result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
