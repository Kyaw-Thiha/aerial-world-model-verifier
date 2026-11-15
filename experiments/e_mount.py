"""Track E Task 3: check the published mount, and report depicted roll/pitch against truth.

Two questions, both per rig:

1. **Is the published mount the right reading?** Fit a constant camera rotation on the
   pre-registered fit sequences and evaluate it on the held-out sequences of the same rig.
   A misread mount (transposed, or applied in the wrong direction) cannot be repaired by any
   constant rotation and would leave that residual large; a correct reading leaves the
   estimator's own bias, a degree or two.
2. **How do depicted roll and pitch compare with truth, in degrees?** The verifier consumes a
   tilt; this is the same quantity in the form a flight controller would report, per sequence,
   with the residual's dependence on speed.

    python experiments/e_mount.py --perframe results/e_perframe/v1_forward --root data/uzh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.depicted.metrics import summarize
from src.depicted.mount import fit_mount_rotation, median_angle_deg, roll_pitch_from_up
from src.depicted.uzh import FIT_SEQUENCES, all_sequences, load_truth, rig_of


def up_body_from_est(tab: dict, R_cb: np.ndarray) -> np.ndarray:
    """Body-frame up implied by the estimate and the published mount: ``R_cb @ up_cam``."""
    return np.einsum("ij,nj->ni", np.asarray(R_cb, dtype=np.float64), tab["up_est"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perframe", default="results/e_perframe/v1_forward")
    ap.add_argument("--root", default="data/uzh")
    ap.add_argument("--out", default="results/e_mount")
    ap.add_argument("--variants", default="abc")
    args = ap.parse_args()
    perframe = Path(args.perframe)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"perframe": str(perframe), "variants": {}}
    for variant in args.variants:
        per_seq = {}
        for seq in all_sequences():
            p = perframe / f"frames_{seq}__{variant}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            truth = load_truth(root, seq)
            R_cb = d["R_cb"] if "R_cb" in d.files else truth.R_cb()
            up_b = up_body_from_est({"up_est": d["up_est"]}, R_cb)
            up_t = truth.up_body
            roll_e, pitch_e = roll_pitch_from_up(up_b)
            roll_t, pitch_t = roll_pitch_from_up(up_t)
            per_seq[seq] = {
                "rig": rig_of(seq), "env": truth.env, "n": int(len(up_b)),
                "fit_sequence": seq in FIT_SEQUENCES,
                "err_body_deg": summarize(np.degrees(np.arccos(np.clip(
                    np.einsum("ij,ij->i", up_b / np.linalg.norm(up_b, axis=1, keepdims=True),
                              up_t), -1, 1)))),
                "roll_err_deg": summarize(np.degrees(np.abs(
                    np.angle(np.exp(1j * (roll_e - roll_t)))))),
                "pitch_err_deg": summarize(np.degrees(np.abs(pitch_e - pitch_t))),
                "speed_p95_mps": float(np.percentile(d["speed"], 95)),
                "speed_vs_err_spearman": None,
                "_up_b": up_b, "_up_t": up_t, "_speed": d["speed"],
            }
            rho = _spearman(np.abs(np.degrees(np.arccos(np.clip(
                np.einsum("ij,ij->i", up_b / np.linalg.norm(up_b, axis=1, keepdims=True), up_t),
                -1, 1)))), d["speed"])
            per_seq[seq]["speed_vs_err_spearman"] = rho

        # per-rig mount check: fit on fit sequences, evaluate on held-out
        rigs: dict[str, list[str]] = {}
        for seq, rec in per_seq.items():
            rigs.setdefault(rec["rig"], []).append(seq)
        rig_out = {}
        for rig, seqs in sorted(rigs.items()):
            fit_seqs = [s for s in seqs if s in FIT_SEQUENCES]
            eval_seqs = [s for s in seqs if s not in FIT_SEQUENCES]
            if not fit_seqs:
                rig_out[rig] = {"note": "no fit sequence for this rig in the prepared set"}
                continue
            est = np.concatenate([per_seq[s]["_up_b"] for s in fit_seqs])
            tru = np.concatenate([per_seq[s]["_up_t"] for s in fit_seqs])
            fit = fit_mount_rotation(est, tru)
            rec = {"fit_sequences": fit_seqs, "eval_sequences": eval_seqs,
                   "fit": fit, "rotation_angle_deg": fit["rotation_angle_deg"]}
            D = np.asarray(fit["rotation"], dtype=np.float64)
            if eval_seqs:
                e = np.concatenate([per_seq[s]["_up_b"] for s in eval_seqs])
                t = np.concatenate([per_seq[s]["_up_t"] for s in eval_seqs])
                rec["heldout_residual_published_deg"] = median_angle_deg(e, t)
                rec["heldout_residual_fitted_deg"] = median_angle_deg(e @ D.T, t)
            rig_out[rig] = rec

        for seq in per_seq:
            for k in ("_up_b", "_up_t", "_speed"):
                per_seq[seq].pop(k, None)
        summary["variants"][variant] = {"per_sequence": per_seq, "per_rig": rig_out}
        print(f"variant {variant}:")
        for rig, rec in rig_out.items():
            if "fit" in rec:
                print(f"  {rig:16s} fitted rotation {rec['rotation_angle_deg']:.2f} deg, "
                      f"residual {rec['fit']['residual_before_deg']:.2f} -> "
                      f"{rec['fit']['residual_after_deg']:.2f} (fit), held-out "
                      f"{rec.get('heldout_residual_published_deg')} -> "
                      f"{rec.get('heldout_residual_fitted_deg')}")
            else:
                print(f"  {rig:16s} {rec}")

    (out / "mount.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'mount.json'}")
    return 0


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
