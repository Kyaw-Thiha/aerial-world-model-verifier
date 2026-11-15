"""Stage 0: validate the attitude-injection conventions before spending GPU time.

No generation here. Runs on one released trajectory (depth, pose, K, frames) and the 15 Regime C
designs. Writes results/anwm_injected/stage0.json plus contact sheets under
results/anwm_injected/stage0/.

    python -m experiments.anwm_stage0_conventions
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "anwm"))

from anwm.projection import (project_to_2d_image_seq2,  # noqa: E402
                             reproject_depth_to_other_pose,
                             reproject_depth_to_other_pose_seq2seq,
                             project_to_2d_image_seq2seq)
from src.generation.attitude import (body_z_from_poses, body_rotation,  # noqa: E402
                                     pose_with_attitude, release_euler_triple,
                                     roll_pitch_from_thrust, rotation_homography,
                                     thrust_from_accel)
from src.generation.paths import PathSpec, design  # noqa: E402
from src.generation.poses import camera_axis_change, pose_from_position_yaw  # noqa: E402

TRAJ = "3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed"
DATA = ROOT / "data/raw/anwm_full" / TRAJ
OUT = ROOT / "results/anwm_injected"
CTX = 16
DESIGNS = [("cruise_ctl", "cruise", None, "course"),
           ("turn_5_course", "turn", 5.0, "course"),
           ("turn_5_fixed", "turn", 5.0, "fixed"),
           ("turn_10_fixed", "turn", 10.0, "fixed"),
           ("turn_20_fixed", "turn", 20.0, "fixed"),
           ("slalom_5_course", "slalom", 5.0, "course"),
           ("slalom_5_fixed", "slalom", 5.0, "fixed"),
           ("slalom_10_fixed", "slalom", 10.0, "fixed"),
           ("accel_5", "accel_brake", 5.0, "course"),
           ("accel_10", "accel_brake", 10.0, "course"),
           ("accel_20", "accel_brake", 20.0, "course"),
           ("climb_5_course", "climb_turn", 5.0, "course"),
           ("climb_5_fixed", "climb_turn", 5.0, "fixed"),
           ("climb_10_fixed", "climb_turn", 10.0, "fixed"),
           ("climb_20_fixed", "climb_turn", 20.0, "fixed")]


def load() -> dict:
    t = pickle.load(open(DATA / "traj_data.pkl", "rb"))
    return {k: np.asarray(v) for k, v in t.items()}


def rgb(traj: dict, i: int) -> np.ndarray:
    im = cv2.imread(str(DATA / f"{int(i)}.jpg"))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def place(spec: PathSpec, pos0, yaw0):
    """Design frame -> world at a native start pose; acceleration rotates with the heading."""
    c, s = np.cos(yaw0), np.sin(yaw0)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(pos0) + spec.pos @ Rz.T, spec.yaw + yaw0, spec.accel @ Rz.T)


def ffp(traj, ctx_times, rgbs, poses_dst):
    depth_seq = traj["depth"][ctx_times]
    H, W = depth_seq.shape[-2:]
    pts, cols = reproject_depth_to_other_pose_seq2seq(
        K=traj["K"], depth_maps=depth_seq, rgb_imgs=rgbs,
        poses_src=traj["pose"][ctx_times], poses_dst=np.asarray(poses_dst))
    return project_to_2d_image_seq2seq(K=traj["K"], points_3d=pts, colors=cols,
                                       image_size=(H, W))


# ------------------------------------------------------------------ test 1: pure rotation
def test_pure_rotation(traj) -> dict:
    rng = np.random.default_rng(0)
    H, W = 64, 64
    # a small principal point for the small synthetic image; using the real 512x512 K here would
    # put every point outside the frame (the first version of this test did exactly that)
    K = np.array([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]])
    depth = rng.uniform(2.0, 20.0, size=(H, W))
    img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)

    # a 12 deg rotation about the camera x axis, and a 10 deg about camera y, and a mix
    from scipy.spatial.transform import Rotation

    out = {}
    for name, R_rel in (("roll_x", Rotation.from_euler("x", 12, degrees=True).as_matrix()),
                        ("pitch_y", Rotation.from_euler("y", -10, degrees=True).as_matrix()),
                        ("mix", Rotation.from_euler("xyz", [8, -6, 3], degrees=True).as_matrix())):
        pose_src = np.eye(4)
        pose_dst = np.eye(4)
        pose_dst[:3, :3] = R_rel.T          # p_dst = R_rel^T p_src ... so that p_dst = R p_src
        pts, cols = reproject_depth_to_other_pose(K, depth, img, pose_src, pose_dst)
        # analytic: uv_dst ~ K R K^-1 uv_src with R = pose_dst.R^T @ pose_src.R
        R = pose_dst[:3, :3].T @ pose_src[:3, :3]
        Hm = rotation_homography(K, R)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth.reshape(-1)
        uv_src = np.stack([u.reshape(-1), v.reshape(-1), np.ones(H * W)], axis=1)
        uv_pred = (Hm @ uv_src.T).T
        uv_pred = uv_pred[:, :2] / uv_pred[:, 2:3]   # K R K^-1 already includes K
        p = pts
        uvp = np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2],
                        K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1)
        err = np.abs(uvp - uv_pred).max()

        # rendered comparison against a nearest-neighbour warp by the same homography
        warped = np.zeros_like(img)
        inv = np.linalg.inv(Hm)
        uv_dst = np.stack([u.reshape(-1), v.reshape(-1), np.ones(H * W)], axis=1)
        src = (inv @ uv_dst.T).T
        src = src[:, :2] / src[:, 2:3]
        su = np.round(src[:, 0]).astype(int).reshape(H, W)
        sv = np.round(src[:, 1]).astype(int).reshape(H, W)
        ok = (su >= 0) & (su < W) & (sv >= 0) & (sv < H)
        warped[ok] = img[sv[ok], su[ok]]
        rendered = project_to_2d_image_seq2(K, pts, cols, (H, W))
        m = (rendered.sum(-1) > 0) & (warped.sum(-1) > 0)
        mad = float(np.abs(rendered.astype(float)[m] - warped.astype(float)[m]).mean()) if m.any() else None
        out[name] = {"point_uv_max_err_px": float(err), "rendered_vs_homography_mad": mad,
                     "rendered_filled": float((rendered.sum(-1) > 0).mean())}

    # same test on a real frame and the recorded pose, rotated in the camera frame
    i = 40
    R_rel = Rotation.from_euler("x", 12, degrees=True).as_matrix()
    pose_src = traj["pose"][i].copy()
    pose_dst = pose_src.copy()
    pose_dst[:3, :3] = pose_src[:3, :3] @ R_rel      # rotate about the camera origin
    pts, cols = reproject_depth_to_other_pose(K, traj["depth"][i], rgb(traj, i), pose_src, pose_dst)
    R = pose_dst[:3, :3].T @ pose_src[:3, :3]
    Hm = rotation_homography(K, R)
    u, v = np.meshgrid(np.arange(512), np.arange(512))
    z = traj["depth"][i].reshape(-1)
    uv_src = np.stack([u.reshape(-1), v.reshape(-1), np.ones(z.size)], axis=1)
    uv_pred = (Hm @ uv_src.T).T
    uv_pred = uv_pred[:, :2] / uv_pred[:, 2:3]
    uvp = np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2],
                    K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], axis=1)
    # image-level check on real texture: the rendered prior must agree with the analytic
    # homography warp (cv2 bilinear) where both are defined
    rendered = project_to_2d_image_seq2(K, pts, cols, (512, 512))
    warped = cv2.warpPerspective(rgb(traj, i), Hm, (512, 512))
    m = (rendered.sum(-1) > 0) & (warped.sum(-1) > 0)
    a = rendered[m].astype(float).ravel()
    b = warped[m].astype(float).ravel()
    a -= a.mean(); b -= b.mean()
    ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    out["real_frame_roll12"] = {"point_uv_max_err_px": float(np.abs(uvp - uv_pred).max()),
                                "n_points": int(len(pts)),
                                "ncc_rendered_vs_homography": ncc,
                                "overlap_frac": float(m.mean())}
    return out


# ------------------------------------------------------------------ test 2 and 3
def build_designs(start_pos, start_yaw):
    rows = {}
    for name, man, tgt, pol in DESIGNS:
        spec = design(man, tgt, 5.0, 17, 1.0, pol)
        pos, yaw, acc = place(spec, start_pos, start_yaw)
        zB = thrust_from_accel(acc)
        rows[name] = {"spec": spec, "pos": pos, "yaw": yaw, "accel": acc, "zB": zB}
    return rows


def test_axis_and_signs(rows, R_ec) -> dict:
    axis_err, sign_rows = {}, {}
    for name, r in rows.items():
        poses = pose_with_attitude(r["pos"][1:], r["yaw"][1:], r["zB"][1:], R_ec)
        rec = body_z_from_poses(poses, R_ec)
        axis_err[name] = float(np.abs(rec - r["zB"][1:]).max())

        # release parameterisation reproduces the same matrix
        phi, theta = roll_pitch_from_thrust(r["zB"][1:], r["yaw"][1:])
        R_wb = body_rotation(r["yaw"][1:], theta, phi)
        y_e, p_e, r_e = release_euler_triple(R_wb)
        from scipy.spatial.transform import Rotation
        Rt = Rotation.from_euler("zyx", np.stack([y_e, p_e, r_e], axis=1)).as_matrix()
        euler_err = float(np.abs(Rt - R_wb).max())

        a_xy = r["accel"][:, :2]
        z_xy = r["zB"][:, :2]
        m = np.linalg.norm(a_xy, axis=1) > 0.1
        dots = np.sum(a_xy[m] * z_xy[m], axis=1)
        sign_rows[name] = {
            "n_tilted_steps": int(m.sum()),
            "min_dot_accel_xy_thrust_xy": float(dots.min()) if dots.size else None,
            "max_abs_roll_deg": float(np.degrees(np.abs(phi)).max()),
            "max_abs_pitch_deg": float(np.degrees(np.abs(theta)).max()),
            "max_zB_tilt_deg": float(np.degrees(np.arccos(np.clip(r["zB"][:, 2].min(), -1, 1)))),
            "euler_reconstruction_err": euler_err,
        }
    return {"axis_max_err": max(axis_err.values()), "axis_err_per_design": axis_err,
            "signs": sign_rows}


# ------------------------------------------------------------------ test 4: coverage + sheets
def coverage_and_sheets(rows, traj, start_t, R_ec) -> dict:
    ctx = list(range(start_t - CTX + 1, start_t + 1))
    rgbs = np.stack([rgb(traj, t) for t in ctx])
    out, sheets = {}, {}
    for name in ("turn_10_fixed", "turn_20_fixed", "slalom_10_fixed", "accel_10", "accel_20",
                 "climb_20_fixed", "cruise_ctl"):
        r = rows[name]
        lvl = pose_from_position_yaw(r["pos"][1:], r["yaw"][1:], R_ec)
        inj = pose_with_attitude(r["pos"][1:], r["yaw"][1:], r["zB"][1:], R_ec)
        img_l = ffp(traj, ctx, rgbs, lvl)
        img_i = ffp(traj, ctx, rgbs, inj)
        cov_l = (img_l.sum(-1) > 0).mean(axis=(1, 2))
        cov_i = (img_i.sum(-1) > 0).mean(axis=(1, 2))
        out[name] = {"level_mean": float(cov_l.mean()), "level_min": float(cov_l.min()),
                     "injected_mean": float(cov_i.mean()), "injected_min": float(cov_i.min()),
                     "max_tilt_deg": float(np.degrees(np.arccos(
                         np.clip(r["zB"][:, 2].min(), -1, 1))))}
        # contact sheet: level row over injected row, 16 steps
        h, w = img_l.shape[1:3]
        sheet = np.zeros((2 * h, 16 * w, 3), np.uint8)
        for i in range(16):
            sheet[:h, i * w:(i + 1) * w] = img_l[i]
            sheet[h:, i * w:(i + 1) * w] = img_i[i]
        p = OUT / "stage0" / f"priors_{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        sheets[name] = str(p.relative_to(ROOT))
    return {"coverage": out, "sheets": sheets}


def main() -> int:
    traj = load()
    R_ec, spread = camera_axis_change(traj["pose"], traj["yaw"])
    start_t = 39
    start_pos = traj["point"][start_t]
    start_yaw = float(traj["yaw"][start_t])

    res = {"traj": TRAJ, "start_time": start_t,
           "R_ec": R_ec.tolist(), "R_ec_spread": spread,
           "R_ec_equals_notebook_t_ec": bool(np.abs(R_ec - np.array(
               [[0, 0, 1], [1, 0, 0], [0, 1, 0]])).max() < 1e-12),
           "pose_translation_equals_point": float(np.abs(
               traj["pose"][:, :3, 3] - traj["point"]).max())}
    res["pure_rotation"] = test_pure_rotation(traj)

    rows = build_designs(start_pos, start_yaw)
    res["axis_and_signs"] = test_axis_and_signs(rows, R_ec)
    res["coverage"] = coverage_and_sheets(rows, traj, start_t, R_ec)
    run = Path(__file__).resolve().parents[1] / "results/anwm_injected/stage0.json"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(json.dumps(res, indent=1))

    print("R_ec == notebook t_ec:", res["R_ec_equals_notebook_t_ec"], "spread", spread)
    print("\npure rotation (point-level max err px, rendered vs homography MAD):")
    for k, v in res["pure_rotation"].items():
        print(f"  {k:16s} {v}")
    print("\naxis max err over designs:", res["axis_and_signs"]["axis_max_err"])
    print("\nper design:")
    for k, v in res["axis_and_signs"]["signs"].items():
        print(f"  {k:16s} tilted {v['n_tilted_steps']:2d} min dot(a_xy,zB_xy) "
              f"{v['min_dot_accel_xy_thrust_xy']} roll {v['max_abs_roll_deg']:5.2f} "
              f"pitch {v['max_abs_pitch_deg']:5.2f} tilt {v['max_zB_tilt_deg']:5.2f} "
              f"euler_err {v['euler_reconstruction_err']:.1e}")
    print("\ncoverage (level vs injected, mean/min over 16 steps):")
    for k, v in res["coverage"]["coverage"].items():
        print(f"  {k:16s} tilt {v['max_tilt_deg']:5.1f} deg  level {v['level_mean']:.3f}/"
              f"{v['level_min']:.3f}  injected {v['injected_mean']:.3f}/{v['injected_min']:.3f}")
    print("\nsheets:", json.dumps(res["coverage"]["sheets"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
