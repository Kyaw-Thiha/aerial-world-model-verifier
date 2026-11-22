"""ANWM rollout helpers shared by the Track H remote apps.

These are the same functions Track B used (``remote/b_anwm.py``), lifted into ``src`` so a
new app can mount them without importing another track's entrypoint. Nothing here changes
the model, the projection code or the conventions; the docstrings carry the verified facts
from ``reports/b05_pose_injection_stage0.md``.
"""
from __future__ import annotations

CTX = 16  # 16 context frames + 1 predicted frame (pos_embed has 17 rows)


def eval_shim(data_folder: str = "/n/airvln_16"):
    """An EvalDataset with attributes set directly, so every method stays ANWM's own code."""
    import numpy as np

    from anwm.config import load_yaml
    from anwm.data.airvln import EvalDataset
    from anwm.utils import transform

    cfg = load_yaml("config/data_config.yaml")
    ds = object.__new__(EvalDataset)
    ds.data_folder = data_folder
    ds.data_split_folder = "/anwm/data/splits/airvln_16"
    ds.dataset_name = "airvln_16"
    ds.image_size = (224, 224)
    ds.min_dist_cat, ds.max_dist_cat = -64, 64
    ds.len_traj_pred, ds.traj_stride, ds.context_size = 16, 80, CTX
    ds.normalize = True
    ds.transform = transform
    ds.goals_per_obs = 1
    ds.data_config = cfg["airvln_16"]
    ds.ACTION_STATS = {k: np.expand_dims(np.asarray(v, dtype=np.float32), 0)
                       for k, v in cfg["action_stats"].items()}
    return ds


def camera_axis_change(traj):
    """R_ec: the world->camera axis change baked into every released pose (constant check)."""
    import numpy as np

    yaw = np.asarray(traj["yaw"], float).ravel()
    R = np.asarray(traj["pose"], float)[:, :3, :3]
    out = []
    for k in range(len(yaw)):
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        out.append(Rz.T @ R[k])
    out = np.asarray(out)
    spread = float(np.abs(out - out[0]).max())
    if spread > 1e-3:
        raise RuntimeError(f"camera axis change is not constant (spread {spread:.2e})")
    return out[0], spread


def pose_from_position_yaw(pos, yaw, R_ec):
    import numpy as np

    poses = np.tile(np.eye(4), (len(pos), 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros((len(pos), 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
    Rz[:, 1, 0], Rz[:, 1, 1] = s, c
    Rz[:, 2, 2] = 1.0
    poses[:, :3, :3] = Rz @ R_ec[None]
    poses[:, :3, 3] = pos
    return poses


def ffp(traj, ctx_times, rgb, target_poses):
    """Future Frame Projection to new target poses, through ANWM's own projection functions."""
    import numpy as np
    from anwm.projection import (project_to_2d_image_seq2seq,
                                 reproject_depth_to_other_pose_seq2seq)

    depth_seq = np.asarray(traj["depth"])[ctx_times]
    poses_src = np.asarray(traj["pose"])[ctx_times]
    H, W = depth_seq.shape[-2:]
    pts, cols = reproject_depth_to_other_pose_seq2seq(
        K=traj["K"], depth_maps=depth_seq, rgb_imgs=rgb,
        poses_src=poses_src, poses_dst=np.asarray(target_poses))
    return project_to_2d_image_seq2seq(K=traj["K"], points_3d=pts, colors=cols,
                                       image_size=(H, W))


def context_inputs(ds, traj, ctx_times, target_poses, data_dir="/n/airvln_16"):
    """(obs tensor, x_supervised tensor, per-step prior coverage) through ANWM's FFP code."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    rgb = []
    for t in ctx_times:
        img = cv2.imread(f"{data_dir}/{traj['name']}/{int(t)}.jpg")
        rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgb = np.stack(rgb, 0)
    proj = ffp(traj, list(ctx_times), rgb, np.asarray(target_poses))
    coverage = [float((proj[i].sum(-1) > 0).mean()) for i in range(len(proj))]
    obs = torch.stack([ds.transform(Image.fromarray(rgb[i])) for i in range(len(ctx_times))])
    sup = torch.stack([ds.transform(Image.fromarray(proj[i])) for i in range(len(proj))])
    return obs.unsqueeze(0), sup.unsqueeze(0), coverage


def load_traj(cache: dict, name: str):
    import pickle

    if name not in cache:
        traj = pickle.load(open(f"/n/airvln_16/{name}/traj_data.pkl", "rb"))
        traj = {k: (v.astype("float") if hasattr(v, "astype") else v) for k, v in traj.items()}
        traj["name"] = name
        cache[name] = traj
    return cache[name]
