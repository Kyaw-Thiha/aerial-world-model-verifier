"""Known-truth testbeds for the path-from-pixels track, as one schema each.

Three sources, all with metric ground truth and a declared time base:

* **UZH-FPV** (``T1``) — real drone racing, Leica laser-tracker truth, and a *proper*
  Kannala-Brandt (equidistant) calibration shipped separately from the sequence
  (``.../calib/*_calib_snapdragon.zip``). Frames are 640x480 grey at 30 Hz; speeds 3.5 to
  20.7 m/s. Undistorted to a pinhole view with the same focal, so the field narrows and the
  calibration stays valid.
* **TII** (``T2``) — the project's own real flights, mocap truth, no trustworthy intrinsics:
  frames are cropped to a central pinhole-like window and the extractor must estimate K.
  Truth comes from :func:`src.datasets.tii.load_tii`.
* **ANWM** (``T3``) — generated clips plus the ground-truth sim frames from the same start
  (Regime A), so extractor error and model error can be separated.

Every loader returns a :class:`PathRecClip`. Units are SI, the world frame is z-up, and the
quaternion order is stated on the object; fps is measured and recorded, never assumed.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

UZH_ROOT_DEFAULT = "data/raw/uzh"
UZH_CALIB = "indoor_forward_calib_snapdragon.zip"
UZH_NS = 1e-9  # UZH timestamps are nanoseconds on the dataset clock


@dataclass
class PathRecClip:
    """One clip with pixels and truth (or a commanded path), all in SI, z-up."""

    name: str
    source: str
    t: np.ndarray                 # (N,) seconds since the first sample
    pos: np.ndarray               # (N,3) metres, world z-up
    quat: np.ndarray              # (N,4) body-to-world, laid out per quat_order
    quat_order: str
    fps: float
    intrinsics: dict
    frames: list[str] = field(default_factory=list)
    gt_frames: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.quat = np.asarray(self.quat, dtype=np.float64)
        if self.quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"quat_order must be xyzw or wxyz, got {self.quat_order}")
        if not (len(self.t) == len(self.pos) == len(self.quat)):
            raise ValueError(f"{self.name}: t/pos/quat length mismatch "
                             f"{len(self.t)}/{len(self.pos)}/{len(self.quat)}")
        if len(self.t) > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError(f"{self.name}: timestamps are not strictly increasing")
        if self.fps <= 0:
            raise ValueError(f"{self.name}: fps must be positive, got {self.fps}")
        if self.frames and len(self.frames) != len(self.t):
            raise ValueError(f"{self.name}: {len(self.frames)} frames for {len(self.t)} poses")

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0


# --------------------------------------------------------------------------- UZH-FPV
def _cv2():
    """Import OpenCV lazily: the default analysis environment has no cv2 (it is GPU-side)."""
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "opencv is only installed in the Modal images; run frame preparation there "
            "(remote/c_prepare.py) instead of locally") from exc
    return cv2


def uzh_calib(calib_zip: str | Path, cam: str = "cam0") -> dict:
    """Read one camera from the Kalibr ``camchain`` YAML in the calibration zip."""
    calib_zip = Path(calib_zip)
    with zipfile.ZipFile(calib_zip) as z:
        name = next(n for n in z.namelist() if "camchain-" in n and n.endswith(".yaml")
                    and "imucam" not in n)
        chain = yaml.safe_load(z.read(name))
    c = chain[cam]
    fx, fy, cx, cy = (float(v) for v in c["intrinsics"])
    dist = [float(v) for v in c["distortion_coeffs"]]
    if c["camera_model"] != "pinhole" or c["distortion_model"] != "equidistant":
        raise ValueError(f"unexpected UZH model {c['camera_model']}/{c['distortion_model']}")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k": dist, "w": int(c["resolution"][0]), "h": int(c["resolution"][1]),
            "model": "pinhole+equidistant", "source": calib_zip.name}


def uzh_image_list(z: zipfile.ZipFile, cam_index: int = 0) -> list[tuple[float, str]]:
    """``left_images.txt`` rows ``(timestamp_unix_s, image_name)`` for one camera.

    The image *filenames* carry an index, not a timestamp (``image_0_1968.png``), so the
    time base must come from this list; the calibration and ground truth share the same
    Unix clock.
    """
    side = "left" if cam_index == 0 else "right"
    name = next(n for n in z.namelist() if Path(n).name == f"{side}_images.txt")
    rows = []
    for line in z.read(name).decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[1]), parts[2]))
    if len(rows) < 2:
        raise ValueError(f"{name}: fewer than 2 image rows")
    rows.sort(key=lambda r: r[0])
    return rows


def uzh_groundtruth(z: zipfile.ZipFile) -> np.ndarray:
    """TUM rows ``timestamp tx ty tz qx qy qz qw`` from ``groundtruth.txt``."""
    txt = z.read("groundtruth.txt").decode()
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    out = np.asarray(rows, dtype=np.float64)
    if len(out) < 2:
        raise ValueError("groundtruth.txt has fewer than 2 rows")
    return out[np.argsort(out[:, 0])]


def prepare_uzh(zip_path: str | Path, calib_zip: str | Path, out_dir: str | Path,
                stride: int = 4, max_frames: int = 600, cam_index: int = 0,
                out_size: tuple[int, int] = (640, 480)) -> PathRecClip:
    """Undistort one UZH sequence to a pinhole view and pair it with Leica truth.

    The output focal equals the calibrated focal, so the undistorted view keeps metric
    geometry and simply sees a narrower field (no rescaling of the calibration).
    """
    cv2 = _cv2()
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    calib = uzh_calib(calib_zip)
    K_in = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1]])
    dist = np.asarray(calib["k"], dtype=np.float64)
    w_out, h_out = out_size
    K_out = np.array([[calib["fx"], 0, w_out / 2], [0, calib["fy"], h_out / 2], [0, 0, 1]],
                     dtype=np.float64)

    with zipfile.ZipFile(zip_path) as z:
        gt = uzh_groundtruth(z)
        rows = uzh_image_list(z, cam_index)
        # the Leica truth covers only part of the recording; keep the overlap so every
        # kept frame has interior truth (no extrapolation)
        lo, hi = gt[0, 0], gt[-1, 0]
        rows = [r for r in rows if lo <= r[0] <= hi]
        if len(rows) < 3:
            raise ValueError(f"{zip_path.name}: no frames inside the truth interval")
        rows = rows[::stride][:max_frames]
        ts_s = np.array([r[0] for r in rows], dtype=np.float64)   # Unix seconds
        t = ts_s - ts_s[0]
        names = [r[1] for r in rows]
        paths = []
        for i, n in enumerate(names):
            img = cv2.imdecode(np.frombuffer(z.read(n), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            und = cv2.fisheye.undistortImage(img, K_in, dist, Knew=K_out,
                                            new_size=(w_out, h_out))
            p = frames_dir / f"{i:05d}.png"
            cv2.imwrite(str(p), und)
            paths.append(str(p))

    # truth on the frame clock (nearest sample: the tracker runs far faster than 30 Hz)
    t_gt = gt[:, 0] - ts_s[0]
    idx = np.searchsorted(t_gt, t).clip(1, len(t_gt) - 1)
    left = np.abs(t_gt[idx - 1] - t) < np.abs(t_gt[idx] - t)
    idx = np.where(left, idx - 1, idx)
    pos = gt[idx, 1:4]
    quat = gt[idx, 4:8]                       # TUM order is xyzw
    fps = 1.0 / float(np.median(np.diff(t)))
    clip = PathRecClip(
        name=zip_path.stem.replace("_snapdragon_with_gt", ""),
        source="uzh",
        t=t, pos=pos, quat=quat, quat_order="xyzw", fps=fps,
        intrinsics={"fx": float(K_out[0, 0]), "fy": float(K_out[1, 1]),
                    "cx": float(K_out[0, 2]), "cy": float(K_out[1, 2]),
                    "w": w_out, "h": h_out, "model": "pinhole",
                    "source_calib": calib},
        frames=paths,
        meta={"sequence": zip_path.stem, "stride": stride, "cam_index": cam_index,
              "gt_source": "Leica tracker (groundtruth.txt, TUM rows xyzw)",
              "undistorted": True, "license": "CC BY-NC-SA 3.0"},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "clip": clip.name, "fps": fps, "n": len(t), "intrinsics": clip.intrinsics,
        "stride": stride, "cam_index": cam_index, "source_zip": str(zip_path)}, indent=1))
    np.savez_compressed(out_dir / "truth.npz", t=t, pos=pos, quat=quat)
    return clip


def load_prepared_uzh(out_dir: str | Path) -> PathRecClip:
    """Reload a prepared UZH clip (frames + truth.npz + meta.json)."""
    out_dir = Path(out_dir)
    truth = np.load(out_dir / "truth.npz")
    meta = json.loads((out_dir / "meta.json").read_text())
    paths = [str(p) for p in sorted((out_dir / "frames").glob("*.png"))]
    return PathRecClip(name=meta["clip"], source="uzh", t=truth["t"], pos=truth["pos"],
                       quat=truth["quat"], quat_order="xyzw", fps=float(meta["fps"]),
                       intrinsics=meta["intrinsics"], frames=paths,
                       meta={"prepared_from": str(out_dir)})


# --------------------------------------------------------------------------- ANWM / T3
def load_anwm(root: str | Path, clip_id: str) -> PathRecClip:
    """One ANWM clip: generated frames, the same-start ground-truth frames, and the command.

    Regime A stores ``frames/`` (generated) and ``gt/`` (simulator frames with true poses);
    the manifest holds the commanded path. The commanded path is a *command*: for generated
    frames it is not necessarily what was depicted, which is exactly the attribution split.
    """
    root = Path(root)
    d = root / clip_id
    if not (d / "manifest.json").exists():      # clips are nested one level (A/A00)
        cands = [m.parent for m in root.rglob(f"*/{clip_id}/manifest.json")]
        if cands:
            d = cands[0]
    man = json.loads((d / "manifest.json").read_text())
    frames = [str(p) for p in sorted((d / "frames").glob("*.png"))]
    gt_frames = [str(p) for p in sorted((d / "gt").glob("*.png"))] if (d / "gt").is_dir() else []
    cmd = np.asarray(man["commanded_path"]["positions"], dtype=np.float64)
    n = len(frames)
    # one frame per commanded step: frame i depicts the pose *after* the action ending at
    # positions[i+1], so the path rows that line up with frames are positions[1:n+1]
    if len(cmd) == n + 1:
        cmd = cmd[1:]
    elif len(cmd) != n:
        cmd = cmd[:n] if len(cmd) > n else np.pad(cmd, ((0, n - len(cmd)), (0, 0)), mode="edge")
    # no fps in the ANWM release; 1 s per step is the documented working assumption
    fps = float(man.get("fps") or 1.0)
    dt = 1.0 / fps
    return PathRecClip(
        name=clip_id, source="anwm",
        t=np.arange(n) * dt, pos=cmd,
        quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), quat_order="xyzw", fps=fps,
        intrinsics={"fx": 112.0, "fy": 112.0, "cx": 112.0, "cy": 112.0, "w": 224, "h": 224,
                    "model": "pinhole", "source": "manifest (f=112 px)"},
        frames=frames, gt_frames=gt_frames,
        meta={"regime": man.get("regime"), "start_state": man.get("start_state"),
              "pose_semantics": "commanded path, not depicted motion",
              "fps_semantics": "1 s per commanded step: working assumption from the "
                               "release, unverified; not measured",
              "gt_frames_semantics": "Regime A ground-truth renders are exactly level "
                                     "(roll = pitch = 0) by construction, not a recorded "
                                     "attitude; use them as a level reference only"},
    )


def anwm_clip_ids(root: str | Path, regime: str | None = None) -> list[str]:
    root = Path(root)
    ids = []
    for man in sorted(root.rglob("manifest.json")):
        m = json.loads(man.read_text())
        if regime is None or m.get("regime") == regime:
            ids.append(str(man.parent.relative_to(root)))
    return ids


# --------------------------------------------------------------------------- TII / T2
def tii_crop_intrinsics(width: int, height: int, fov_deg: float = 100.0) -> dict:
    """Intrinsics of the central pinhole-like crop used for TII (no released calibration).

    The released TII calibration is a pinhole fit of a 175 deg fisheye and is not used.
    """
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return {"fx": float(f), "fy": float(f), "cx": width / 2.0, "cy": height / 2.0,
            "w": int(width), "h": int(height), "model": "pinhole (central crop)",
            "source": "computed from the crop field of view, not from the release"}


def load_tii_truth(mocap_root: str | Path, flight: str):
    """Mocap truth through the project loader (native 274.73 Hz, z-up, xyzw)."""
    from src.datasets.tii import load_tii
    control = "piloted" if flight.split("-")[1].endswith("p") else "autonomous"
    return load_tii(Path(mocap_root) / control / f"mocap_{flight}.csv", name=flight)
