"""GeoCalib on TII frames, on Modal.

Fisheye frames are undistorted into the virtual pinhole view (same numpy code as
``src/perception/undistort.py``, vendored into the image via
``add_local_python_source``), the calibration is cached per flight, and GeoCalib
runs per frame. Two input paths are reported:

* ``undistorted`` + ``camera_model="pinhole"``, with and without the known
  virtual focal passed as a prior;
* ``raw`` fisheye + ``camera_model="simple_divisional"`` (GeoCalib's strong
  distortion model), which avoids the released calibration entirely.

Output is one JSONL per flight in the Volume plus a summary, so the mocap
comparison runs locally where the truth is.

    pixi run modal run remote/a_geocalib.py --mode prior --per-flight 16
    pixi run modal run remote/a_geocalib.py --mode free --stride 4 --all-frames
"""
from __future__ import annotations

import modal

app = modal.App("awmv-a-geocalib")

volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)
VOLUME_MOUNT = "/vol"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)

FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=2 * 3600)
def run(mode: str = "prior", per_flight: int = 16, stride: int = 4,
        fov: float = 100.0, source: str = "frames", run_tag: str = "",
        flights: list[str] | None = None) -> dict:
    import json
    import time
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.datasets.tii_video import flight_parts, load_calibration
    from src.perception import undistort as U

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole" if mode != "divisional" else "distorted").to(device)
    tag = run_tag or f"{mode}_{source}_{'all' if per_flight == 0 else per_flight}"
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    info: dict = {"mode": mode, "device": device, "fov_deg": fov, "stride": stride,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                  "geocalib_commit": GEOCALIB_COMMIT, "flights": {}}

    for flight in (flights or [f for v in FLIGHTS.values() for f in v]):
        control, track = flight_parts(flight)
        cal = load_calibration(Path(VOLUME_MOUNT) / "tii_video" / "calibration", control, track)
        vc = U.virtual_camera(cal["K"], cal["dist"], (640, 480), fov_deg=fov)
        rows = []

        if source == "frames":
            fdir = Path(VOLUME_MOUNT) / "tii_frames" / flight
            names = sorted(fdir.glob("*.jpg"))
            step = max(1, len(names) // per_flight) if per_flight else 1
            items = [(p.name, cv2.imread(str(p))) for p in names[::step]]
        else:  # stream the zip itself
            zp = Path(VOLUME_MOUNT) / "tii_video" / control / f"camera_{flight}.zip"
            with zipfile.ZipFile(zp) as z:
                names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
                step = stride if per_flight == 0 else max(1, len(names) // per_flight)
                items = []
                for n in names[::step]:
                    buf = np.frombuffer(z.read(n), dtype=np.uint8)
                    items.append((Path(n).name, cv2.imdecode(buf, cv2.IMREAD_COLOR)))

        t0 = time.time()
        for name, img in items:
            if img is None:
                continue
            if mode in ("undistorted", "free", "prior"):
                x = U.undistort(img, vc)
                priors = {"focal": torch.tensor([vc.K[0, 0], vc.K[1, 1]], device=device)} \
                    if mode == "prior" else None
                cm = "pinhole"
            else:  # divisional on the raw fisheye
                x = img
                priors = None
                cm = "simple_divisional"
            feed = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(feed).permute(2, 0, 1).float().to(device) / 255.0
            with torch.no_grad():
                res = model.calibrate(t, camera_model=cm, priors=priors)
            g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
            row = {"frame": name, "gravity_up_cam": [float(v) for v in g]}
            cam = res["camera"]
            for k in ("f", "k1", "k2"):
                if hasattr(cam, k):
                    row[k] = np.asarray(getattr(cam, k).detach().cpu()).ravel().tolist()
            rows.append(row)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        (out_dir / f"{flight}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        info["flights"][flight] = {"n": len(rows), "seconds": dt,
                                   "fps": len(rows) / dt if dt else 0.0,
                                   "peak_gpu_gb": peak, "virtual_valid_frac": vc.valid_fraction}
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        print(f"{flight}: {len(rows)} frames in {dt:.1f}s "
              f"({len(rows)/dt:.1f} fps), peak {peak:.2f} GB", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.local_entrypoint()
def main(mode: str = "prior", per_flight: int = 16, stride: int = 4, fov: float = 100.0,
         source: str = "frames", run_tag: str = ""):
    import json

    print(json.dumps(run.remote(mode, per_flight, stride, fov, source, run_tag), indent=2))
