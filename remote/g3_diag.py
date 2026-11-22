"""Track G3: one-off I/O diagnostic (ffmpeg PNG write vs read vs volume write)."""
from __future__ import annotations

import modal

app = modal.App("awmv-g3-diag")
in_vol = modal.Volume.from_name("awmv-i-data", create_if_missing=False)
out_vol = modal.Volume.from_name("awmv-g3-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libgl1")
         .pip_install("numpy<2", "opencv-python-headless==4.10.0.84"))


@app.function(image=image, gpu="L40S", timeout=1200,
              volumes={"/in": in_vol, "/out": out_vol})
def diag(clip_id: str = "i02_dev_022") -> dict:
    import subprocess
    import time
    from pathlib import Path

    import cv2
    import numpy as np

    cdir = Path("/in/i02/clips") / clip_id
    out = {}
    t = time.time()
    npz = np.load(cdir / "render.npz")
    out["npz_load_s"] = round(time.time() - t, 2)
    out["npz_keys"] = list(npz.files)
    t = time.time()
    frames = Path("/tmp/f") / clip_id
    frames.mkdir(parents=True, exist_ok=True)
    for f in frames.glob("*.png"):
        f.unlink()
    out["cleanup_s"] = round(time.time() - t, 2)
    out["files"] = sorted(p.name for p in cdir.iterdir())
    t = time.time()
    pr = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cdir / f"{clip_id}.mp4"),
                         "-vf", "select=not(mod(n,1))", "-vsync", "0", str(frames / "%04d.png")],
                        capture_output=True, text=True)
    out["ffmpeg_png_s"] = round(time.time() - t, 2)
    out["ffmpeg_rc"] = pr.returncode
    out["ffmpeg_err"] = (pr.stderr or "")[-400:]
    names = sorted(frames.glob("*.png"))
    out["n_png"] = len(names)
    t = time.time()
    for p in names[:50]:
        cv2.imread(str(p))
    out["imread50_s"] = round(time.time() - t, 2)
    t = time.time()
    for i in range(50):
        (frames / f"x{i}.png").write_bytes(b"x" * 1000)
    out["write50_small_s"] = round(time.time() - t, 2)
    t = time.time()
    (Path("/out") / "diag.json").write_text("{}" * 20000)
    out["volume_write_until_commit_s"] = round(time.time() - t, 2)
    t = time.time()
    out_vol.commit()
    out["volume_commit_s"] = round(time.time() - t, 2)
    import os
    out["cpus"] = os.cpu_count()
    (Path("/out") / "diag.json").write_text(__import__("json").dumps(out, indent=1))
    out_vol.commit()
    return out
