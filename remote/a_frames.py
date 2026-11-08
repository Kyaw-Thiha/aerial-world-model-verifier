"""Extract a uniform sample of TII frames from the Volume to a small local set.

The eight camera zips (3.94 GB) live on Volume ``awmv-a-data`` because local disk
is nearly full. Track A needs only a few hundred frames to develop undistortion,
the extrinsic check and the overlay plots, so this writes those to
``/vol/tii_frames/<flight>/`` and they are pulled with ``modal volume get``.

    pixi run modal run remote/a_frames.py --per-flight 40
    pixi run modal volume get awmv-a-data tii_frames data/raw/tii_frames
"""
from __future__ import annotations

import modal

app = modal.App("awmv-a-frames")

volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)
VOLUME_MOUNT = "/vol"

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.1.3")

FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=1800, cpu=2)
def extract(per_flight: int = 40, out: str = "tii_frames") -> dict:
    import json
    import zipfile
    from pathlib import Path

    src = Path(VOLUME_MOUNT) / "tii_video"
    dst_root = Path(VOLUME_MOUNT) / out
    summary = {}
    for kind, flights in FLIGHTS.items():
        for flight in flights:
            zp = src / kind / f"camera_{flight}.zip"
            d = dst_root / flight
            d.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as z:
                names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
                idx = sorted({int(round(i)) for i in
                              __import__("numpy").linspace(0, len(names) - 1, per_flight)})
                for i in idx:
                    memb = names[i]
                    (d / Path(memb).name).write_bytes(z.read(memb))
            summary[flight] = {"n_in_zip": len(names), "written": len(idx),
                               "first": Path(names[0]).name, "last": Path(names[-1]).name}
    volume.commit()
    (dst_root / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=1800, cpu=2)
def extract_windows(spec: dict, out: str = "tii_windows") -> dict:
    """Frames at a fixed rate inside explicit absolute time windows."""
    import json
    import zipfile
    from pathlib import Path

    import numpy as np

    src = Path(VOLUME_MOUNT) / "tii_video"
    dst_root = Path(VOLUME_MOUNT) / out
    fps = float(spec["fps"])
    summary = {}
    for flight, w in spec["flights"].items():
        kind = "autonomous" if flight.split("-")[1].endswith("a") else "piloted"
        zp = src / kind / f"camera_{flight}.zip"
        d = dst_root / flight
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
            ts = np.array([int(Path(n).stem.split("_")[1]) for n in names])
            want = np.arange(w["abs_from_us"], w["abs_to_us"], 1e6 / fps)
            picks = sorted({int(np.argmin(np.abs(ts - t))) for t in want})
            for i in picks:
                memb = names[i]
                (d / Path(memb).name).write_bytes(z.read(memb))
        summary[flight] = {"written": len(picks), "t0": float(w["t0"]), "t1": float(w["t1"])}
    volume.commit()
    return summary


@app.local_entrypoint()
def main(per_flight: int = 40, out: str = "tii_frames", windows_spec: str = ""):
    import json

    if windows_spec:
        print(json.dumps(extract_windows.remote(json.loads(open(windows_spec).read())), indent=2))
        return
    print(json.dumps(extract.remote(per_flight, out), indent=2))
