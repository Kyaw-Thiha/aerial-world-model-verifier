"""Pull a few *native* ANWM test trajectories (full pickle + JPEGs) onto a Modal Volume.

Only 4 of the 29 trajectories in `data/splits/airvln_16/test/rollout_16.pkl` are covered by
the shard index we already have, so this indexes more shards until it finds the requested
number, then streams each trajectory's `traj_data.pkl` (which carries depth and pose) and the
JPEGs of its first rollout entry onto `awmv-b-native` at `/n/airvln_16/<traj>/`.

Shard indexing is header-only over HTTP Range (see scripts/fetch_anwm.py); one shard costs
about two minutes of range requests, so the scan is threaded and stops as soon as it has enough.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
vol_native = modal.Volume.from_name("awmv-b-native", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "pillow")
    .add_local_file(str(ROOT / "scripts" / "fetch_anwm.py"), "/fetch_anwm.py", copy=True)
    .add_local_dir(str(ROOT / "third_party" / "anwm" / "data" / "splits" / "airvln_16"),
                   "/splits", copy=True)
)
app = modal.App("awmv-b-native", image=image)


@app.function(cpu=8, memory=16384, timeout=3 * 3600, volumes={"/n": vol_native})
def fetch(names_wanted: int = 5, max_shards: int = 140, workers: int = 32) -> dict:
    import socket

    socket.setdefaulttimeout(90)  # a hung range read must not stall the whole scan
    sys.path.insert(0, "/")
    from fetch_anwm import N_SHARDS, RangeFile, index_shard, shard_url

    split = Path("/splits")
    if not split.is_dir():
        raise SystemExit("split metadata missing")
    rollout = pickle.load(open(split / "test" / "rollout_16.pkl", "rb"))
    test_names = list(dict.fromkeys(x[0] for x in rollout))
    first_entry = {}
    for t, ct, lo, hi in rollout:
        first_entry.setdefault(t, (ct, lo, hi))
    # four candidate start times per trajectory, spaced so the windows do not overlap
    CAND = (39, 79, 119, 159)

    index = {}
    if os.path.exists("/n/shard_index.json"):
        index = json.load(open("/n/shard_index.json"))
    missing_names = [n for n in test_names if n not in index]
    print(f"{len(test_names)} rollout trajectories, {len(test_names)-len(missing_names)} indexed")

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    todo = [i for i in range(N_SHARDS) if not any(v[0] == i for v in index.values())][:max_shards]
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = {ex.submit(index_shard, i): i for i in todo}
    done = 0
    try:
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    index.update(fut.result())
                except Exception as e:
                    print("shard failed", type(e).__name__, e)
                done += 1
            Path("/n/shard_index.json").write_text(json.dumps(index))
            vol_native.commit()
            found = [n for n in test_names if n in index]
            if done % 5 == 0 or len(found) >= names_wanted:
                print(f"{done} shards scanned, {len(found)}/{names_wanted} wanted found, "
                      f"{time.time()-t0:.0f}s", flush=True)
            if len(found) >= names_wanted:
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    Path("/n/shard_index.json").write_text(json.dumps(index))
    vol_native.commit()
    found = [n for n in test_names if n in index][:names_wanted]
    print(f"index now {len(index)} trajectories, rollout hits {len(found)} in {time.time()-t0:.0f}s")
    if len(found) < names_wanted:
        raise SystemExit(f"only {len(found)} of the {names_wanted} wanted trajectories indexed")

    base = "/n/airvln_16"
    done = {}
    for name in found:
        shard, off, size = index[name]
        ct = first_entry[name][0]
        # 16 context frames + 16 future frames: the released checkpoint's pos_embed has 17 rows
        # (16 context + 1 predicted), whatever config/anwm.yaml says
        need = set(range(max(0, ct - 15), ct + 17))
        for c in CAND:
            need |= set(range(max(0, c - 15), c + 17))
        f = RangeFile(shard_url(shard))
        outd = Path(base, name)
        outd.mkdir(parents=True, exist_ok=True)
        got = {"shard": shard, "offset": off, "pickle_bytes": size, "curr_time": ct,
               "start_candidates": [c for c in CAND], "frames": []}
        with tarfile.open(fileobj=f, mode="r:") as tf:
            for m in tf:
                if m.name == f"{name}/traj_data.pkl":
                    src = tf.extractfile(m)
                    left = m.size
                    with open(outd / "traj_data.pkl", "wb") as fh:
                        while left > 0:
                            chunk = src.read(min(1 << 24, left))
                            if not chunk:
                                break
                            fh.write(chunk)
                            left -= len(chunk)
                    got["pickle_bytes_written"] = (outd / "traj_data.pkl").stat().st_size
                elif m.name.startswith(f"{name}/") and m.name.endswith(".jpg"):
                    frame = int(Path(m.name).stem)
                    if frame in need:
                        img = tf.extractfile(m).read()
                        (outd / f"{frame}.jpg").write_bytes(img)
                        got["frames"].append(frame)
                elif m.name.startswith(f"{name}/") and not m.name.endswith((".jpg", ".pkl")):
                    got.setdefault("other_members", []).append(m.name)
        done[name] = got
        print(name, "pickle MB", round(got.get("pickle_bytes_written", 0) / 1e6, 1),
              "frames", len(got["frames"]))
        vol_native.commit()
    Path("/n/manifest.json").write_text(json.dumps(done, indent=1))
    vol_native.commit()
    return done


@app.local_entrypoint()
def main(names_wanted: int = 5):
    print(json.dumps(fetch.remote(names_wanted), indent=1)[:4000])
