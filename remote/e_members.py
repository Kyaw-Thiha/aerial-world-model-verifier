"""Track E: pull the three small text members out of a UZH sequence zip.

The clock audit needs ``groundtruth.txt``, ``imu.txt`` and ``left_images.txt`` and nothing else;
they total a few MB out of a 0.7-2.6 GB archive. Doing this in one tiny Modal job and running
the correlation locally keeps the analysis where it can be inspected and re-run instantly.

    pixi run modal run remote/e_members.py --seqs indoor_forward_5 --workers 4
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-members")

e_data = modal.Volume.from_name("awmv-e-data", create_if_missing=True)
c_data = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
E_DATA = "/e-data"
C_DATA = "/c-data"
UZH_BASE = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3"

MEMBERS = ("groundtruth.txt", "imu.txt", "left_images.txt", "right_images.txt")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


def _zip_path(seq: str):
    import urllib.request
    from pathlib import Path

    stem = f"{seq}_snapdragon_with_gt.zip"
    own = Path(E_DATA) / "zips" / stem
    if own.exists():
        return own
    alt = Path(C_DATA) / "zips" / stem
    if alt.exists():
        return alt
    own.parent.mkdir(parents=True, exist_ok=True)
    tmp = own.with_suffix(".part")
    with urllib.request.urlopen(f"{UZH_BASE}/{stem}", timeout=1800) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(own)
    return own


@app.function(image=image, volumes={E_DATA: e_data, C_DATA: c_data}, timeout=1800, cpu=2)
def members(seq: str) -> str:
    """Copy the text members of one sequence to ``/e-data/members/<seq>/``."""
    import json
    import shutil
    import zipfile
    from pathlib import Path

    zpath = _zip_path(seq)
    out = Path(E_DATA) / "members" / seq
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    with zipfile.ZipFile(zpath) as z:
        have = set(z.namelist())
        for m in MEMBERS:
            if m in have:
                (out / m).write_bytes(z.read(m))
                got[m] = (out / m).stat().st_size
        sizes = {"zip_bytes": zpath.stat().st_size, "n_members": len(have)}
    e_data.commit()
    rec = {"seq": seq, "members": got, **sizes, "zip": str(zpath)}
    print(json.dumps(rec), flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "", order: str = "") -> None:
    """``order`` is an optional comma list to run first (cheapest / already-cached zips)."""
    import json

    from src.depicted.uzh import all_sequences

    wanted = [s for s in seqs.split(",") if s] or all_sequences()
    if order:
        first = [s for s in order.split(",") if s]
        wanted = first + [s for s in wanted if s not in first]
    handles = [(s, members.spawn(s)) for s in wanted]
    for s, h in handles:
        try:
            print(s, json.loads(h.get())["members"], flush=True)
        except Exception as exc:
            print(f"FAILED {s}: {exc}", flush=True)
