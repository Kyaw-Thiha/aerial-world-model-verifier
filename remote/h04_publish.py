"""H04: publish the campaign clips to R2 under ``track-h3/`` with a sha256 manifest.

Uploads only the per-clip ``clip.mp4`` and ``manifest.json`` (the PNG frames stay on the
volume), then re-reads every object and records its sha256.  The manifest itself is uploaded
last.  Reads from R2 are allowed under this key only for objects it wrote, so the check is a
GET of the object just written.

    set -a; . ./.env; set +a
    pixi run modal run remote/h04_publish.py::main --tag main
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-h3-publish")
vol_r = modal.Volume.from_name("awmv-h3-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")
secret = modal.Secret.from_dotenv()


@app.function(image=image, cpu=4, timeout=2 * 3600, volumes={"/r": vol_r}, secrets=[secret])
def publish(tag: str = "main", prefix: str = "track-h3/clips/solarwm-main") -> dict:
    import hashlib
    import os

    import boto3

    c = boto3.client("s3",
                     endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                     aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                     aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                     region_name="auto")
    bucket = os.environ["R2_BUCKET"]
    root = Path("/r/solarwm") / tag
    rows = []
    for clip in sorted(root.iterdir()):
        mp4 = clip / "clip.mp4"
        man = clip / "manifest.json"
        if not mp4.exists():
            continue
        body = mp4.read_bytes()
        key = f"{prefix}/{clip.name}/clip.mp4"
        c.put_object(Bucket=bucket, Key=key, Body=body)
        back = c.get_object(Bucket=bucket, Key=key)["Body"].read()
        sha = hashlib.sha256(body).hexdigest()
        rows.append({"clip": clip.name, "key": key, "bytes": len(body),
                     "sha256": sha, "sha256_readback": hashlib.sha256(back).hexdigest(),
                     "match": hashlib.sha256(back).hexdigest() == sha})
        if man.exists():
            mb = man.read_bytes()
            mk = f"{prefix}/{clip.name}/manifest.json"
            c.put_object(Bucket=bucket, Key=mk, Body=mb)
            rows[-1].update({"manifest_key": mk,
                             "manifest_sha256": hashlib.sha256(mb).hexdigest()})
    doc = {"prefix": prefix, "tag": tag, "n": len(rows), "files": rows}
    payload = json.dumps(doc, indent=1).encode()
    mk = f"track-h3/manifests/{tag}.json"
    c.put_object(Bucket=bucket, Key=mk, Body=payload)
    vol_r.commit()
    return {"n": len(rows), "manifest_key": mk, "all_match": all(r["match"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows)}


@app.local_entrypoint()
def main(tag: str = "main"):
    print(json.dumps(publish.remote(tag=tag), indent=1))
