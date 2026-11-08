"""Fetch TII camera video into Modal Volume ``awmv-a-data`` and mirror it to R2.

Local disk has ~7 GB free, so the eight camera zips (3.94 GB) live on the Volume
and in R2; only the small CSV/YAML files are kept locally. The code is the same
``scripts/fetch_tii_video.py`` used locally, so the manifest and sha256 checks are
identical.

    pixi run modal run remote/a_fetch_video.py
    pixi run modal run remote/a_fetch_video.py --no-r2      # volume only
"""
from __future__ import annotations

import modal

app = modal.App("awmv-a-fetch-video")

volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)
VOLUME_MOUNT = "/vol"
R2_PREFIX = "track-a/tii_video"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3>=1.34", "python-dotenv>=1.0")
    .add_local_file("scripts/fetch_tii.py", "/root/scripts/fetch_tii.py", copy=True)
    .add_local_file("scripts/fetch_tii_video.py", "/root/scripts/fetch_tii_video.py", copy=True)
    .add_local_file("docs/decisions.md", "/root/docs/decisions.md", copy=True)
)


@app.function(image=image, volumes={VOLUME_MOUNT: volume},
              secrets=[modal.Secret.from_dotenv()],
              timeout=3 * 3600, cpu=2, memory=4096)
def fetch(upload_r2: bool = True) -> dict:
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from scripts.fetch_tii_video import fetch as run_fetch

    out = Path(VOLUME_MOUNT) / "tii_video"
    lines: list[str] = []
    manifest = run_fetch(out, small_only=False, log=lines.append)
    volume.commit()

    uploaded = []
    if upload_r2:
        import boto3

        env = {k: os.environ[k] for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                                          "R2_SECRET_ACCESS_KEY", "R2_BUCKET")}
        s3 = boto3.client("s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                          aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"], region_name="auto")
        for rel in manifest:
            if not rel.endswith(".zip"):
                continue
            p = out / rel
            key = f"{R2_PREFIX}/{rel}"
            s3.upload_file(str(p), env["R2_BUCKET"], key)
            head = s3.head_object(Bucket=env["R2_BUCKET"], Key=key)
            h = hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < (1 << 27) else None
            uploaded.append({"key": key, "bytes": p.stat().st_size,
                             "remote_bytes": head["ContentLength"],
                             "sha256_local": manifest[rel]["sha256"], "sha256_checked": h})
    total = sum(v["bytes"] for v in manifest.values())
    return {"log_tail": lines[-12:], "files": len(manifest), "bytes": total,
            "uploaded": uploaded}


@app.local_entrypoint()
def main(upload_r2: bool = True):
    import json

    print(json.dumps(fetch.remote(upload_r2), indent=2))
