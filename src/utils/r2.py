"""Cloudflare R2 helpers (S3-compatible), used to move artifacts between machines.

Credentials come from ``.env`` via python-dotenv. Values are never printed or
logged: :func:`config_report` only reports whether each name is set.

Uploads are verified by comparing the sha256 of the local file with the sha256 of
the object re-read from R2, so a silent truncation cannot pass.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import boto3
from dotenv import load_dotenv

REQUIRED = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def load_env(path: str | Path = ".env") -> dict:
    """Load .env and return the R2 settings, raising with names only if incomplete."""
    load_dotenv(path, override=False)
    import os

    missing = [k for k in REQUIRED if not (os.getenv(k) or "").strip()]
    if missing:
        raise RuntimeError(f"missing R2 settings in environment/.env: {', '.join(missing)}")
    return {k: os.environ[k] for k in REQUIRED}


def config_report(path: str | Path = ".env") -> dict[str, str]:
    """Which R2 names are present. Never returns values."""
    from dotenv import dotenv_values

    v = dotenv_values(path)
    return {k: ("set" if (v.get(k) or "").strip() else "MISSING") for k in REQUIRED}


def client(env: dict | None = None):
    env = env or load_env()
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(path: str | Path, key: str, env: dict | None = None) -> dict:
    """Upload one file, then read the object back and compare sha256."""
    env = env or load_env()
    path = Path(path)
    local = sha256_file(path)
    client(env).upload_file(str(path), env["R2_BUCKET"], key)
    remote = hashlib.sha256(client(env).get_object(Bucket=env["R2_BUCKET"], Key=key)["Body"].read()).hexdigest()
    return {"key": key, "bytes": path.stat().st_size, "sha256": local,
            "remote_sha256": remote, "match": local == remote}


def download(key: str, path: str | Path, env: dict | None = None) -> dict:
    """Download one object and verify that the local copy hashes to the object's."""
    env = env or load_env()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client(env).download_file(env["R2_BUCKET"], key, str(path))
    local = sha256_file(path)
    remote = hashlib.sha256(client(env).get_object(Bucket=env["R2_BUCKET"], Key=key)["Body"].read()).hexdigest()
    return {"key": key, "path": str(path), "bytes": path.stat().st_size,
            "sha256": local, "remote_sha256": remote, "match": local == remote}


def list_objects(prefix: str = "", env: dict | None = None, limit: int = 50) -> list[dict]:
    env = env or load_env()
    out = []
    for page in client(env).get_paginator("list_objects_v2").paginate(
            Bucket=env["R2_BUCKET"], Prefix=prefix):
        for o in page.get("Contents", []):
            out.append({"key": o["Key"], "bytes": o["Size"]})
            if len(out) >= limit:
                return out
    return out
