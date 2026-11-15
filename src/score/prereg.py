"""Pre-registration guard for the Phase 3a verdict test.

Scoring an edited clip before the pre-registration is committed is the one failure this
handoff explicitly forbids, so the check is code, not discipline: the scoring script
calls :func:`verify_prereg` and stops if the file is untracked, modified, or absent from
``HEAD``. The hash of the committed file is recorded in every run's metadata.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs/verdict_preregistration.yaml"


class PreregError(RuntimeError):
    """The pre-registration is not in the state the scoring code requires."""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.STDOUT).strip()


def verify_prereg(path: str | Path = DEFAULT_PATH, allow_dirty: bool = False) -> dict:
    """Return the resolved pre-registration, or raise :class:`PreregError`.

    Checks, in order: the file exists; git tracks it; it is not modified relative to
    ``HEAD`` (unless ``allow_dirty``, used only by dry runs on real clips); and it is
    present in ``HEAD`` itself, so a run is reproducible from the commit.
    """
    p = Path(path)
    if not p.is_file():
        raise PreregError(f"pre-registration missing: {p}")
    rel_path = p.resolve()
    try:
        rel = str(rel_path.relative_to(ROOT))
    except ValueError as exc:
        raise PreregError(f"pre-registration is outside the repository: {rel_path}") from exc
    try:
        _git("ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError as exc:
        raise PreregError(f"pre-registration is not tracked by git: {rel}") from exc
    try:
        _git("cat-file", "-e", f"HEAD:{rel}")
    except subprocess.CalledProcessError as exc:
        raise PreregError(f"pre-registration is not committed in HEAD: {rel}") from exc
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0
    if dirty and not allow_dirty:
        raise PreregError(
            f"pre-registration has uncommitted changes: {rel}. Commit the amendment, "
            "or pass allow_dirty only for a real-clip dry run (never for an edit).")
    raw = p.read_bytes()
    doc = yaml.safe_load(raw)
    return {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(),
            "dirty": bool(dirty), "git_hash": _git("rev-parse", "HEAD")[:7],
            "doc": doc}
