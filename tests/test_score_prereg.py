"""The pre-registration guard: it must reject a missing, untracked or dirty spec."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.score import PreregError, verify_prereg

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/verdict_preregistration.yaml"


def test_committed_prereg_verifies():
    info = verify_prereg(SPEC)
    assert info["sha256"] and len(info["sha256"]) == 64
    doc = info["doc"]
    for key in ("mode", "nuisances", "abstention", "thresholds", "success_criteria"):
        assert key in doc, key
    assert doc["nuisances"]["uptilt_offset_deg"]["count"] == 1
    assert doc["success_criteria"]["auroc_above"] == 0.9


def test_missing_spec_is_rejected(tmp_path):
    with pytest.raises(PreregError, match="missing"):
        verify_prereg(tmp_path / "nope.yaml")


def test_untracked_spec_is_rejected(tmp_path):
    p = tmp_path / "spec.yaml"
    shutil.copy(SPEC, p)
    with pytest.raises(PreregError, match="outside the repository"):
        verify_prereg(p)


def test_allow_dirty_only_relaxes_the_modified_check():
    """A dirty spec is refused for a scoring run; allow_dirty exists for real-clip dry runs."""
    info = verify_prereg(SPEC, allow_dirty=True)
    assert info["sha256"]
    if info["dirty"]:
        with pytest.raises(PreregError, match="uncommitted changes"):
            verify_prereg(SPEC)
