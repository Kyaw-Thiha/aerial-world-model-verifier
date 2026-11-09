"""Import helpers for ANWM's own action-construction code.

ANWM's modules import torch/torchvision/tqdm at module import time. On the local CPU
environment torch is absent, and the `gpu` pixi environment lacks tqdm; this helper installs a
no-op `tqdm` stub only when the real package is missing, so ANWM's code can be *called*
unmodified. Nothing about the maths is reimplemented here.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "third_party" / "anwm"


def _ensure_tqdm_stub() -> None:
    try:
        importlib.import_module("tqdm")
    except ModuleNotFoundError:
        stub = types.ModuleType("tqdm")
        stub.__spec__ = importlib.machinery.ModuleSpec("tqdm", None)

        class _NoOp:
            def __init__(self, *a, **k):
                pass

            def __call__(self, x=None, *a, **k):
                return x

            def __iter__(self):
                return iter(())

            def update(self, *a, **k):
                pass

            def set_description(self, *a, **k):
                pass

            def close(self):
                pass

        stub.tqdm = _NoOp
        sys.modules["tqdm"] = stub


def anwm_modules():
    """(TrainingDataset, get_delta_np, normalize_data, data_config) from the pinned clone."""
    if not (REPO / "anwm").is_dir():
        raise FileNotFoundError(f"ANWM clone missing at {REPO}; see docs/anwm_feasibility.md")
    _ensure_tqdm_stub()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    airvln = importlib.import_module("anwm.data.airvln")
    utils = importlib.import_module("anwm.utils")
    config = importlib.import_module("anwm.config")
    return (airvln.TrainingDataset, utils.get_delta_np, utils.normalize_data,
            config.load_yaml("config/data_config.yaml"))
