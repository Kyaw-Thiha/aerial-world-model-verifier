"""The gain test: per-clip regression of depicted lean on required lean.

Public API
----------
``Clip`` / ``load_clip`` / ``save_clip``   the shared ``verdict_clip.npz`` schema
``required_lean`` / ``depicted_lean_from_gravity``   the two sides of the test
``FitOptions`` / ``fit_gain`` / ``GainResult``       the regression and its intervals
``assess``                                  pre-registered abstention
``fit_bands`` / ``flag``                    conformal bands and decision rules
"""
from .lean import (RequiredLean, depicted_lean_from_gravity, lean_to_roll_pitch,
                   principal_spread_deg, required_lean, roll_pitch_to_lean,
                   world_to_yaw_frame)
from .schema import CONDITIONS, Clip, load_clip, save_clip, validate_clip
from .dense import DenseRequired, dense_required_lean, truth_lean
from .coherence import CoherenceResult, coherence
from .deming import DemingFit, deming_gain
from .fit import FitOptions, GainResult, fit_gain
from .abstain import Abstention, assess, gate_kept_from_conf, validation_status
from .decide import (Bands, JointBand, ScalarBand, conformal_band, fit_bands,
                     fit_joint_band, flag)

__all__ = [
    "RequiredLean", "depicted_lean_from_gravity", "lean_to_roll_pitch",
    "principal_spread_deg", "required_lean", "roll_pitch_to_lean",
    "world_to_yaw_frame", "CONDITIONS", "Clip", "load_clip", "save_clip",
    "validate_clip", "DenseRequired", "dense_required_lean", "truth_lean",
    "CoherenceResult", "coherence", "DemingFit", "deming_gain", "FitOptions", "GainResult", "fit_gain", "Abstention", "assess", "validation_status",
    "gate_kept_from_conf", "Bands", "JointBand", "ScalarBand", "conformal_band",
    "fit_bands", "fit_joint_band", "flag",
]
