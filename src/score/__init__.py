"""Score: residual, clip score, abstention and conformal thresholds (Phase 3)."""
from src.score.prereg import PreregError, verify_prereg
from src.score.verdict import (auroc, auroc_detect, conformal_quantile,
                               depicted_axis_world, fit_nuisances, score_clip, spearman)

__all__ = ["PreregError", "verify_prereg", "auroc", "conformal_quantile",
           "auroc_detect", "depicted_axis_world", "fit_nuisances", "score_clip",
           "spearman"]
