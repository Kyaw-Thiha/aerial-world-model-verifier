"""Total least squares (Deming) for the gain, as a comparison with OLS.

The gain regression has error on both sides: the depicted lean is an estimator output and
the required lean is differentiated (more so on the 12 Hz frame grid than on dense truth).
OLS is attenuated toward zero when the regressor is noisy.  Deming corrects it with the
error-variance ratio ``lambda = sigma_y^2 / sigma_x^2``.

Implementation.  The drag and intercept are first residualised out with OLS (they are
nuisances, not the tested quantity); the scalar gain is then the Deming slope through the
origin of the residualised stacked x/y data::

    beta = (S_yy - lambda*S_xx + sqrt((S_yy - lambda*S_xx)^2 + 4*lambda*S_xy^2))
           / (2*S_xy)

As ``lambda -> infinity`` (no x error) this tends to the OLS slope ``S_xy / S_xx``, which
is the check used in the unit tests.  ``lambda`` is supplied per clip from measured error
variances, never tuned on verdict outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DemingFit:
    gain: float
    gain_ols: float
    drag: float
    intercept: tuple[float, float]
    lam: float
    sigma_x: float
    sigma_y: float


def _ols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, y, rcond=None)[0]


def deming_gain(l_dep: np.ndarray, l_req: np.ndarray, w: np.ndarray,
                sigma_x: float, sigma_y: float | None = None,
                intercept: bool = True) -> DemingFit:
    """Scalar Deming gain after residualising the drag and intercept."""
    l_dep = np.asarray(l_dep, dtype=np.float64)
    l_req = np.asarray(l_req, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(l_dep)
    ex_parts, ey_parts = [], []
    drags, ints = [], []
    for axis in range(2):
        cols = [w[:, axis]]
        if intercept:
            cols.append(np.ones(n))
        Z = np.column_stack(cols)
        cy = _ols(Z, l_dep[:, axis])
        cx = _ols(Z, l_req[:, axis])
        ey_parts.append(l_dep[:, axis] - Z @ cy)
        ex_parts.append(l_req[:, axis] - Z @ cx)
        drags.append(float(cy[0]))
        ints.append(float(cy[1]) if intercept else 0.0)
    ex = np.concatenate(ex_parts)
    ey = np.concatenate(ey_parts)
    S_xx = float(ex @ ex)
    S_yy = float(ey @ ey)
    S_xy = float(ex @ ey)
    gain_ols = S_xy / S_xx if S_xx > 0 else float("nan")
    if sigma_y is None:
        sigma_y = float(np.sqrt(S_yy / max(len(ey), 1)))
    else:
        sigma_y = float(sigma_y)
    if not np.isfinite(sigma_x) or sigma_x <= 0:
        sigma_x = 1e-6
    lam = (sigma_y ** 2) / (sigma_x ** 2)
    disc = (S_yy - lam * S_xx) ** 2 + 4.0 * lam * S_xy ** 2
    if S_xy == 0 or not np.isfinite(disc):
        gain = 0.0
    else:
        gain = (S_yy - lam * S_xx + np.sqrt(disc)) / (2.0 * S_xy)
    return DemingFit(gain=float(gain), gain_ols=float(gain_ols),
                     drag=float(np.mean(drags)), intercept=(ints[0], ints[1]),
                     lam=float(lam), sigma_x=float(sigma_x), sigma_y=sigma_y)
