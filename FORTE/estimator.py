"""Riemannian translational stiffness estimator on SPD(3).

Online estimation of a 3x3 symmetric positive-definite stiffness matrix
using the affine-invariant exponential map retraction.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


_DEFAULT_STIFFNESS_SCALE: Dict[str, float] = {
    "HIGH": 200.0,
    "MEDIUM": 50.0,
    "LOW": 10.0,
}

_AXIS_ORDER = ["x", "y", "z"]


class RiemannianStiffnessEstimator:
    """Online translational stiffness estimator using the SPD(3) exponential map."""

    def __init__(
        self,
        K_init: np.ndarray,
        eta: float = 0.01,
        min_eigenvalue: float = 0.1,
        max_update_norm: float = 5.0,
        max_exp_argument: float = 20.0,
    ) -> None:
        K_init = np.asarray(K_init, dtype=np.float64)
        if K_init.shape != (3, 3):
            raise ValueError(f"K_init must be (3,3), got {K_init.shape}")
        K_init = 0.5 * (K_init + K_init.T)
        vals = np.linalg.eigvalsh(K_init)
        if np.any(vals <= 0):
            raise ValueError(f"K_init not positive definite (min eigenvalue={vals.min():.6e})")
        self.eta = float(eta)
        self.min_eigenvalue = float(min_eigenvalue)
        self.max_update_norm = float(max_update_norm)
        self.max_exp_argument = float(max_exp_argument)
        self.K = self._project_spd(K_init)

    def update(self, F_meas: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
        """One SPD(3) exponential-map update."""
        F_meas = np.asarray(F_meas, dtype=np.float64).ravel()
        delta_x = np.asarray(delta_x, dtype=np.float64).ravel()
        if F_meas.shape != (3,):
            raise ValueError(f"F_meas must be (3,), got {F_meas.shape}")
        if delta_x.shape != (3,):
            raise ValueError(f"delta_x must be (3,), got {delta_x.shape}")
        if np.linalg.norm(delta_x) < 1e-12:
            return self.K.copy()

        previous = self.K.copy()
        e = F_meas - self.K @ delta_x
        grad_eucl = -np.outer(e, delta_x)
        grad_sym = 0.5 * (grad_eucl + grad_eucl.T)
        grad_nat = self.K @ grad_sym @ self.K

        k_half = self._matrix_sqrt(self.K)
        k_inv_half = self._matrix_inv_sqrt(self.K)
        tangent = k_inv_half @ grad_nat @ k_inv_half
        tangent = self._clip_symmetric_spectrum(tangent, self.max_update_norm)
        step = self._matrix_exp(-self.eta * tangent)
        candidate = k_half @ step @ k_half
        if not np.all(np.isfinite(candidate)):
            self.K = previous
            return self.K.copy()
        self.K = self._project_spd(candidate)
        return self.K.copy()

    def predict_force(self, delta_x: np.ndarray) -> np.ndarray:
        delta_x = np.asarray(delta_x, dtype=np.float64).ravel()
        return self.K @ delta_x

    def get_stiffness(self) -> np.ndarray:
        return self.K.copy()

    def get_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.K)

    def reset(self, K_init: np.ndarray) -> None:
        K_init = np.asarray(K_init, dtype=np.float64)
        if K_init.shape != (3, 3):
            raise ValueError(f"K_init must be (3,3), got {K_init.shape}")
        self.K = self._project_spd(0.5 * (K_init + K_init.T))

    def _project_spd(self, M: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
        vals = np.maximum(vals, self.min_eigenvalue)
        return vecs @ np.diag(vals) @ vecs.T

    def _matrix_sqrt(self, M: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
        vals = np.maximum(vals, self.min_eigenvalue)
        return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

    def _matrix_inv_sqrt(self, M: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
        vals = np.maximum(vals, self.min_eigenvalue)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    def _matrix_exp(self, A: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
        vals = np.clip(vals, -self.max_exp_argument, self.max_exp_argument)
        return vecs @ np.diag(np.exp(vals)) @ vecs.T

    def _clip_symmetric_spectrum(self, M: np.ndarray, max_abs: float) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
        vals = np.clip(vals, -max_abs, max_abs)
        return vecs @ np.diag(vals) @ vecs.T

    @staticmethod
    def from_vlm_prior(
        stiffness_labels: Dict[str, str],
        eta: float = 0.01,
        min_eigenvalue: float = 0.1,
        scale: Optional[Dict[str, float]] = None,
    ) -> RiemannianStiffnessEstimator:
        sc = {**_DEFAULT_STIFFNESS_SCALE, **(scale or {})}
        diag_vals = []
        for axis in _AXIS_ORDER:
            label = stiffness_labels.get(axis, "MEDIUM").upper()
            if label not in sc:
                raise ValueError(f"Unknown stiffness label '{label}' for axis '{axis}'")
            diag_vals.append(sc[label])
        return RiemannianStiffnessEstimator(
            K_init=np.diag(diag_vals).astype(np.float64),
            eta=eta,
            min_eigenvalue=min_eigenvalue,
        )
