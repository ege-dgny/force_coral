"""
Riemannian Stiffness Estimator on SPD(6).

Implements first-order retraction on the manifold of 6x6 Symmetric Positive
Definite matrices for online contact stiffness estimation.

Theory
------
The stiffness matrix K ∈ SPD(6) maps penetration vectors Δx ∈ R^6 to contact
wrenches F ∈ R^6 via F = K·Δx.  Given measured wrench F_meas and penetration
Δx, we minimize ||F_meas - K·Δx||² on the SPD manifold using the
affine-invariant Riemannian metric.

Update law (first-order retraction):
    1. Prediction error:       e = F_meas - K·Δx
    2. Euclidean gradient:     ∇_E = -(e ⊗ Δx^T)
    3. Symmetrize:             G = 0.5·(∇_E + ∇_E^T)
    4. Natural gradient:       ∇_nat = K·G·K
    5. Retraction step:        K_new = K - η·∇_nat
    6. SPD projection:         clip eigenvalues ≥ ε

The natural gradient K·G·K is the steepest descent direction under the
affine-invariant metric on SPD manifolds.  Eigenvalue clipping projects
onto the convex cone {M : λ_min(M) ≥ ε}, guaranteeing SPD.

Implementation uses only numpy (no scipy.linalg.expm/logm) for latency <1ms
at n=6, enabling >100Hz update rates.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# Default mapping from semantic labels to stiffness values (N/m or Nm/rad)
_DEFAULT_STIFFNESS_SCALE = {
    "HIGH": 1000.0,
    "MEDIUM": 100.0,
    "LOW": 10.0,
}

# Axis ordering for the 6D wrench vector [Fx, Fy, Fz, τx, τy, τz]
_AXIS_ORDER = ["x", "y", "z", "rx", "ry", "rz"]


class RiemannianStiffnessEstimator:
    """Online stiffness estimator using first-order Riemannian retraction on SPD(6).

    Parameters
    ----------
    K_init : np.ndarray, shape (6, 6)
        Initial stiffness matrix.  Must be symmetric positive definite.
    eta : float
        Learning rate for the retraction step.
    min_eigenvalue : float
        Floor for eigenvalue clipping (SPD guarantee).
    """

    def __init__(
        self,
        K_init: np.ndarray,
        eta: float = 0.01,
        min_eigenvalue: float = 0.1,
    ) -> None:
        K_init = np.asarray(K_init, dtype=np.float64)
        if K_init.shape != (6, 6):
            raise ValueError(f"K_init must be (6,6), got {K_init.shape}")
        # Symmetrize (in case of tiny floating-point asymmetry)
        K_init = 0.5 * (K_init + K_init.T)
        vals = np.linalg.eigvalsh(K_init)
        if np.any(vals < 0):
            raise ValueError(
                f"K_init is not positive semi-definite (min eigenvalue={vals.min():.6e})"
            )
        self.K: np.ndarray = K_init.copy()
        self.eta: float = float(eta)
        self.min_eigenvalue: float = float(min_eigenvalue)
        # Ensure initial K satisfies the eigenvalue floor
        self.K = self._project_spd(self.K)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, F_meas: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
        """Perform one Riemannian retraction update.

        Parameters
        ----------
        F_meas : np.ndarray, shape (6,)
            Measured wrench [Fx, Fy, Fz, τx, τy, τz].
        delta_x : np.ndarray, shape (6,)
            Penetration vector (x_surface - x_ee) in task frame.

        Returns
        -------
        np.ndarray, shape (6, 6)
            Updated stiffness matrix.
        """
        F_meas = np.asarray(F_meas, dtype=np.float64).ravel()
        delta_x = np.asarray(delta_x, dtype=np.float64).ravel()
        if F_meas.shape != (6,):
            raise ValueError(f"F_meas must be (6,), got {F_meas.shape}")
        if delta_x.shape != (6,):
            raise ValueError(f"delta_x must be (6,), got {delta_x.shape}")

        # Skip update if delta_x is near zero (no penetration → no gradient info)
        dx_norm = np.linalg.norm(delta_x)
        if dx_norm < 1e-12:
            return self.K.copy()

        # 1. Prediction error
        e = F_meas - self.K @ delta_x  # (6,)

        # 2. Euclidean gradient of L = 0.5·||e||²  w.r.t. K
        grad_eucl = -np.outer(e, delta_x)  # (6, 6)

        # 3. Symmetrize (K is symmetric, so gradient should be too)
        grad_sym = 0.5 * (grad_eucl + grad_eucl.T)  # (6, 6)

        # 4. Natural gradient on SPD manifold (affine-invariant metric)
        grad_nat = self.K @ grad_sym @ self.K  # (6, 6)

        # 5. Retraction step (first-order: subtract in ambient space)
        K_new = self.K - self.eta * grad_nat

        # 6. Project back onto SPD cone
        self.K = self._project_spd(K_new)

        return self.K.copy()

    # ------------------------------------------------------------------
    # SPD projection
    # ------------------------------------------------------------------

    def _project_spd(self, M: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix onto SPD cone by clipping eigenvalues.

        Parameters
        ----------
        M : np.ndarray, shape (6, 6)
            Symmetric matrix (may have eigenvalues below threshold).

        Returns
        -------
        np.ndarray, shape (6, 6)
            SPD matrix with all eigenvalues ≥ self.min_eigenvalue.
        """
        # Ensure symmetry (numerical safety)
        M = 0.5 * (M + M.T)
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, self.min_eigenvalue)
        return (vecs * vals) @ vecs.T  # equivalent to vecs @ diag(vals) @ vecs.T

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_stiffness(self) -> np.ndarray:
        """Return a copy of the current stiffness matrix."""
        return self.K.copy()

    def get_eigenvalues(self) -> np.ndarray:
        """Return sorted eigenvalues of current K (ascending)."""
        return np.linalg.eigvalsh(self.K)

    def reset(self, K_init: np.ndarray) -> None:
        """Reset stiffness to a new initial value."""
        K_init = np.asarray(K_init, dtype=np.float64)
        if K_init.shape != (6, 6):
            raise ValueError(f"K_init must be (6,6), got {K_init.shape}")
        K_init = 0.5 * (K_init + K_init.T)
        self.K = self._project_spd(K_init)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_vlm_prior(
        stiffness_labels: Dict[str, str],
        eta: float = 0.01,
        min_eigenvalue: float = 0.1,
        scale: Optional[Dict[str, float]] = None,
    ) -> "RiemannianStiffnessEstimator":
        """Create an estimator from VLM-provided stiffness labels.

        Parameters
        ----------
        stiffness_labels : dict
            Mapping of axis name to stiffness level.
            Keys: any subset of {"x", "y", "z", "rx", "ry", "rz"}.
            Values: "HIGH", "MEDIUM", or "LOW".
            Missing axes default to "MEDIUM".
        eta : float
            Learning rate.
        min_eigenvalue : float
            Eigenvalue floor.
        scale : dict, optional
            Override for label → numeric mapping.
            Default: {"HIGH": 1000, "MEDIUM": 100, "LOW": 10}.

        Returns
        -------
        RiemannianStiffnessEstimator
        """
        sc = {**_DEFAULT_STIFFNESS_SCALE, **(scale or {})}
        diag_vals = []
        for axis in _AXIS_ORDER:
            label = stiffness_labels.get(axis, "MEDIUM").upper()
            if label not in sc:
                raise ValueError(
                    f"Unknown stiffness label '{label}' for axis '{axis}'. "
                    f"Expected one of {list(sc.keys())}"
                )
            diag_vals.append(sc[label])
        K_init = np.diag(diag_vals).astype(np.float64)
        return RiemannianStiffnessEstimator(
            K_init=K_init, eta=eta, min_eigenvalue=min_eigenvalue
        )
