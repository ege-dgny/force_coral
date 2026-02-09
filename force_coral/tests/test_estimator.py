"""
Unit tests for RiemannianStiffnessEstimator.

Tests cover:
1. SPD preservation under random updates
2. Convergence toward known ground-truth stiffness
3. Zero-error stability (no drift when prediction is exact)
4. Numerical edge cases
5. Factory method from_vlm_prior
"""

import numpy as np
import pytest

from force_coral.dynamics.estimator import RiemannianStiffnessEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(n: int = 6, seed: int = 42) -> np.ndarray:
    """Generate a random SPD matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n)


def _is_spd(M: np.ndarray, tol: float = 0.0) -> bool:
    """Check if M is symmetric positive definite with eigenvalues > tol."""
    if not np.allclose(M, M.T, atol=1e-12):
        return False
    return float(np.linalg.eigvalsh(M).min()) > tol


# ---------------------------------------------------------------------------
# Test: SPD preservation
# ---------------------------------------------------------------------------

class TestSPDPreservation:
    """After many updates with random data, K must remain SPD."""

    def test_random_updates_preserve_spd(self):
        rng = np.random.default_rng(123)
        K_init = np.diag([100.0, 200.0, 50.0, 10.0, 20.0, 30.0])
        est = RiemannianStiffnessEstimator(K_init, eta=0.001, min_eigenvalue=0.1)

        for _ in range(200):
            F = rng.standard_normal(6) * 10.0
            dx = rng.standard_normal(6) * 0.1
            est.update(F, dx)
            K = est.get_stiffness()
            assert _is_spd(K, tol=0.0), f"K not SPD: eigenvalues={np.linalg.eigvalsh(K)}"

    def test_eigenvalue_floor_respected(self):
        rng = np.random.default_rng(456)
        est = RiemannianStiffnessEstimator(
            np.eye(6) * 1.0, eta=0.1, min_eigenvalue=0.5
        )
        for _ in range(100):
            F = rng.standard_normal(6) * 50.0
            dx = rng.standard_normal(6) * 0.5
            est.update(F, dx)
            evals = est.get_eigenvalues()
            assert evals.min() >= 0.5 - 1e-12, (
                f"Eigenvalue below floor: {evals.min()}"
            )


# ---------------------------------------------------------------------------
# Test: Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """Given F = K_true @ Δx + noise, estimate should approach K_true."""

    def test_convergence_to_diagonal_K(self):
        rng = np.random.default_rng(789)
        K_true = np.diag([500.0, 100.0, 200.0, 50.0, 30.0, 80.0])

        # Start from a different diagonal
        K_init = np.diag([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        est = RiemannianStiffnessEstimator(K_init, eta=0.0001, min_eigenvalue=0.1)

        errors = []
        for i in range(1000):
            dx = rng.standard_normal(6) * 0.05
            noise = rng.standard_normal(6) * 0.5
            F_meas = K_true @ dx + noise
            est.update(F_meas, dx)
            if i % 100 == 0:
                err = np.linalg.norm(est.K - K_true, ord="fro")
                errors.append(err)

        # Error should generally decrease
        assert errors[-1] < errors[0], (
            f"Did not converge: initial error={errors[0]:.2f}, final={errors[-1]:.2f}"
        )

    def test_convergence_to_full_spd_K(self):
        """Convergence with a non-diagonal ground truth."""
        rng = np.random.default_rng(321)
        # Random SPD ground truth
        A = rng.standard_normal((6, 6))
        K_true = A @ A.T + 10.0 * np.eye(6)

        K_init = np.eye(6) * 50.0
        est = RiemannianStiffnessEstimator(K_init, eta=0.00001, min_eigenvalue=0.1)

        err_start = np.linalg.norm(est.K - K_true, ord="fro")
        for _ in range(2000):
            dx = rng.standard_normal(6) * 0.02
            F_meas = K_true @ dx + rng.standard_normal(6) * 0.1
            est.update(F_meas, dx)

        err_end = np.linalg.norm(est.K - K_true, ord="fro")
        assert err_end < err_start, (
            f"Did not converge: start={err_start:.2f}, end={err_end:.2f}"
        )


# ---------------------------------------------------------------------------
# Test: Zero-error stability
# ---------------------------------------------------------------------------

class TestZeroErrorStability:
    """When F_meas = K @ Δx exactly, K should not change."""

    def test_no_drift_on_perfect_prediction(self):
        K_init = np.diag([300.0, 150.0, 75.0, 40.0, 20.0, 10.0])
        est = RiemannianStiffnessEstimator(K_init.copy(), eta=0.01, min_eigenvalue=0.1)

        rng = np.random.default_rng(111)
        for _ in range(50):
            dx = rng.standard_normal(6) * 0.1
            F_meas = K_init @ dx  # perfect prediction
            est.update(F_meas, dx)

        # K should be essentially unchanged
        assert np.allclose(est.K, K_init, atol=1e-6), (
            f"K drifted: max diff={np.abs(est.K - K_init).max():.2e}"
        )


# ---------------------------------------------------------------------------
# Test: Numerical edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: large forces, near-zero delta_x, etc."""

    def test_near_zero_delta_x_no_update(self):
        """When delta_x ≈ 0, update should be a no-op."""
        K_init = np.eye(6) * 100.0
        est = RiemannianStiffnessEstimator(K_init.copy(), eta=0.01)

        dx = np.zeros(6)
        F = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.1])
        K_before = est.get_stiffness()
        est.update(F, dx)
        K_after = est.get_stiffness()
        assert np.allclose(K_before, K_after)

    def test_large_force_remains_spd(self):
        """Large F_meas should not break SPD property."""
        est = RiemannianStiffnessEstimator(np.eye(6) * 50.0, eta=0.001)
        F_large = np.array([1e4, -1e4, 1e4, -1e3, 1e3, -1e3])
        dx = np.array([0.01, 0.02, -0.01, 0.005, -0.005, 0.01])
        for _ in range(10):
            est.update(F_large, dx)
        assert _is_spd(est.K, tol=0.0)

    def test_invalid_K_init_raises(self):
        """Non-SPD initial K should raise ValueError."""
        with pytest.raises(ValueError, match="not positive semi-definite"):
            RiemannianStiffnessEstimator(np.diag([1, 1, 1, 1, 1, -1]))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="must be \\(6,6\\)"):
            RiemannianStiffnessEstimator(np.eye(3))

    def test_wrong_input_shapes(self):
        est = RiemannianStiffnessEstimator(np.eye(6) * 10.0)
        with pytest.raises(ValueError):
            est.update(np.zeros(3), np.zeros(6))
        with pytest.raises(ValueError):
            est.update(np.zeros(6), np.zeros(4))


# ---------------------------------------------------------------------------
# Test: Factory method
# ---------------------------------------------------------------------------

class TestFactoryMethod:
    """from_vlm_prior should create correct diagonal K."""

    def test_default_scale(self):
        labels = {"x": "HIGH", "y": "LOW", "z": "MEDIUM",
                  "rx": "LOW", "ry": "HIGH", "rz": "MEDIUM"}
        est = RiemannianStiffnessEstimator.from_vlm_prior(labels)
        K = est.get_stiffness()
        expected_diag = [1000.0, 10.0, 100.0, 10.0, 1000.0, 100.0]
        assert np.allclose(np.diag(K), expected_diag)
        # Off-diagonal should be zero
        assert np.allclose(K - np.diag(np.diag(K)), 0.0)

    def test_custom_scale(self):
        labels = {"x": "HIGH", "y": "LOW"}
        scale = {"HIGH": 500.0, "MEDIUM": 50.0, "LOW": 5.0}
        est = RiemannianStiffnessEstimator.from_vlm_prior(labels, scale=scale)
        K = est.get_stiffness()
        assert K[0, 0] == 500.0
        assert K[1, 1] == 5.0
        # Missing axes default to MEDIUM
        assert K[2, 2] == 50.0

    def test_missing_axes_default_medium(self):
        labels = {"x": "HIGH"}  # only x specified
        est = RiemannianStiffnessEstimator.from_vlm_prior(labels)
        K = est.get_stiffness()
        assert K[0, 0] == 1000.0
        for i in range(1, 6):
            assert K[i, i] == 100.0  # MEDIUM default

    def test_unknown_label_raises(self):
        labels = {"x": "ULTRA"}
        with pytest.raises(ValueError, match="Unknown stiffness label"):
            RiemannianStiffnessEstimator.from_vlm_prior(labels)

    def test_is_spd_after_creation(self):
        labels = {"x": "LOW", "y": "LOW", "z": "LOW",
                  "rx": "LOW", "ry": "LOW", "rz": "LOW"}
        est = RiemannianStiffnessEstimator.from_vlm_prior(labels)
        assert _is_spd(est.K, tol=0.0)


# ---------------------------------------------------------------------------
# Test: Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_changes_K(self):
        est = RiemannianStiffnessEstimator(np.eye(6) * 10.0)
        K_new = np.eye(6) * 500.0
        est.reset(K_new)
        assert np.allclose(est.K, K_new)

    def test_reset_enforces_spd(self):
        est = RiemannianStiffnessEstimator(np.eye(6) * 10.0, min_eigenvalue=1.0)
        # Reset with a matrix that has small eigenvalues
        K_small = np.eye(6) * 0.01
        est.reset(K_small)
        assert est.get_eigenvalues().min() >= 1.0 - 1e-12
