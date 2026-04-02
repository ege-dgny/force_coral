"""Unit tests for the Phase-1 translational SPD estimator."""

import numpy as np
import pytest

from force_coral.dynamics.estimator import RiemannianStiffnessEstimator


def _is_spd(matrix: np.ndarray, *, tol: float = 0.0) -> bool:
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        return False
    return float(np.linalg.eigvalsh(matrix).min()) > tol


class TestConstruction:
    def test_requires_3x3_matrix(self):
        with pytest.raises(ValueError, match=r"must be \(3,3\)"):
            RiemannianStiffnessEstimator(np.eye(6))

    def test_requires_positive_definite_matrix(self):
        bad = np.diag([1.0, 0.0, 2.0])
        with pytest.raises(ValueError, match="not positive definite"):
            RiemannianStiffnessEstimator(bad)

    def test_initial_floor_is_respected(self):
        init = np.diag([1.0, 2.0, 3.0])
        estimator = RiemannianStiffnessEstimator(init, min_eigenvalue=0.5)
        assert np.all(estimator.get_eigenvalues() >= 0.5 - 1e-12)


class TestUpdates:
    def test_random_updates_preserve_spd(self):
        rng = np.random.default_rng(7)
        estimator = RiemannianStiffnessEstimator(
            np.diag([250.0, 120.0, 60.0]),
            eta=0.002,
            min_eigenvalue=0.5,
        )

        for _ in range(200):
            delta = rng.standard_normal(3) * 0.03
            measured_force = rng.standard_normal(3) * 4.0
            stiffness = estimator.update(measured_force, delta)
            assert _is_spd(stiffness)
            assert np.all(np.linalg.eigvalsh(stiffness) >= 0.5 - 1e-9)

    def test_zero_delta_is_noop(self):
        init = np.diag([100.0, 50.0, 25.0])
        estimator = RiemannianStiffnessEstimator(init.copy(), eta=0.01)
        before = estimator.get_stiffness()
        after = estimator.update(np.array([3.0, -1.0, 0.5]), np.zeros(3))
        assert np.allclose(before, after)

    def test_perfect_prediction_does_not_drift(self):
        init = np.diag([300.0, 80.0, 20.0])
        estimator = RiemannianStiffnessEstimator(init.copy(), eta=0.03)
        rng = np.random.default_rng(11)

        for _ in range(50):
            delta = rng.standard_normal(3) * 0.02
            measured_force = init @ delta
            estimator.update(measured_force, delta)

        assert np.allclose(estimator.get_stiffness(), init, atol=1e-6)

    def test_predict_force_matches_matrix_vector_product(self):
        init = np.diag([10.0, 20.0, 30.0])
        estimator = RiemannianStiffnessEstimator(init)
        delta = np.array([0.1, 0.2, -0.1])
        assert np.allclose(estimator.predict_force(delta), init @ delta)

    def test_reset_reprojects_to_floor(self):
        estimator = RiemannianStiffnessEstimator(np.eye(3) * 10.0, min_eigenvalue=1.0)
        estimator.reset(np.eye(3) * 0.1)
        assert np.all(estimator.get_eigenvalues() >= 1.0 - 1e-12)

    def test_large_update_stays_finite(self):
        estimator = RiemannianStiffnessEstimator(
            np.diag([1000.0, 100.0, 10.0]),
            eta=0.02,
            min_eigenvalue=0.1,
        )
        stiffness = estimator.update(
            np.array([1.0e5, -2.0e4, 5.0e3]),
            np.array([0.25, -0.10, 0.05]),
        )
        assert np.all(np.isfinite(stiffness))
        assert _is_spd(stiffness)


class TestConvergence:
    def test_moves_toward_known_diagonal_stiffness(self):
        rng = np.random.default_rng(21)
        k_true = np.diag([400.0, 60.0, 25.0])
        estimator = RiemannianStiffnessEstimator(
            np.diag([100.0, 100.0, 100.0]),
            eta=0.0002,
            min_eigenvalue=0.5,
        )

        start_error = np.linalg.norm(estimator.get_stiffness() - k_true, ord="fro")
        for _ in range(1500):
            delta = rng.standard_normal(3) * 0.03
            measured_force = k_true @ delta + rng.standard_normal(3) * 0.05
            estimator.update(measured_force, delta)

        end_error = np.linalg.norm(estimator.get_stiffness() - k_true, ord="fro")
        assert end_error < start_error


class TestFactory:
    def test_from_vlm_prior_uses_xyz_only(self):
        estimator = RiemannianStiffnessEstimator.from_vlm_prior(
            {"x": "HIGH", "y": "LOW", "z": "MEDIUM"}
        )
        stiffness = estimator.get_stiffness()
        assert np.allclose(np.diag(stiffness), [1000.0, 10.0, 100.0])
        assert np.allclose(stiffness - np.diag(np.diag(stiffness)), 0.0)

    def test_unknown_vlm_label_raises(self):
        with pytest.raises(ValueError, match="Unknown stiffness label"):
            RiemannianStiffnessEstimator.from_vlm_prior({"x": "ULTRA"})
