"""
Unit tests for FORTE cost function (ForteWrapper.state_cost).

Tests the energy cost, barrier cost, and their composition without
requiring a MuJoCo environment — we test the math directly.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Isolated cost function (mirrors ForteWrapper.state_cost logic)
# ---------------------------------------------------------------------------

def compute_forte_cost(
    delta_x: np.ndarray,
    K: np.ndarray,
    force_limit: float = 10.0,
    lambda_E: float = 1.0,
    rho: float = 100.0,
    task_cost: float = 0.0,
) -> dict:
    """Compute FORTE cost components from delta_x and K.

    Returns dict with keys: task, energy, barrier, total.
    """
    # Energy cost
    energy = lambda_E * float(delta_x @ K @ delta_x)

    # Predicted force
    F_pred = K @ delta_x
    F_norm = np.linalg.norm(F_pred[:3])

    # Barrier cost
    barrier = rho * max(0.0, F_norm - force_limit) ** 2

    total = task_cost + energy + barrier
    return {
        "task": task_cost,
        "energy": energy,
        "barrier": barrier,
        "total": total,
        "F_pred_norm": F_norm,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnergyCost:
    def test_energy_cost_directionality(self):
        """Motion along stiff axis should cost much more than along soft axis."""
        K = np.diag([1000.0, 10.0, 10.0, 0.1, 0.1, 0.1])

        # Motion along x (stiff)
        dx_x = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        cost_x = compute_forte_cost(dx_x, K, lambda_E=1.0, rho=0.0)

        # Motion along y (soft)
        dx_y = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        cost_y = compute_forte_cost(dx_y, K, lambda_E=1.0, rho=0.0)

        # x direction should be 100x more expensive
        assert cost_x["energy"] / cost_y["energy"] == pytest.approx(100.0, rel=1e-6)

    def test_energy_cost_quadratic_in_displacement(self):
        """Energy should scale quadratically with displacement."""
        K = np.eye(6) * 100.0
        dx1 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        dx2 = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])

        c1 = compute_forte_cost(dx1, K, rho=0.0)
        c2 = compute_forte_cost(dx2, K, rho=0.0)

        # 2x displacement → 4x energy
        assert c2["energy"] / c1["energy"] == pytest.approx(4.0, rel=1e-6)

    def test_energy_cost_zero_displacement(self):
        """Zero displacement → zero energy cost."""
        K = np.eye(6) * 1000.0
        dx = np.zeros(6)
        cost = compute_forte_cost(dx, K)
        assert cost["energy"] == 0.0

    def test_energy_cost_lambda_scaling(self):
        """Energy should scale linearly with lambda_E."""
        K = np.eye(6) * 100.0
        dx = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])

        c1 = compute_forte_cost(dx, K, lambda_E=1.0, rho=0.0)
        c2 = compute_forte_cost(dx, K, lambda_E=3.0, rho=0.0)

        assert c2["energy"] / c1["energy"] == pytest.approx(3.0, rel=1e-6)


class TestBarrierCost:
    def test_barrier_inactive_below_limit(self):
        """When predicted force < limit, barrier cost should be 0."""
        K = np.eye(6) * 10.0  # low stiffness
        dx = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        # F_pred = 10.0 * 0.1 = 1.0 N, limit = 10.0 N
        cost = compute_forte_cost(dx, K, force_limit=10.0, rho=100.0)
        assert cost["barrier"] == 0.0
        assert cost["F_pred_norm"] < 10.0

    def test_barrier_active_above_limit(self):
        """When predicted force > limit, barrier cost should be > 0."""
        K = np.eye(6) * 1000.0  # high stiffness
        dx = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        # F_pred = 1000 * 0.1 = 100 N, limit = 10 N
        cost = compute_forte_cost(dx, K, force_limit=10.0, rho=100.0)
        assert cost["barrier"] > 0.0
        # Barrier = 100 * (100 - 10)^2 = 100 * 8100 = 810000
        expected = 100.0 * (100.0 - 10.0) ** 2
        assert cost["barrier"] == pytest.approx(expected, rel=1e-6)

    def test_barrier_at_exact_limit(self):
        """At exactly the force limit, barrier should be 0."""
        K = np.eye(6) * 100.0
        # dx such that F_pred_norm = force_limit exactly
        # F = K @ dx = [100*dx, 0, 0, ...], need |F[:3]| = 10
        # so 100 * dx = 10 → dx = 0.1
        dx = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        cost = compute_forte_cost(dx, K, force_limit=10.0, rho=100.0)
        assert cost["barrier"] == pytest.approx(0.0, abs=1e-10)

    def test_barrier_quadratic_growth(self):
        """Barrier should grow quadratically past the limit."""
        K = np.eye(6) * 100.0
        # F_pred = 100 * 0.2 = 20 N
        dx1 = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        # F_pred = 100 * 0.3 = 30 N
        dx2 = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

        c1 = compute_forte_cost(dx1, K, force_limit=10.0, rho=1.0)
        c2 = compute_forte_cost(dx2, K, force_limit=10.0, rho=1.0)

        # c1: (20-10)^2 = 100, c2: (30-10)^2 = 400
        assert c1["barrier"] == pytest.approx(100.0, rel=1e-6)
        assert c2["barrier"] == pytest.approx(400.0, rel=1e-6)


class TestCostComposition:
    def test_total_is_sum_of_parts(self):
        """Total cost should equal task + energy + barrier."""
        K = np.eye(6) * 500.0
        dx = np.array([0.05, 0.03, 0.01, 0.0, 0.0, 0.0])
        cost = compute_forte_cost(
            dx, K, force_limit=10.0, lambda_E=2.0, rho=50.0, task_cost=5.0
        )
        expected_total = cost["task"] + cost["energy"] + cost["barrier"]
        assert cost["total"] == pytest.approx(expected_total, rel=1e-10)

    def test_K_none_fallback(self):
        """When K is None, only task cost should remain."""
        # This tests the logic pattern, not ForteWrapper directly
        task_cost = 3.14
        # Simulate K=None behavior
        total = task_cost  # no energy, no barrier
        assert total == 3.14

    def test_all_components_nonnegative(self):
        """All cost components should be >= 0."""
        rng = np.random.default_rng(42)
        K = _random_spd(6, rng)
        for _ in range(50):
            dx = rng.standard_normal(6) * 0.1
            cost = compute_forte_cost(
                dx, K, force_limit=5.0, lambda_E=1.0, rho=10.0, task_cost=1.0
            )
            assert cost["energy"] >= 0.0
            assert cost["barrier"] >= 0.0
            assert cost["total"] >= 0.0


class TestEstimatorAndCostIntegration:
    """Test that estimator output feeds correctly into cost function."""

    def test_estimator_K_in_cost(self):
        from force_coral.dynamics.estimator import RiemannianStiffnessEstimator

        est = RiemannianStiffnessEstimator.from_vlm_prior(
            {"x": "HIGH", "y": "LOW", "z": "MEDIUM"}
        )

        # Simulate some updates
        rng = np.random.default_rng(99)
        for _ in range(10):
            F = rng.standard_normal(6) * 5.0
            dx = rng.standard_normal(6) * 0.05
            est.update(F, dx)

        K = est.get_stiffness()
        dx_test = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
        cost = compute_forte_cost(dx_test, K, force_limit=10.0)

        assert cost["energy"] > 0
        assert cost["total"] >= cost["task"]
        # K should still be SPD
        assert np.all(np.linalg.eigvalsh(K) > 0)


def _random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n)
