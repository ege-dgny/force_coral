"""Tests for the Phase-1 FORTE cost structure."""

import numpy as np
import pytest

from force_coral.controllers.task_geometry import compute_box_face_anchor, compute_wall_lift_task_cost
from force_coral.dynamics.estimator import RiemannianStiffnessEstimator


def compute_phase1_cost(
    *,
    delta_task: np.ndarray,
    stiffness: np.ndarray,
    force_band: tuple[float, float] = (2.5, 12.0),
    weights: dict | None = None,
    task_cost: float = 0.0,
) -> dict:
    weights = {
        "energy": 0.5,
        "force_upper": 50.0,
        "force_lower": 25.0,
        **(weights or {}),
    }
    predicted_force_task = stiffness @ delta_task
    predicted_force_normal = float(predicted_force_task[0])
    energy = float(weights["energy"]) * float(delta_task @ stiffness @ delta_task)
    upper = float(weights["force_upper"]) * max(0.0, predicted_force_normal - force_band[1]) ** 2
    lower = float(weights["force_lower"]) * max(0.0, force_band[0] - predicted_force_normal) ** 2
    total = float(task_cost) + energy + upper + lower
    return {
        "task": float(task_cost),
        "energy": energy,
        "force_upper": upper,
        "force_lower": lower,
        "predicted_force_normal": predicted_force_normal,
        "predicted_force_task": predicted_force_task,
        "total": total,
    }


class TestEnergyTerm:
    def test_stiff_wall_normal_direction_costs_more(self):
        stiffness = np.diag([1000.0, 20.0, 20.0])
        normal_cost = compute_phase1_cost(
            delta_task=np.array([0.1, 0.0, 0.0]),
            stiffness=stiffness,
            weights={"energy": 1.0, "force_upper": 0.0, "force_lower": 0.0},
        )
        upward_cost = compute_phase1_cost(
            delta_task=np.array([0.0, 0.1, 0.0]),
            stiffness=stiffness,
            weights={"energy": 1.0, "force_upper": 0.0, "force_lower": 0.0},
        )
        assert normal_cost["energy"] / upward_cost["energy"] == pytest.approx(50.0)

    def test_energy_is_quadratic_in_displacement(self):
        stiffness = np.diag([100.0, 50.0, 25.0])
        small = compute_phase1_cost(
            delta_task=np.array([0.05, 0.0, 0.0]),
            stiffness=stiffness,
            weights={"energy": 1.0, "force_upper": 0.0, "force_lower": 0.0},
        )
        large = compute_phase1_cost(
            delta_task=np.array([0.10, 0.0, 0.0]),
            stiffness=stiffness,
            weights={"energy": 1.0, "force_upper": 0.0, "force_lower": 0.0},
        )
        assert large["energy"] / small["energy"] == pytest.approx(4.0)


class TestForceBandTerms:
    def test_upper_penalty_is_zero_in_band(self):
        stiffness = np.diag([50.0, 10.0, 10.0])
        cost = compute_phase1_cost(
            delta_task=np.array([0.1, 0.0, 0.0]),
            stiffness=stiffness,
            force_band=(2.5, 12.0),
        )
        assert cost["predicted_force_normal"] == pytest.approx(5.0)
        assert cost["force_upper"] == 0.0
        assert cost["force_lower"] == 0.0

    def test_upper_penalty_activates_above_band(self):
        stiffness = np.diag([200.0, 10.0, 10.0])
        cost = compute_phase1_cost(
            delta_task=np.array([0.1, 0.0, 0.0]),
            stiffness=stiffness,
            force_band=(2.5, 12.0),
            weights={"force_upper": 2.0, "force_lower": 0.0, "energy": 0.0},
        )
        expected = 2.0 * (20.0 - 12.0) ** 2
        assert cost["force_upper"] == pytest.approx(expected)
        assert cost["force_lower"] == 0.0

    def test_lower_penalty_activates_below_band(self):
        stiffness = np.diag([10.0, 10.0, 10.0])
        cost = compute_phase1_cost(
            delta_task=np.array([0.1, 0.0, 0.0]),
            stiffness=stiffness,
            force_band=(2.5, 12.0),
            weights={"force_upper": 0.0, "force_lower": 4.0, "energy": 0.0},
        )
        expected = 4.0 * (2.5 - 1.0) ** 2
        assert cost["force_lower"] == pytest.approx(expected)
        assert cost["force_upper"] == 0.0


class TestTotalCost:
    def test_total_is_sum_of_terms(self):
        stiffness = np.diag([100.0, 40.0, 10.0])
        cost = compute_phase1_cost(
            delta_task=np.array([0.06, 0.02, 0.0]),
            stiffness=stiffness,
            task_cost=3.5,
        )
        assert cost["total"] == pytest.approx(
            cost["task"] + cost["energy"] + cost["force_upper"] + cost["force_lower"]
        )

    def test_estimator_output_feeds_cost(self):
        estimator = RiemannianStiffnessEstimator.from_vlm_prior(
            {"x": "HIGH", "y": "LOW", "z": "MEDIUM"}
        )
        rng = np.random.default_rng(4)
        for _ in range(20):
            delta = rng.standard_normal(3) * 0.01
            force = rng.standard_normal(3) * 2.0
            estimator.update(force, delta)

        stiffness = estimator.get_stiffness()
        cost = compute_phase1_cost(
            delta_task=np.array([0.03, 0.01, 0.0]),
            stiffness=stiffness,
            task_cost=1.0,
        )
        assert cost["energy"] >= 0.0
        assert cost["total"] >= 1.0
        assert np.all(np.linalg.eigvalsh(stiffness) > 0.0)


class TestObjectAwareGeometry:
    def test_contact_anchor_targets_robot_facing_box_face(self):
        box_pos = np.array([0.0, 0.0, 0.2])
        box_rot = np.eye(3)
        half_extents = np.array([0.04, 0.05, 0.06])
        anchor = compute_box_face_anchor(
            box_pos,
            box_rot,
            half_extents,
            face_axis=1,
            face_sign=-1.0,
        )
        assert np.allclose(anchor, np.array([0.0, -0.05, 0.2]))

    def test_task_cost_activates_before_force(self):
        weights = {"task_height": 14.0, "task_contact": 8.0, "task_pose": 18.0}
        far = compute_wall_lift_task_cost(
            box_top_height=0.08,
            target_height=0.50,
            wall_gap=0.10,
            eef_to_contact_distance=0.25,
            weights=weights,
        )
        near = compute_wall_lift_task_cost(
            box_top_height=0.08,
            target_height=0.50,
            wall_gap=0.02,
            eef_to_contact_distance=0.03,
            weights=weights,
        )
        assert far["pose"] > near["pose"]
        assert far["total"] > near["total"]
