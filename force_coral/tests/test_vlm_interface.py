"""Unit tests for the bounded Phase-1 VLM semantic package."""

import numpy as np
import pytest

from force_coral.dynamics.estimator import RiemannianStiffnessEstimator
from force_coral.perception.vlm_interface import (
    ForceBand,
    PhysicsConfig,
    TaskPhysicsParser,
    default_physics_config,
)


class TestPhysicsConfig:
    def test_default_config_matches_wall_lift_defaults(self):
        config = default_physics_config()
        assert config.stiffness_prior == {"x": "HIGH", "y": "LOW", "z": "MEDIUM"}
        assert config.force_band == ForceBand(lower=2.5, upper=12.0)
        assert config.goal["target_height"] == pytest.approx(0.50)
        assert np.allclose(config.task_frame, np.eye(3))

    def test_task_frame_is_derived_from_euler_when_missing(self):
        config = PhysicsConfig(
            stiffness_prior={"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
            force_band=ForceBand(lower=1.0, upper=6.0),
            task_frame_euler=[90.0, 0.0, 0.0],
            task_frame=None,
            goal={"target_height": 0.4},
            cost_weights={"task_height": 10.0},
            recovery_hints=["re_establish_contact"],
        )
        assert config.task_frame.shape == (3, 3)
        assert np.allclose(config.task_frame @ config.task_frame.T, np.eye(3), atol=1e-6)

    def test_to_dict_contains_phase1_fields(self):
        config = default_physics_config()
        payload = config.to_dict()
        assert payload["force_band"] == {"lower": 2.5, "upper": 12.0}
        assert "goal" in payload
        assert "cost_weights" in payload
        assert "recovery_hints" in payload


class TestResponseParsing:
    def test_valid_response_is_parsed(self):
        raw = """```json
{
  "stiffness": {"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
  "force_band": {"lower": 3.0, "upper": 8.5},
  "task_frame_euler": [0.0, 0.0, 90.0],
  "goal": {"target_height": 0.55},
  "cost_weights": {
    "task_height": 20.0,
    "task_contact": 5.0,
    "task_pose": 2.5,
    "energy": 0.75,
    "force_upper": 60.0,
    "force_lower": 30.0
  },
  "recovery_hints": ["re_establish_contact", "reduce_normal_force_if_stalled"]
}
```"""
        config = TaskPhysicsParser._parse_response(raw)
        assert config.stiffness_prior["x"] == "HIGH"
        assert config.force_band.lower == pytest.approx(3.0)
        assert config.force_band.upper == pytest.approx(8.5)
        assert config.goal["target_height"] == pytest.approx(0.55)
        assert config.cost_weights["force_lower"] == pytest.approx(30.0)
        assert "reduce_normal_force_if_stalled" in config.recovery_hints

    def test_missing_fields_fall_back_to_defaults(self):
        raw = """```json
{
  "stiffness": {"x": "low"}
}
```"""
        config = TaskPhysicsParser._parse_response(raw)
        defaults = default_physics_config()
        assert config.stiffness_prior == {"x": "LOW", "y": "MEDIUM", "z": "MEDIUM"}
        assert config.force_band == defaults.force_band
        assert config.goal == defaults.goal
        assert config.cost_weights == defaults.cost_weights

    def test_bad_force_band_is_repaired(self):
        raw = """```json
{
  "stiffness": {"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
  "force_band": {"lower": 9.0, "upper": 4.0}
}
```"""
        config = TaskPhysicsParser._parse_response(raw)
        assert config.force_band.upper > config.force_band.lower

    def test_bad_euler_defaults_to_identity(self):
        raw = """```json
{
  "stiffness": {"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
  "task_frame_euler": "invalid"
}
```"""
        config = TaskPhysicsParser._parse_response(raw)
        assert config.task_frame_euler == [0.0, 0.0, 0.0]
        assert np.allclose(config.task_frame, np.eye(3))

    def test_no_json_block_raises(self):
        with pytest.raises(ValueError, match="No JSON block found"):
            TaskPhysicsParser._parse_response("not json")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            TaskPhysicsParser._parse_response("```json\n{\"broken\": nope}\n```")


class TestIntegration:
    def test_config_can_seed_estimator(self):
        raw = """```json
{
  "stiffness": {"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
  "force_band": {"lower": 2.0, "upper": 10.0}
}
```"""
        config = TaskPhysicsParser._parse_response(raw)
        estimator = RiemannianStiffnessEstimator.from_vlm_prior(config.stiffness_prior)
        stiffness = estimator.get_stiffness()
        assert stiffness.shape == (3, 3)
        assert np.all(np.linalg.eigvalsh(stiffness) > 0.0)

    def test_parse_task_raises_without_client(self):
        parser = TaskPhysicsParser(client=None)
        parser.client = None
        with pytest.raises(RuntimeError, match="No OpenAI client"):
            from PIL import Image

            parser.parse_task(Image.new("RGB", (32, 32)), "lift the box")
