"""
Unit tests for VLM physics parser (mock-based, no API calls).
"""

import numpy as np
import pytest

from force_coral.perception.vlm_interface import (
    PhysicsConfig,
    TaskPhysicsParser,
    default_physics_config,
)
from force_coral.dynamics.estimator import RiemannianStiffnessEstimator


# ---------------------------------------------------------------------------
# Test: PhysicsConfig basics
# ---------------------------------------------------------------------------

class TestPhysicsConfig:
    def test_default_config(self):
        cfg = default_physics_config()
        assert cfg.stiffness_prior["x"] == "MEDIUM"
        assert len(cfg.constraints) == 1
        assert np.allclose(cfg.task_frame, np.eye(3))
        assert cfg.task_frame_euler == [0.0, 0.0, 0.0]

    def test_task_frame_derived_from_euler(self):
        cfg = PhysicsConfig(
            stiffness_prior={"x": "HIGH"},
            constraints=[],
            task_frame_euler=[90.0, 0.0, 0.0],
            task_frame=None,  # should be derived
        )
        # 90-degree rotation about x: y→z, z→-y
        assert cfg.task_frame is not None
        assert cfg.task_frame.shape == (3, 3)
        # Check that z-axis rotated correctly
        assert abs(cfg.task_frame[1, 2] - (-1.0)) < 1e-6 or abs(cfg.task_frame[2, 1] - 1.0) < 1e-6

    def test_to_dict(self):
        cfg = default_physics_config()
        d = cfg.to_dict()
        assert "stiffness" in d
        assert "constraints" in d
        assert "task_frame_euler" in d


# ---------------------------------------------------------------------------
# Test: Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_valid_response(self):
        raw = '''Some text before
```json
{
  "stiffness": {
    "x": "HIGH",
    "y": "LOW",
    "z": "MEDIUM",
    "rx": "LOW",
    "ry": "MEDIUM",
    "rz": "HIGH"
  },
  "constraints": ["force_z < 10.0", "torque_x < 2.0"],
  "task_frame_euler": [0.0, 0.0, 45.0]
}
```
Some text after.'''
        cfg = TaskPhysicsParser._parse_response(raw)
        assert cfg.stiffness_prior["x"] == "HIGH"
        assert cfg.stiffness_prior["y"] == "LOW"
        assert cfg.stiffness_prior["rz"] == "HIGH"
        assert len(cfg.constraints) == 2
        assert cfg.task_frame_euler == [0.0, 0.0, 45.0]
        assert cfg.task_frame.shape == (3, 3)

    def test_missing_axes_default_medium(self):
        raw = '''```json
{
  "stiffness": {"x": "HIGH"},
  "constraints": [],
  "task_frame_euler": [0, 0, 0]
}
```'''
        cfg = TaskPhysicsParser._parse_response(raw)
        assert cfg.stiffness_prior["x"] == "HIGH"
        assert cfg.stiffness_prior["y"] == "MEDIUM"
        assert cfg.stiffness_prior["z"] == "MEDIUM"
        assert cfg.stiffness_prior["rx"] == "MEDIUM"

    def test_no_json_block_raises(self):
        raw = "No JSON here, just plain text."
        with pytest.raises(ValueError, match="No JSON block found"):
            TaskPhysicsParser._parse_response(raw)

    def test_invalid_json_raises(self):
        # Syntactically broken JSON that still matches the regex pattern
        raw = '```json\n{"x": broken}\n```'
        with pytest.raises(ValueError, match="Invalid JSON"):
            TaskPhysicsParser._parse_response(raw)

    def test_case_normalization(self):
        raw = '''```json
{
  "stiffness": {"x": "high", "y": "Low", "z": "Medium"},
  "constraints": [],
  "task_frame_euler": [0, 0, 0]
}
```'''
        cfg = TaskPhysicsParser._parse_response(raw)
        assert cfg.stiffness_prior["x"] == "HIGH"
        assert cfg.stiffness_prior["y"] == "LOW"
        assert cfg.stiffness_prior["z"] == "MEDIUM"

    def test_bad_euler_defaults_to_zero(self):
        raw = '''```json
{
  "stiffness": {"x": "HIGH"},
  "constraints": [],
  "task_frame_euler": "invalid"
}
```'''
        cfg = TaskPhysicsParser._parse_response(raw)
        assert cfg.task_frame_euler == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Test: Integration with estimator factory
# ---------------------------------------------------------------------------

class TestEstimatorIntegration:
    def test_physics_config_to_estimator(self):
        """PhysicsConfig → from_vlm_prior → valid estimator."""
        raw = '''```json
{
  "stiffness": {
    "x": "HIGH", "y": "LOW", "z": "MEDIUM",
    "rx": "LOW", "ry": "LOW", "rz": "MEDIUM"
  },
  "constraints": ["force_z < 10.0"],
  "task_frame_euler": [0, 0, 0]
}
```'''
        cfg = TaskPhysicsParser._parse_response(raw)
        est = RiemannianStiffnessEstimator.from_vlm_prior(cfg.stiffness_prior)
        K = est.get_stiffness()
        assert K.shape == (6, 6)
        assert K[0, 0] == 1000.0  # HIGH
        assert K[1, 1] == 10.0    # LOW
        assert K[2, 2] == 100.0   # MEDIUM
        # Verify SPD
        assert np.all(np.linalg.eigvalsh(K) > 0)


# ---------------------------------------------------------------------------
# Test: No client available
# ---------------------------------------------------------------------------

class TestNoClient:
    def test_parse_task_raises_without_client(self):
        parser = TaskPhysicsParser(client=None)
        parser.client = None  # force no client
        with pytest.raises(RuntimeError, match="No OpenAI client"):
            from PIL import Image
            img = Image.new("RGB", (64, 64))
            parser.parse_task(img, "test task")
