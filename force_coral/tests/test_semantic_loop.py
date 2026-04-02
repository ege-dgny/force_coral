"""Tests for the bounded semantic loop, task monitor, and artifacts."""

from __future__ import annotations

import os

import numpy as np

from force_coral.controllers.forte_support import ArtifactManager, WallLiftTaskMonitor
from force_coral.perception.semantic_manager import SemanticManager
from force_coral.perception.vlm_interface import default_physics_config


class TestSemanticManager:
    def test_periodic_and_event_reviews(self):
        manager = SemanticManager(review_interval=5)
        manager.initialize(task_prompt="lift the box")
        assert manager.should_review(5, {"stall": False, "drop": False, "contact_lost": False, "repeated_over_force": False})
        assert manager.should_review(2, {"stall": True, "drop": False, "contact_lost": False, "repeated_over_force": False})
        assert not manager.should_review(2, {"stall": False, "drop": False, "contact_lost": False, "repeated_over_force": False})

    def test_drop_revision_increases_lower_force(self):
        manager = SemanticManager(review_interval=10)
        config = manager.initialize(task_prompt="lift the box")
        before = config.force_band.lower
        revision = manager.revise(
            monitor_status={"drop": True, "contact_lost": False, "reason": "drop"},
            recent_metrics={"box_height": 0.12},
        )
        assert revision.recovery_mode == "re_establish_contact"
        assert manager.active_config.force_band.lower >= before

    def test_stall_revision_reduces_upper_force_and_raises_task_weight(self):
        manager = SemanticManager(review_interval=10)
        config = manager.initialize(task_prompt="lift the box")
        before_upper = config.force_band.upper
        before_height_weight = config.cost_weights["task_height"]
        revision = manager.revise(
            monitor_status={"stall": True, "reason": "stall"},
            recent_metrics={"box_height": 0.15},
        )
        assert revision.recovery_mode == "reduce_normal_force_if_stalled"
        assert manager.active_config.force_band.upper <= before_upper
        assert manager.active_config.cost_weights["task_height"] > before_height_weight


class TestWallLiftTaskMonitor:
    def test_detects_under_force_and_drop(self):
        monitor = WallLiftTaskMonitor(target_height=0.50, force_lower=2.5, force_upper=12.0)
        first = monitor.update(box_height=0.10, normal_force=3.0, wall_contact=True)
        second = monitor.update(box_height=0.08, normal_force=0.5, wall_contact=False)
        assert not first["drop"]
        assert second["drop"]
        assert second["contact_lost"]
        assert second["regime"] == "under-force"

    def test_detects_stall_and_repeated_over_force(self):
        monitor = WallLiftTaskMonitor(
            target_height=0.50,
            force_lower=2.5,
            force_upper=12.0,
            stall_window=3,
            over_force_window=2,
        )
        stalled = None
        for _ in range(3):
            stalled = monitor.update(box_height=0.10, normal_force=5.0, wall_contact=True)
        assert stalled["stall"]

        over = None
        for _ in range(2):
            over = monitor.update(box_height=0.11, normal_force=20.0, wall_contact=True)
        assert over["repeated_over_force"]
        assert over["regime"] == "over-force"

    def test_detects_success(self):
        monitor = WallLiftTaskMonitor(target_height=0.50, force_lower=2.5, force_upper=12.0)
        status = monitor.update(box_height=0.52, normal_force=6.0, wall_contact=True)
        assert status["success"]

    def test_requires_wall_contact_for_success(self):
        monitor = WallLiftTaskMonitor(target_height=0.50, force_lower=2.5, force_upper=12.0)
        status = monitor.update(box_height=0.52, normal_force=6.0, wall_contact=False)
        assert not status["success"]


class TestArtifactManager:
    def test_finalize_writes_logs_plots_and_video(self, tmp_path):
        artifacts = ArtifactManager(str(tmp_path), save_video=True, overlay=True)
        record = {
            "step": 0,
            "box_height": 0.12,
            "measured_force_normal": 4.0,
            "predicted_force_normal": 3.5,
            "force_band_lower": 2.5,
            "force_band_upper": 12.0,
            "cost_energy": 1.0,
            "cost_total": 3.0,
            "regime": "in-band",
            "success": False,
            "drop": False,
            "stall": False,
        }
        artifacts.log_step(record)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        artifacts.add_frame(frame, record)

        paths = artifacts.finalize()
        assert os.path.exists(paths["json"])
        assert os.path.exists(paths["csv"])
        assert os.path.exists(paths["summary"])
        for plot_name in ("force_band.png", "height_progress.png", "costs.png"):
            assert os.path.exists(tmp_path / plot_name)
        assert os.path.exists(paths["video"])
        assert os.path.getsize(paths["video"]) > 0


def test_default_semantic_config_stays_phase1_bounded():
    config = default_physics_config()
    assert config.goal == {"target_height": 0.50}
    assert sorted(config.cost_weights.keys()) == sorted(
        ["task_height", "task_contact", "task_pose", "energy", "force_upper", "force_lower"]
    )
