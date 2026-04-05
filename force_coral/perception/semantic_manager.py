"""Bounded semantic supervision for the Phase-1 FORTE pipeline."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from force_coral.perception.vlm_interface import PhysicsConfig, TaskPhysicsParser, default_physics_config


@dataclasses.dataclass
class SemanticRevision:
    """Constrained update emitted by the outer semantic supervisor."""

    goal: Optional[Dict[str, Any]] = None
    force_band: Optional[Dict[str, float]] = None
    cost_weights: Optional[Dict[str, float]] = None
    recovery_mode: Optional[str] = None
    review_reason: str = "periodic"


class SemanticManager:
    """Owns one-shot semantic initialization and bounded revisions."""

    def __init__(
        self,
        parser: Optional[TaskPhysicsParser] = None,
        review_interval: int = 10,
    ) -> None:
        self.parser = parser
        self.review_interval = max(1, int(review_interval))
        self.active_config: PhysicsConfig = default_physics_config()
        self.revision_history: List[SemanticRevision] = []

    def initialize(
        self,
        *,
        image=None,
        task_prompt: str = "",
        scene_info: dict | None = None,
    ) -> PhysicsConfig:
        if image is not None and self.parser is not None:
            try:
                self.active_config = self.parser.parse_task(
                    image, task_prompt, scene_info=scene_info,
                )
            except Exception:
                self.active_config = default_physics_config()
        else:
            self.active_config = default_physics_config()
        return self.active_config

    def should_review(self, step_idx: int, monitor_status: Dict[str, Any]) -> bool:
        if step_idx > 0 and step_idx % self.review_interval == 0:
            return True
        return bool(
            monitor_status.get("stall", False)
            or monitor_status.get("drop", False)
            or monitor_status.get("contact_lost", False)
            or monitor_status.get("repeated_over_force", False)
        )

    def revise(
        self,
        *,
        monitor_status: Dict[str, Any],
        recent_metrics: Dict[str, Any],
    ) -> SemanticRevision:
        revision = SemanticRevision(review_reason=str(monitor_status.get("reason", "periodic")))
        cfg = self.active_config
        if monitor_status.get("drop", False) or monitor_status.get("contact_lost", False):
            lower = min(cfg.force_band.upper - 0.5, cfg.force_band.lower + 1.0)
            revision.force_band = {"lower": max(0.0, lower), "upper": cfg.force_band.upper}
            revision.recovery_mode = "re_establish_contact"
        elif monitor_status.get("stall", False):
            revision.force_band = {
                "lower": cfg.force_band.lower,
                "upper": max(cfg.force_band.lower + 0.5, cfg.force_band.upper - 0.5),
            }
            weights = dict(cfg.cost_weights)
            weights["task_height"] = float(weights.get("task_height", 16.0)) * 1.1
            revision.cost_weights = weights
            revision.recovery_mode = "reduce_normal_force_if_stalled"
        elif monitor_status.get("repeated_over_force", False):
            revision.force_band = {
                "lower": cfg.force_band.lower,
                "upper": max(cfg.force_band.lower + 0.5, cfg.force_band.upper - 1.0),
            }
            revision.recovery_mode = "reduce_normal_force_if_stalled"
        else:
            target_height = float(cfg.goal.get("target_height", 0.50))
            current_height = float(recent_metrics.get("box_height", 0.0))
            if current_height + 0.10 < target_height:
                revision.goal = {"target_height": target_height}
                revision.recovery_mode = "raise_subgoal"

        self.revision_history.append(revision)
        self._apply_revision(revision)
        return revision

    def _apply_revision(self, revision: SemanticRevision) -> None:
        if revision.goal:
            self.active_config.goal.update(revision.goal)
        if revision.force_band:
            self.active_config.force_band.lower = float(
                revision.force_band.get("lower", self.active_config.force_band.lower)
            )
            self.active_config.force_band.upper = float(
                revision.force_band.get("upper", self.active_config.force_band.upper)
            )
        if revision.cost_weights:
            self.active_config.cost_weights.update(
                {key: float(value) for key, value in revision.cost_weights.items()}
            )
        if revision.recovery_mode:
            hints = list(self.active_config.recovery_hints)
            if revision.recovery_mode not in hints:
                hints.append(revision.recovery_mode)
            self.active_config.recovery_hints = hints
