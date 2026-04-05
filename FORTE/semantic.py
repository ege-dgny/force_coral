"""Bounded semantic supervision with phase transitions for FORTE.

The VLM generates a sequence of TaskPhases. The SemanticManager:
1. Initializes with the VLM's phase plan (or defaults)
2. Evaluates phase triggers against live metrics each step
3. On transition: updates active_config's cost_weights, contact_strategy,
   force_band, and goal from the new phase
4. Within a phase: bounded revisions (stall → bump height weight, etc.)

The FORTE cost structure (Eq. 5) is unchanged — only its parameters rotate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from FORTE.types import (
    ContactStrategy,
    ForceBand,
    PhysicsConfig,
    SemanticRevision,
    TaskPhase,
    default_physics_config,
)
from FORTE.vlm import TaskPhysicsParser


class SemanticManager:
    """Phase-aware semantic supervisor."""

    def __init__(
        self,
        parser: Optional[TaskPhysicsParser] = None,
        review_interval: int = 10,
    ) -> None:
        self.parser = parser
        self.review_interval = max(1, int(review_interval))
        self.active_config: PhysicsConfig = default_physics_config()
        self.revision_history: List[SemanticRevision] = []
        self.phase_index: int = 0

    @property
    def current_phase(self) -> TaskPhase:
        return self.active_config.phases[self.phase_index]

    def initialize(
        self,
        *,
        image: Any = None,
        task_prompt: str = "",
        scene_info: Optional[Dict[str, Any]] = None,
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
        self.phase_index = 0
        self._apply_phase(self.active_config.phases[0])
        return self.active_config

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def check_phase_transition(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Check if next phase's trigger is met. Returns phase name or None."""
        phases = self.active_config.phases
        if self.phase_index >= len(phases) - 1:
            return None  # already at last phase
        next_phase = phases[self.phase_index + 1]
        if _evaluate_trigger(next_phase.trigger, metrics):
            self.phase_index += 1
            self._apply_phase(next_phase)
            return next_phase.name
        return None

    def _apply_phase(self, phase: TaskPhase) -> None:
        """Set active_config's per-phase fields from a TaskPhase."""
        cfg = self.active_config
        cfg.cost_weights = dict(phase.cost_weights)
        cfg.force_band = ForceBand(
            lower=phase.force_band.lower, upper=phase.force_band.upper,
        )
        cfg.goal = dict(phase.goal)
        cfg.contact_strategy = ContactStrategy(
            approach_face_axis=phase.contact_strategy.approach_face_axis,
            approach_face_sign=phase.contact_strategy.approach_face_sign,
            contact_standoff=phase.contact_strategy.contact_standoff,
            contact_vertical_offset_scale=phase.contact_strategy.contact_vertical_offset_scale,
        )

    # ------------------------------------------------------------------
    # Within-phase revisions (reactive adjustments)
    # ------------------------------------------------------------------

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
        revision = SemanticRevision(
            review_reason=str(monitor_status.get("reason", "periodic")),
        )
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
            target = float(cfg.goal.get("target_height", 0.50))
            current = float(recent_metrics.get("box_height", 0.0))
            if current + 0.10 < target:
                revision.goal = {"target_height": target}
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
                {k: float(v) for k, v in revision.cost_weights.items()}
            )
        if revision.recovery_mode:
            hints = list(self.active_config.recovery_hints)
            if revision.recovery_mode not in hints:
                hints.append(revision.recovery_mode)
            self.active_config.recovery_hints = hints


def _evaluate_trigger(trigger: str, metrics: Dict[str, Any]) -> bool:
    """Evaluate a phase trigger string against current metrics.

    Supported triggers:
      "initial"               — always True
      "eef_near_box:<dist>"   — eef_to_contact_distance < dist
      "wall_contact"          — wall_contact is True
      "contact_force:<thresh>"— wall_contact AND |F_normal| > thresh
      "height_above:<height>" — box_top_height > height
    """
    if trigger == "initial":
        return True

    if trigger == "wall_contact":
        return bool(metrics.get("wall_contact", False))

    if trigger.startswith("eef_near_box:"):
        threshold = float(trigger.split(":")[1])
        return float(metrics.get("eef_to_contact_distance", 999.0)) < threshold

    if trigger.startswith("contact_force:"):
        threshold = float(trigger.split(":")[1])
        return (
            bool(metrics.get("wall_contact", False))
            and abs(float(metrics.get("wall_normal_force", 0.0))) > threshold
        )

    if trigger.startswith("height_above:"):
        threshold = float(trigger.split(":")[1])
        return float(metrics.get("box_top_height", 0.0)) > threshold

    return False
