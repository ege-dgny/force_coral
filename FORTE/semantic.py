"""Bounded semantic supervision with phase transitions and LLM revision.

The VLM generates a sequence of TaskPhases. The SemanticManager:
1. Initializes with the VLM's phase plan (or defaults)
2. Evaluates phase triggers against live metrics each step
3. On transition: updates active_config's cost_weights, contact_strategy,
   force_band, and goal from the new phase
4. On failure: re-queries the LLM with current config + metrics + image
   (CoRAL-style refine_plan). The LLM can change everything: cost weights,
   force bands, task_frame, stiffness_prior, phases, contact strategy.

The FORTE cost structure (Eq. 5) is unchanged — only its parameters rotate.
"""

from __future__ import annotations

import logging
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

LOGGER = logging.getLogger(__name__)


class SemanticManager:
    """Phase-aware semantic supervisor with LLM revision."""

    def __init__(
        self,
        parser: Optional[TaskPhysicsParser] = None,
        review_interval: int = 20,
        max_revisions: int = 5,
    ) -> None:
        self.parser = parser
        self.review_interval = max(1, int(review_interval))
        self.max_revisions = int(max_revisions)
        self.active_config: PhysicsConfig = default_physics_config()
        self.revision_history: List[SemanticRevision] = []
        self.phase_index: int = 0
        self._llm_revision_count: int = 0
        self._llm_fail_count: int = 0
        self._max_llm_failures: int = 3
        self._last_revision_height: float = 0.0

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
            except Exception as exc:
                LOGGER.warning("VLM init failed (%s), using defaults", exc)
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
            return None
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
    # Revision logic (LLM-based when parser available, bounded fallback otherwise)
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
        step_idx: int = 0,
        estimator_state: Optional[Dict[str, Any]] = None,
        image: Any = None,
    ) -> SemanticRevision:
        """Revise the plan. Uses LLM when parser available, bounded fallback otherwise."""

        # Try LLM revision if parser available, under revision cap, and not too many failures
        if (
            self.parser is not None
            and self._llm_revision_count < self.max_revisions
            and self._llm_fail_count < self._max_llm_failures
        ):
            return self._llm_revise(
                monitor_status=monitor_status,
                recent_metrics=recent_metrics,
                step_idx=step_idx,
                estimator_state=estimator_state,
                image=image,
            )

        # Fallback: bounded local revision (no LLM)
        return self._bounded_revise(
            monitor_status=monitor_status,
            recent_metrics=recent_metrics,
        )

    def _llm_revise(
        self,
        *,
        monitor_status: Dict[str, Any],
        recent_metrics: Dict[str, Any],
        step_idx: int,
        estimator_state: Optional[Dict[str, Any]],
        image: Any,
    ) -> SemanticRevision:
        """Re-query LLM with failure feedback (CoRAL's refine_plan pattern)."""
        revision = SemanticRevision(
            review_reason=f"llm_revision_{self._llm_revision_count}",
        )
        current_height = float(recent_metrics.get("box_height", 0.0))

        try:
            new_config = self.parser.refine_phases(
                current_config=self.active_config,
                current_phase_name=self.current_phase.name,
                phase_index=self.phase_index,
                steps_executed=step_idx,
                monitor_feedback=monitor_status,
                recent_metrics=recent_metrics,
                estimator_state=estimator_state,
                image=image,
            )

            # Apply the new config
            old_phase_name = self.current_phase.name
            self.active_config = new_config

            # Try to resume at same phase name, else restart from phase 0
            self.phase_index = 0
            for i, phase in enumerate(new_config.phases):
                if phase.name == old_phase_name:
                    self.phase_index = i
                    break
            self._apply_phase(self.active_config.phases[self.phase_index])

            self._llm_revision_count += 1
            self._last_revision_height = current_height

            revision.phase_transition = self.current_phase.name
            revision.cost_weights = dict(self.active_config.cost_weights)
            revision.force_band = {
                "lower": self.active_config.force_band.lower,
                "upper": self.active_config.force_band.upper,
            }
            LOGGER.info(
                "LLM revision %d: phase=%s, heights=%.3f→?",
                self._llm_revision_count, self.current_phase.name, current_height,
            )

        except Exception as exc:
            self._llm_fail_count += 1
            LOGGER.warning(
                "LLM revision failed (%d/%d): %s",
                self._llm_fail_count, self._max_llm_failures, exc,
            )
            return self._bounded_revise(
                monitor_status=monitor_status,
                recent_metrics=recent_metrics,
            )

        self.revision_history.append(revision)
        return revision

    def _bounded_revise(
        self,
        *,
        monitor_status: Dict[str, Any],
        recent_metrics: Dict[str, Any],
    ) -> SemanticRevision:
        """Bounded local revision (no LLM). Fallback when parser unavailable."""
        revision = SemanticRevision(
            review_reason=str(monitor_status.get("reason", "periodic")),
        )
        cfg = self.active_config

        MAX_HEIGHT_WEIGHT = 60.0
        MAX_FORCE_LOWER = 5.0

        # Don't escalate force_lower in pre-contact phases (force costs disabled)
        force_phase = float(cfg.cost_weights.get("force_lower", 0.0)) > 0

        if (monitor_status.get("drop", False) or monitor_status.get("contact_lost", False)) and force_phase:
            lower = min(
                MAX_FORCE_LOWER,
                min(cfg.force_band.upper - 0.5, cfg.force_band.lower + 0.5),
            )
            revision.force_band = {"lower": max(0.0, lower), "upper": cfg.force_band.upper}
            revision.recovery_mode = "re_establish_contact"

        elif monitor_status.get("stall", False):
            revision.force_band = {
                "lower": cfg.force_band.lower,
                "upper": max(cfg.force_band.lower + 0.5, cfg.force_band.upper - 0.5),
            }
            weights = dict(cfg.cost_weights)
            cur_h = float(weights.get("task_height", 16.0))
            weights["task_height"] = min(MAX_HEIGHT_WEIGHT, cur_h * 1.05)
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
    """Evaluate a phase trigger string against current metrics."""
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
