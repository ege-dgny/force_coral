"""FORTE data types: TaskPhase, PhysicsConfig, ForceBand, ContactStrategy."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class ForceBand:
    lower: float
    upper: float

    def to_dict(self) -> Dict[str, float]:
        return {"lower": float(self.lower), "upper": float(self.upper)}


@dataclasses.dataclass
class ContactStrategy:
    """Which face to approach, standoff distance, vertical offset."""

    approach_face_axis: int = 1        # 0=x, 1=y, 2=z
    approach_face_sign: float = -1.0   # direction along axis
    contact_standoff: float = 0.03     # meters from face surface
    contact_vertical_offset_scale: float = 0.0  # fraction of half-extent
    gripper_command: float = -1.0  # keep CoRAL/FORTE convention used in this stack
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ContactHypothesis:
    """Candidate contact parameterization scored online."""

    contact_strategy: ContactStrategy
    score: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_strategy": self.contact_strategy.to_dict(),
            "score": float(self.score),
            "reason": self.reason,
        }


@dataclasses.dataclass
class ContactBelief:
    """Lightweight contact mode belief used for guarded fallback."""

    mode: str
    confidence: float
    uncertain_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": float(self.confidence),
            "uncertain_steps": int(self.uncertain_steps),
        }


@dataclasses.dataclass
class ContactSelectorState:
    """Temporal state for contact-point selection and fallback decisions."""

    active_strategy: ContactStrategy
    last_switch_step: int = -1
    switch_count: int = 0
    high_error_steps: int = 0
    fallback_active: bool = False
    fallback_reason: str = ""
    tracking_error_ema: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_strategy": self.active_strategy.to_dict(),
            "last_switch_step": int(self.last_switch_step),
            "switch_count": int(self.switch_count),
            "high_error_steps": int(self.high_error_steps),
            "fallback_active": bool(self.fallback_active),
            "fallback_reason": self.fallback_reason,
            "tracking_error_ema": float(self.tracking_error_ema),
        }


@dataclasses.dataclass
class TaskPhase:
    """One phase of a multi-phase manipulation task.

    The VLM generates an ordered sequence of phases. Each phase carries its own
    Q-weights (cost_weights), x_goal (goal), force band [F_min, F_max], and
    contact strategy. The FORTE cost structure (Eq. 5) is unchanged — only its
    parameters rotate when the trigger fires.

    Triggers evaluated by SemanticManager against live metrics dict:
      "initial"               — always true (first phase)
      "eef_near_box:0.05"     — eef_to_contact_distance < 0.05
      "wall_contact"          — wall_gap <= tolerance
      "contact_force:0.5"     — wall_contact AND |F_normal| > 0.5
      "height_above:0.25"     — box_top_height > 0.25
    """

    name: str
    trigger: str
    cost_weights: Dict[str, float]
    contact_strategy: ContactStrategy
    force_band: ForceBand
    goal: Dict[str, Any]
    action_prior: List[float] = dataclasses.field(default_factory=lambda: [0.0]*6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trigger": self.trigger,
            "cost_weights": self.cost_weights,
            "contact_strategy": self.contact_strategy.to_dict(),
            "force_band": self.force_band.to_dict(),
            "goal": self.goal,
            "action_prior": self.action_prior,
        }


@dataclasses.dataclass
class PhysicsConfig:
    """Structured semantic package emitted by the VLM.

    Global fields (stiffness_prior, task_frame) stay constant across phases.
    Per-phase fields (cost_weights, contact_strategy, force_band, goal) are
    set by SemanticManager when a phase transition fires.
    """

    # Global (constant across phases)
    stiffness_prior: Dict[str, str]
    task_frame_euler: List[float]
    task_frame: np.ndarray
    phases: List[TaskPhase]
    recovery_hints: List[str]

    # Active phase (updated by SemanticManager on phase transition)
    force_band: ForceBand
    goal: Dict[str, Any]
    cost_weights: Dict[str, float]
    contact_strategy: ContactStrategy = dataclasses.field(
        default_factory=ContactStrategy,
    )

    def __post_init__(self) -> None:
        if self.task_frame is None and self.task_frame_euler is not None:
            self.task_frame = Rotation.from_euler(
                "xyz", self.task_frame_euler, degrees=True
            ).as_matrix()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stiffness": self.stiffness_prior,
            "task_frame_euler": self.task_frame_euler,
            "phases": [p.to_dict() for p in self.phases],
            "recovery_hints": self.recovery_hints,
            "active_phase": {
                "force_band": self.force_band.to_dict(),
                "goal": self.goal,
                "cost_weights": self.cost_weights,
                "contact_strategy": self.contact_strategy.to_dict(),
            },
        }


@dataclasses.dataclass
class SemanticRevision:
    """Constrained update emitted by the semantic supervisor."""

    goal: Dict[str, Any] | None = None
    force_band: Dict[str, float] | None = None
    cost_weights: Dict[str, float] | None = None
    recovery_mode: str | None = None
    phase_transition: str | None = None  # name of new phase, if transitioned
    review_reason: str = "periodic"


def infer_task_family(task_name: str) -> str:
    name = (task_name or "").lower()
    if "flip" in name and "wall" in name:
        return "wall_flip"
    if "force_hold" in name or "constant_force" in name:
        return "force_hold"
    if "spring" in name or "press_the" in name:
        return "spring_press"
    return "wall_lift"


def default_phases(task_name: str = "") -> List[TaskPhase]:
    """Default phase plans for supported task families."""
    task_family = infer_task_family(task_name)

    if task_family == "force_hold":
        # Two phases: drive the EEF to the box's -Y face, then maintain F_n in band.
        # Hold-phase weights collapse to force terms only (paper Eq. 5 γ + ρ).
        return [
            TaskPhase(
                name="approach",
                trigger="initial",
                cost_weights={
                    "task_height": 0.0,
                    "task_contact": 8.0,
                    "task_pose": 18.0,
                    "task_lateral": 4.0,
                    "task_tilt": 5.0,
                    "energy": 0.0,
                    "force_upper": 0.0,
                    "force_lower": 0.0,
                },
                contact_strategy=ContactStrategy(
                    approach_face_axis=1, approach_face_sign=-1.0,
                    contact_standoff=0.02, contact_vertical_offset_scale=0.0,
                    metadata={"world_offset": [0.0, 0.0, 0.0]},
                ),
                force_band=ForceBand(lower=0.0, upper=100.0),
                goal={"target_force": 5.0, "force_band_lower": 3.0, "force_band_upper": 8.0},
            ),
            TaskPhase(
                name="hold",
                trigger="contact_force:0.5",
                cost_weights={
                    "task_height": 0.0,
                    "task_contact": 4.0,
                    "task_pose": 2.0,
                    "task_lateral": 30.0,
                    "task_tilt": 8.0,
                    "energy": 0.2,         # λ_E: now contributes (soft wall → real δ)
                    "force_upper": 30.0,   # ρ: hard cap on N
                    "force_lower": 30.0,   # γ: contact maintenance (the FORTE story)
                },
                contact_strategy=ContactStrategy(
                    approach_face_axis=1, approach_face_sign=-1.0,
                    contact_standoff=0.005, contact_vertical_offset_scale=0.0,
                    gripper_command=-1.0,
                    metadata={"world_offset": [0.0, 0.0, 0.0]},
                ),
                force_band=ForceBand(lower=3.0, upper=8.0),
                goal={"target_force": 5.0, "force_band_lower": 3.0, "force_band_upper": 8.0},
                action_prior=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # steady push into wall
            ),
        ]

    if task_family == "wall_flip":
        return [
            TaskPhase(
                name="approach",
                trigger="initial",
                cost_weights={
                    "task_height": 1.0,
                    "task_contact": 8.0,
                    "task_pose": 18.0,
                    "task_lateral": 4.0,
                    "energy": 0.0,
                    "force_upper": 0.0,
                    "force_lower": 0.0,
                },
                contact_strategy=ContactStrategy(
                    approach_face_axis=1, approach_face_sign=-1.0,
                    contact_standoff=0.03, contact_vertical_offset_scale=0.0,
                    metadata={"world_offset": [0.0, 0.0, -0.05]},
                ),
                force_band=ForceBand(lower=0.0, upper=100.0),
                goal={"target_tilt_deg": 80.0, "target_height": 0.10},
            ),
            TaskPhase(
                name="push_to_wall",
                trigger="eef_near_box:0.05",
                cost_weights={
                    "task_height": 1.0,
                    "task_contact": 18.0,
                    "task_pose": 6.0,
                    "task_lateral": 20.0,
                    "energy": 0.0,
                    "force_upper": 0.0,
                    "force_lower": 0.0,
                },
                contact_strategy=ContactStrategy(
                    approach_face_axis=1, approach_face_sign=-1.0,
                    contact_standoff=0.02, contact_vertical_offset_scale=0.15,
                    metadata={"world_offset": [0.0, 0.0, -0.05]},
                ),
                force_band=ForceBand(lower=0.0, upper=100.0),
                goal={"target_tilt_deg": 80.0, "target_height": 0.10},
            ),
            TaskPhase(
                name="flip",
                trigger="wall_contact",
                cost_weights={
                    "task_height": 0.5,
                    "task_contact": 8.0,
                    "task_pose": 4.0,
                    "task_lateral": 30.0,
                    "task_tilt": 42.0,
                    "energy": 0.1,
                    "force_upper": 20.0,
                    "force_lower": 2.0,
                },
                contact_strategy=ContactStrategy(
                    approach_face_axis=1, approach_face_sign=-1.0,
                    contact_standoff=0.01, contact_vertical_offset_scale=0.22,
                    gripper_command=-1.0,
                    metadata={"world_offset": [0.0, 0.0, -0.05]},
                ),
                force_band=ForceBand(lower=1.0, upper=18.0),
                goal={"target_tilt_deg": 80.0, "target_height": 0.12, "gap_target": 0.0},
                action_prior=[0.0, 0.25, 0.60, 0.0, 0.0, 0.0],
            ),
        ]

    return [
        TaskPhase(
            name="approach",
            trigger="initial",
            cost_weights={
                "task_height": 2.0,
                "task_contact": 8.0,
                "task_pose": 18.0,   # drive EEF to contact point
                "task_lateral": 4.0, # prevent drift during approach
                "energy": 0.0,       # no force terms pre-contact
                "force_upper": 0.0,
                "force_lower": 0.0,
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.03, contact_vertical_offset_scale=0.0,
                metadata={"world_offset": [0.0, 0.0, -0.05]},
            ),
            force_band=ForceBand(lower=0.0, upper=100.0),
            goal={"target_height": 0.50},
        ),
        TaskPhase(
            name="push_to_wall",
            trigger="eef_near_box:0.05",
            cost_weights={
                "task_height": 4.0,
                "task_contact": 18.0,  # wall proximity dominates
                "task_pose": 4.0,
                "task_lateral": 20.0,  # prevent drift during push
                "energy": 0.0,
                "force_upper": 0.0,
                "force_lower": 0.0,
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.03, contact_vertical_offset_scale=0.0,
                metadata={"world_offset": [0.0, 0.0, -0.05]},
            ),
            force_band=ForceBand(lower=0.0, upper=100.0),
            goal={"target_height": 0.50},
        ),
        TaskPhase(
            name="lift",
            trigger="wall_contact",
            cost_weights={
                "task_height": 30.0,    # lift is primary objective
                "task_contact": 6.0,    # light wall contact (minimize friction)
                "task_pose": 4.0,       # keep EEF near contact point
                "task_lateral": 50.0,   # prevent lateral drift
                "task_tilt": 20.0,      # prevent tipping
                "energy": 0.1,          # Eq.5 λ_E: light stiffness penalty
                "force_upper": 20.0,    # Eq.5 ρ: prevent jamming
                "force_lower": 2.0,     # Eq.5 γ: very light contact maintenance
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.02,
                contact_vertical_offset_scale=0.0,
                gripper_command=-1.0,
                metadata={"world_offset": [0.0, 0.0, -0.05]},
            ),
            force_band=ForceBand(lower=1.0, upper=15.0),
            goal={"target_height": 0.50, "gap_target": 0.0},
            action_prior=[0.0, 0.2, 0.8, 0.0, 0.0, 0.0],  # push into wall + lift
        ),
    ]


def default_physics_config(task_name: str = "") -> PhysicsConfig:
    phases = default_phases(task_name=task_name)
    first = phases[0]
    return PhysicsConfig(
        stiffness_prior={"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
        task_frame_euler=[0.0, 0.0, 0.0],
        task_frame=np.eye(3),
        phases=phases,
        recovery_hints=["re_establish_contact", "reduce_normal_force_if_stalled"],
        # Active = first phase
        force_band=ForceBand(lower=first.force_band.lower, upper=first.force_band.upper),
        goal=dict(first.goal),
        cost_weights=dict(first.cost_weights),
        contact_strategy=ContactStrategy(
            approach_face_axis=first.contact_strategy.approach_face_axis,
            approach_face_sign=first.contact_strategy.approach_face_sign,
            contact_standoff=first.contact_strategy.contact_standoff,
            contact_vertical_offset_scale=first.contact_strategy.contact_vertical_offset_scale,
            gripper_command=first.contact_strategy.gripper_command,
            metadata=dict(first.contact_strategy.metadata),
        ),
    )
