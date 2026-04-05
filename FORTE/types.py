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

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


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


def default_phases() -> List[TaskPhase]:
    """Default 3-phase plan for wall-assisted box lift."""
    return [
        TaskPhase(
            name="approach",
            trigger="initial",
            cost_weights={
                "task_height": 2.0,
                "task_contact": 8.0,
                "task_pose": 18.0,   # drive EEF to contact point
                "energy": 0.0,       # no force terms pre-contact
                "force_upper": 0.0,
                "force_lower": 0.0,
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.03, contact_vertical_offset_scale=0.0,
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
                "energy": 0.0,
                "force_upper": 0.0,
                "force_lower": 0.0,
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.03, contact_vertical_offset_scale=0.0,
            ),
            force_band=ForceBand(lower=0.0, upper=100.0),
            goal={"target_height": 0.50},
        ),
        TaskPhase(
            name="lift",
            trigger="wall_contact",
            cost_weights={
                "task_height": 18.0,    # lift dominates
                "task_contact": 14.0,   # maintain wall contact
                "task_pose": 2.0,       # EEF loosely tracks contact
                "energy": 0.2,          # Eq.5 λ_E: stiffness-aware
                "force_upper": 25.0,    # Eq.5 ρ: don't jam
                "force_lower": 12.0,    # Eq.5 γ: maintain contact
            },
            contact_strategy=ContactStrategy(
                approach_face_axis=1, approach_face_sign=-1.0,
                contact_standoff=0.02,
                contact_vertical_offset_scale=-0.5,  # push from below center
            ),
            force_band=ForceBand(lower=15.0, upper=35.0),
            goal={"target_height": 0.50, "gap_target": -0.03},
            action_prior=[0.0, 0.6, 0.4, 0.0, 0.0, 0.0],  # push into wall (y+) and up (z+)
        ),
    ]


def default_physics_config() -> PhysicsConfig:
    phases = default_phases()
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
        ),
    )
