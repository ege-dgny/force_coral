"""Task progress monitors for FORTE tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WallLiftTaskMonitor:
    """Detects success, stall, drop, and repeated over-force events."""

    def __init__(
        self,
        *,
        target_height: float,
        force_lower: float,
        force_upper: float,
        target_metric_name: str = "height",
        success_requires_contact: bool = True,
        progress_eps: float = 1e-3,
        stall_window: int = 8,
        drop_threshold: float = 0.01,
        over_force_window: int = 4,
        contact_grace_factor: float = 0.5,
    ) -> None:
        self.target_height = float(target_height)
        self.force_lower = float(force_lower)
        self.force_upper = float(force_upper)
        self.target_metric_name = str(target_metric_name)
        self.success_requires_contact = bool(success_requires_contact)
        self.progress_eps = float(progress_eps)
        self.stall_window = int(stall_window)
        self.drop_threshold = float(drop_threshold)
        self.over_force_window = int(over_force_window)
        self.contact_grace_factor = float(contact_grace_factor)
        self.prev_height: float | None = None
        self.stall_counter = 0
        self.over_force_counter = 0

    def update(
        self,
        *,
        box_height: float,
        normal_force: float,
        wall_contact: bool,
    ) -> Dict[str, Any]:
        if self.prev_height is None:
            height_delta = 0.0
        else:
            height_delta = float(box_height - self.prev_height)
        self.prev_height = float(box_height)

        # Only flag contact_lost if force_lower > 0 (i.e., force is expected)
        contact_lost = (
            self.force_lower > 0.0
            and normal_force < (self.force_lower * self.contact_grace_factor)
        )
        drop = height_delta < -self.drop_threshold
        over_force = normal_force > self.force_upper

        if over_force:
            self.over_force_counter += 1
        else:
            self.over_force_counter = 0

        stalled = abs(height_delta) < self.progress_eps and normal_force >= self.force_lower
        if stalled:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        contact_ok = bool(wall_contact)
        success = box_height >= self.target_height and (
            contact_ok if self.success_requires_contact else True
        )
        repeated_over_force = self.over_force_counter >= self.over_force_window
        stall = self.stall_counter >= self.stall_window

        reason = "ok"
        if success:
            reason = "success"
        elif drop:
            reason = "drop"
        elif repeated_over_force:
            reason = "repeated_over_force"
        elif stall:
            reason = "stall"
        elif contact_lost:
            reason = "contact_lost"

        regime = "in-band"
        if normal_force < self.force_lower:
            regime = "under-force"
        elif normal_force > self.force_upper:
            regime = "over-force"

        return {
            "success": success,
            "drop": drop,
            "stall": stall,
            "contact_lost": contact_lost,
            "repeated_over_force": repeated_over_force,
            "reason": reason,
            "regime": regime,
            "height_delta": height_delta,
            "wall_contact": contact_ok,
            "target_metric_name": self.target_metric_name,
        }


class SustainedForceMonitor:
    """Success monitor for force-tracking tasks (force_hold, spring_press).

    Success: a chosen scalar metric stays inside ``[lower, upper]`` for
    ``required_steps`` consecutive steps. The metric can be raw wall-normal
    force (force_hold) or a depth value mapped to an equivalent band
    (spring_press: F = k·d so a depth band is just a force band divided by k).
    """

    def __init__(
        self,
        *,
        lower: float,
        upper: float,
        required_steps: int = 20,
        target_metric_name: str = "wall_normal_force",
        drop_threshold: float = 1.0,
    ) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.required_steps = int(required_steps)
        self.target_metric_name = str(target_metric_name)
        self.drop_threshold = float(drop_threshold)
        self.in_band_counter = 0
        self.steps_in_band_total = 0
        self.prev_value: Optional[float] = None
        # Compatibility with WallLiftTaskMonitor's interface (run_forte
        # reads these fields when toggling between monitors per phase).
        self.target_height = 0.5 * (self.lower + self.upper)
        self.force_lower = self.lower
        self.force_upper = self.upper

    def update(
        self,
        *,
        value: float,
        wall_contact: bool = True,
        force_band_lower: Optional[float] = None,
        force_band_upper: Optional[float] = None,
    ) -> Dict[str, Any]:
        if force_band_lower is not None:
            self.lower = float(force_band_lower)
        if force_band_upper is not None:
            self.upper = float(force_band_upper)
        v = float(value)
        # Only treat the sample as "in band" once contact has been made AND
        # the value is meaningfully non-zero. Pre-contact, lower=0 would make
        # an idle F_n=0 readout count as in-band and trip the 20-step success.
        meaningful = bool(wall_contact) and v >= max(0.25, 0.25 * max(self.lower, 1e-6))
        in_band = meaningful and (self.lower <= v <= self.upper)
        if in_band:
            self.in_band_counter += 1
            self.steps_in_band_total += 1
        else:
            self.in_band_counter = 0

        delta = 0.0 if self.prev_value is None else v - self.prev_value
        self.prev_value = v

        success = self.in_band_counter >= self.required_steps
        drop = (
            self.prev_value is not None
            and v < self.lower
            and abs(delta) > self.drop_threshold
        )
        # No stall concept for force_hold — the value is supposed to be flat.
        regime = "in-band" if in_band else ("under-force" if v < self.lower else "over-force")
        reason = "ok"
        if success:
            reason = "success"
        elif drop:
            reason = "drop"
        elif not in_band:
            reason = "out_of_band"

        return {
            "success": success,
            "drop": drop,
            "stall": False,
            "contact_lost": (v < self.lower * 0.5),
            "repeated_over_force": (v > self.upper * 1.2),
            "reason": reason,
            "regime": regime,
            "value": v,
            "in_band": in_band,
            "in_band_counter": int(self.in_band_counter),
            "steps_in_band_total": int(self.steps_in_band_total),
            "wall_contact": bool(wall_contact),
            "target_metric_name": self.target_metric_name,
            "height_delta": float(delta),
        }
