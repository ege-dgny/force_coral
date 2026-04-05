"""Task progress monitor for wall-lift tasks."""

from __future__ import annotations

from typing import Any, Dict


class WallLiftTaskMonitor:
    """Detects success, stall, drop, and repeated over-force events."""

    def __init__(
        self,
        *,
        target_height: float,
        force_lower: float,
        force_upper: float,
        progress_eps: float = 1e-3,
        stall_window: int = 8,
        drop_threshold: float = 0.01,
        over_force_window: int = 4,
        contact_grace_factor: float = 0.5,
    ) -> None:
        self.target_height = float(target_height)
        self.force_lower = float(force_lower)
        self.force_upper = float(force_upper)
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
        success = box_height >= self.target_height and contact_ok
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
        }
