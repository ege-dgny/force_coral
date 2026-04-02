"""Support classes for the Phase-1 FORTE controller."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from force_coral.controllers.task_geometry import (
    build_wall_lift_task_frame,
    compute_wall_lift_task_cost,
    world_to_task,
)


@dataclasses.dataclass
class CostBreakdown:
    task: float
    energy: float
    force_upper: float
    force_lower: float
    total: float
    predicted_force_normal: float
    predicted_force_task: np.ndarray
    delta_task: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": float(self.task),
            "energy": float(self.energy),
            "force_upper": float(self.force_upper),
            "force_lower": float(self.force_lower),
            "total": float(self.total),
            "predicted_force_normal": float(self.predicted_force_normal),
            "predicted_force_task": np.asarray(self.predicted_force_task).tolist(),
            "delta_task": np.asarray(self.delta_task).tolist(),
        }


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

        contact_lost = normal_force < (self.force_lower * self.contact_grace_factor)
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


class ArtifactManager:
    """Collect logs, plots, and a replayable MP4 for a FORTE rollout."""

    def __init__(self, out_dir: str, save_video: bool = True, overlay: bool = True):
        self.out_dir = out_dir
        self.save_video = save_video
        self.overlay = overlay
        self.records: List[Dict[str, Any]] = []
        self.frames: List[np.ndarray] = []
        os.makedirs(self.out_dir, exist_ok=True)

    def log_step(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def add_frame(self, frame_bgr: np.ndarray, overlay_info: Dict[str, Any]) -> None:
        if not self.save_video:
            return
        frame = np.asarray(frame_bgr).copy()
        if self.overlay:
            import cv2

            lines = [
                f"step: {overlay_info.get('step', 0):03d}",
                f"height: {overlay_info.get('box_height', 0.0):.3f} m",
                f"F_meas_n: {overlay_info.get('measured_force_normal', 0.0):.2f} N",
                f"F_pred_n: {overlay_info.get('predicted_force_normal', 0.0):.2f} N",
                f"regime: {overlay_info.get('regime', 'unknown')}",
            ]
            y = 24
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 24
        self.frames.append(frame)

    def finalize(self) -> Dict[str, str]:
        json_path = os.path.join(self.out_dir, "forte_log.json")
        csv_path = os.path.join(self.out_dir, "forte_log.csv")
        summary_path = os.path.join(self.out_dir, "summary.json")
        plots = self._write_plots()
        video_path = self._write_video()
        with open(json_path, "w") as handle:
            json.dump(self.records, handle, indent=2)
        self._write_csv(csv_path)
        summary = self._build_summary(video_path=video_path, plots=plots)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        return {
            "json": json_path,
            "csv": csv_path,
            "summary": summary_path,
            "video": video_path or "",
            "plots": ",".join(plots),
        }

    def _write_csv(self, path: str) -> None:
        if not self.records:
            return
        keys = sorted(self.records[0].keys())
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: self._csv_value(record.get(key)) for key in keys})

    def _csv_value(self, value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    def _write_plots(self) -> List[str]:
        if not self.records:
            return []
        step = [record["step"] for record in self.records]
        normal_force = [record["measured_force_normal"] for record in self.records]
        pred_force = [record["predicted_force_normal"] for record in self.records]
        lower = [record["force_band_lower"] for record in self.records]
        upper = [record["force_band_upper"] for record in self.records]
        height = [record["box_height"] for record in self.records]
        energy = [record["cost_energy"] for record in self.records]
        total = [record["cost_total"] for record in self.records]
        out_paths: List[str] = []

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(step, normal_force, label="measured")
        ax.plot(step, pred_force, label="predicted")
        ax.plot(step, lower, "--", label="F_min")
        ax.plot(step, upper, "--", label="F_max")
        ax.set_xlabel("step")
        ax.set_ylabel("normal force (N)")
        ax.legend()
        force_plot = os.path.join(self.out_dir, "force_band.png")
        fig.tight_layout()
        fig.savefig(force_plot)
        plt.close(fig)
        out_paths.append(force_plot)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(step, height, label="box height")
        ax.set_xlabel("step")
        ax.set_ylabel("height (m)")
        height_plot = os.path.join(self.out_dir, "height_progress.png")
        fig.tight_layout()
        fig.savefig(height_plot)
        plt.close(fig)
        out_paths.append(height_plot)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(step, energy, label="energy")
        ax.plot(step, total, label="total")
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.legend()
        cost_plot = os.path.join(self.out_dir, "costs.png")
        fig.tight_layout()
        fig.savefig(cost_plot)
        plt.close(fig)
        out_paths.append(cost_plot)
        return out_paths

    def _write_video(self) -> str | None:
        if not self.save_video or not self.frames:
            return None
        import cv2

        height, width = self.frames[0].shape[:2]
        path = os.path.join(self.out_dir, "rollout.mp4")
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (width, height),
        )
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        return path

    def _build_summary(self, *, video_path: str | None, plots: List[str]) -> Dict[str, Any]:
        if not self.records:
            return {"num_steps": 0, "video": video_path, "plots": plots}
        final = self.records[-1]
        return {
            "num_steps": len(self.records),
            "success": bool(final.get("success", False)),
            "final_height": float(final.get("box_height", 0.0)),
            "max_normal_force": float(max(r["measured_force_normal"] for r in self.records)),
            "time_in_band": int(sum(1 for r in self.records if r["regime"] == "in-band")),
            "num_drop_events": int(sum(1 for r in self.records if r.get("drop", False))),
            "num_stall_events": int(sum(1 for r in self.records if r.get("stall", False))),
            "video": video_path,
            "plots": plots,
        }
