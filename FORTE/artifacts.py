"""Logging, debug overlays, plots, and video for FORTE rollouts."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


# ── 3D → pixel projection ──────────────────────────────────────────────

def world_to_pixel(
    sim: Any,
    camera_name: str,
    point_world: np.ndarray,
    img_h: int,
    img_w: int,
) -> Optional[Tuple[int, int]]:
    """Project a 3D world point to pixel coords using MuJoCo camera."""
    model, data = sim.model, sim.data
    cam_id = model.camera_name2id(camera_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # MuJoCo GL convention: camera looks along -Z, Y is up
    # Convert to OpenCV: camera looks along +Z, Y is down
    gl_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    R_cv = gl_to_cv @ cam_mat.T  # world-to-camera rotation (OpenCV)
    t_cv = R_cv @ (np.asarray(point_world) - cam_pos)

    if t_cv[2] <= 0.01:
        return None  # behind camera

    # Intrinsics from fovy
    fovy_rad = float(model.cam_fovy[cam_id]) * np.pi / 180.0
    fy = img_h / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy  # square pixels
    cx, cy = img_w / 2.0, img_h / 2.0

    px = int(fx * t_cv[0] / t_cv[2] + cx)
    py = int(fy * t_cv[1] / t_cv[2] + cy)

    if 0 <= px < img_w and 0 <= py < img_h:
        return (px, py)
    return None


# ── Color palette ───────────────────────────────────────────────────────

_PHASE_COLORS = {
    "approach": (255, 255, 0),     # cyan (BGR)
    "push_to_wall": (0, 255, 255), # yellow (BGR)
    "lift": (0, 255, 0),           # green (BGR)
}
_EEF_COLOR = (0, 255, 0)           # green
_ANCHOR_COLOR = (0, 0, 255)        # red
_DESIRED_COLOR = (0, 0, 200)       # dark red
_BOX_COLOR = (255, 100, 0)         # blue
_FORCE_COLOR = (0, 255, 255)       # yellow
_EVENT_COLOR = (0, 0, 255)         # red
_WHITE = (255, 255, 255)
_GRAY = (180, 180, 180)

# Cost bar colors (BGR)
_COST_COLORS = {
    "cost_height": (180, 120, 0),
    "cost_contact": (0, 180, 180),
    "cost_pose": (0, 180, 0),
    "cost_energy": (180, 0, 180),
    "cost_force_upper": (0, 0, 220),
    "cost_force_lower": (220, 0, 0),
}


def _draw_debug_overlay(
    frame: np.ndarray,
    sim: Any,
    info: Dict[str, Any],
    camera_name: str = "frontview",
) -> np.ndarray:
    """Draw markers, force arrows, cost bar, and text HUD on a frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # ── Phase banner (top-left) ──
    phase = info.get("phase", "?")
    step = info.get("step", 0)
    color = _PHASE_COLORS.get(phase, _WHITE)
    cv2.putText(out, f"{phase}  step:{step:03d}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # ── Event banners ──
    if info.get("phase_transition"):
        cv2.putText(out, f">> {info.get('phase_transition_to', '?')}",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _EVENT_COLOR, 2)
    if info.get("semantic_revision"):
        cv2.putText(out, "REVISION", (w - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _EVENT_COLOR, 1)

    # ── 3D markers ──
    if sim is not None:
        # EEF
        eef = info.get("eef_pos")
        if eef is not None:
            px = world_to_pixel(sim, camera_name, eef, h, w)
            if px:
                cv2.circle(out, px, 5, _EEF_COLOR, -1)
                cv2.putText(out, "EEF", (px[0]+7, px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, _EEF_COLOR, 1)

        # Contact anchor (on box surface)
        anchor = info.get("contact_anchor_world")
        if anchor is not None:
            px = world_to_pixel(sim, camera_name, anchor, h, w)
            if px:
                cv2.circle(out, px, 5, _ANCHOR_COLOR, -1)
                cv2.putText(out, "anc", (px[0]+7, px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, _ANCHOR_COLOR, 1)

        # Desired contact (with standoff)
        desired = info.get("desired_contact_world")
        if desired is not None:
            px = world_to_pixel(sim, camera_name, desired, h, w)
            if px:
                cv2.circle(out, px, 6, _DESIRED_COLOR, 1)  # hollow
                cv2.putText(out, "tgt", (px[0]+7, px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, _DESIRED_COLOR, 1)

        # Box center
        bpos = info.get("box_pos")
        if bpos is not None:
            px = world_to_pixel(sim, camera_name, bpos, h, w)
            if px:
                cv2.circle(out, px, 4, _BOX_COLOR, -1)

        # Force arrow from anchor
        if anchor is not None:
            f_task = info.get("measured_force_task")
            if f_task is not None and np.linalg.norm(f_task) > 0.5:
                # Task frame: [normal(+y_world), lift(+z_world), lateral(+x_world)]
                f_world = np.array([f_task[2], f_task[0], f_task[1]])
                arrow_end = np.array(anchor) + f_world * 0.005  # 5mm per N
                px_start = world_to_pixel(sim, camera_name, anchor, h, w)
                px_end = world_to_pixel(sim, camera_name, arrow_end, h, w)
                if px_start and px_end:
                    cv2.arrowedLine(out, px_start, px_end, _FORCE_COLOR, 2,
                                    tipLength=0.3)

    # ── Text HUD (right side) ──
    hud_x = w - 130
    hud_lines = [
        f"h={info.get('box_height', 0):.3f}m",
        f"gap={info.get('wall_gap', 0):.4f}m",
        f"F_n={info.get('wall_normal_force', 0):.1f}N",
        f"e2c={info.get('eef_to_contact_error', info.get('eef_to_contact', 0)):.3f}m",
        f"tilt={_max_tilt(info):.1f}deg",
        f"ctct={'Y' if info.get('contact_latched') else 'N'}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(out, line, (hud_x, 20 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1, cv2.LINE_AA)

    # ── Cost breakdown bar (bottom) ──
    cost_keys = ["cost_height", "cost_contact", "cost_pose",
                 "cost_energy", "cost_force_upper", "cost_force_lower"]
    costs = [max(0.0, float(info.get(k, 0.0))) for k in cost_keys]
    total = sum(costs)
    if total > 0.001:
        bar_y = h - 16
        bar_h = 12
        bar_w = w - 20
        x = 10
        for k, c in zip(cost_keys, costs):
            seg_w = max(1, int(bar_w * c / total))
            cv2.rectangle(out, (x, bar_y), (x + seg_w, bar_y + bar_h),
                          _COST_COLORS.get(k, _GRAY), -1)
            if seg_w > 20:
                label = k.replace("cost_", "")[:4]
                cv2.putText(out, label, (x + 2, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
            x += seg_w

    return out


def _max_tilt(info: Dict[str, Any]) -> float:
    euler = info.get("box_euler_deg")
    if euler is None:
        return 0.0
    # ZYX euler: [yaw, pitch, roll]. Tilt = max(|pitch|, |roll|)
    return float(max(abs(euler[1]), abs(euler[2])))


# ── ArtifactManager ─────────────────────────────────────────────────────

class ArtifactManager:
    """Collect logs, debug frames, plots, and video for a FORTE rollout."""

    def __init__(
        self,
        out_dir: str,
        save_video: bool = True,
        overlay: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.save_video = save_video
        self.overlay = overlay
        self.records: List[Dict[str, Any]] = []
        self.frames: List[np.ndarray] = []
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

    def log_step(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def add_frame(
        self,
        frame_bgr: np.ndarray,
        overlay_info: Dict[str, Any],
        sim: Any = None,
    ) -> None:
        if not self.save_video:
            return
        frame = np.asarray(frame_bgr).copy()
        if self.overlay:
            frame = _draw_debug_overlay(frame, sim, overlay_info)
        self.frames.append(frame)
        # Save per-frame image
        step = overlay_info.get("step", len(self.frames) - 1)
        cv2.imwrite(
            os.path.join(self.frames_dir, f"{step:04d}.png"), frame,
        )

    def finalize(self) -> Dict[str, str]:
        json_path = os.path.join(self.out_dir, "forte_log.json")
        csv_path = os.path.join(self.out_dir, "forte_log.csv")
        summary_path = os.path.join(self.out_dir, "summary.json")
        plots = self._write_plots()
        video_path = self._write_video()

        with open(json_path, "w") as f:
            json.dump(self.records, f, indent=2)
        self._write_csv(csv_path)

        summary = self._build_summary(video_path=video_path, plots=plots)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "json": json_path, "csv": csv_path, "summary": summary_path,
            "video": video_path or "", "plots": ",".join(plots),
        }

    # ── CSV ──

    def _write_csv(self, path: str) -> None:
        if not self.records:
            return
        keys = sorted(self.records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for record in self.records:
                writer.writerow({k: self._csv_value(record.get(k)) for k in keys})

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    # ── Plots ──

    def _write_plots(self) -> List[str]:
        if not self.records:
            return []
        out: List[str] = []
        out.extend(self._plot_force_band())
        out.extend(self._plot_height())
        out.extend(self._plot_cost_breakdown())
        out.extend(self._plot_contact_tracking())
        out.extend(self._plot_box_orientation())
        out.extend(self._plot_action_profile())
        out.extend(self._plot_phase_timeline())
        # Spring-press headline plot: Σ̂_xx convergence to ground-truth k.
        if any("button_stiffness" in r for r in self.records):
            out.extend(self._plot_stiffness_convergence())
            out.extend(self._plot_in_band_ratio())
        return out

    def _steps(self) -> List[int]:
        return [r["step"] for r in self.records]

    def _plot_force_band(self) -> List[str]:
        steps = self._steps()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, [r["measured_force_normal"] for r in self.records],
                label="measured", linewidth=1.5)
        ax.plot(steps, [r["predicted_force_normal"] for r in self.records],
                label="predicted", linewidth=1, alpha=0.7)
        ax.plot(steps, [r["force_band_lower"] for r in self.records],
                "--", label="F_min", linewidth=1)
        ax.plot(steps, [r["force_band_upper"] for r in self.records],
                "--", label="F_max", linewidth=1)
        self._add_phase_bg(ax, steps)
        ax.set_xlabel("step"); ax.set_ylabel("normal force (N)")
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(self.out_dir, "force_band.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_height(self) -> List[str]:
        steps = self._steps()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, [r["box_height"] for r in self.records], linewidth=1.5)
        target = self.records[0].get("cost_total", 0.5)
        # Draw target if available
        if self.records and "box_height" in self.records[0]:
            ax.axhline(0.50, color="red", linestyle="--", alpha=0.5, label="target")
        self._add_phase_bg(ax, steps)
        ax.set_xlabel("step"); ax.set_ylabel("height (m)")
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(self.out_dir, "height_progress.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_cost_breakdown(self) -> List[str]:
        steps = self._steps()
        cost_keys = ["cost_height", "cost_contact", "cost_pose",
                     "cost_energy", "cost_force_upper", "cost_force_lower"]
        fig, ax = plt.subplots(figsize=(10, 4))
        bottom = np.zeros(len(steps))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, key in enumerate(cost_keys):
            vals = np.array([float(r.get(key, 0.0)) for r in self.records])
            vals = np.maximum(vals, 0.0)
            ax.bar(steps, vals, bottom=bottom, width=1.0, label=key.replace("cost_", ""),
                   color=colors[i], alpha=0.8)
            bottom += vals
        self._add_phase_bg(ax, steps)
        ax.set_xlabel("step"); ax.set_ylabel("cost")
        ax.set_title("Cost Breakdown per Step")
        ax.legend(fontsize=7, ncol=3); fig.tight_layout()
        p = os.path.join(self.out_dir, "cost_breakdown.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_contact_tracking(self) -> List[str]:
        steps = self._steps()
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        eef_err = [float(r.get("eef_to_contact_error", r.get("eef_to_contact", 0.0))) for r in self.records]
        switches = [int(r.get("face_switch_count", 0)) for r in self.records]

        axes[0].plot(steps, eef_err, linewidth=1.3, label="eef_to_contact_error")
        axes[0].axhline(0.09, color="red", linestyle="--", alpha=0.5, label="fallback threshold")
        self._add_phase_bg(axes[0], steps)
        axes[0].set_ylabel("error (m)")
        axes[0].legend(fontsize=8)

        axes[1].plot(steps, switches, linewidth=1.2, label="face_switch_count")
        self._add_phase_bg(axes[1], steps)
        axes[1].set_ylabel("count")
        axes[1].set_xlabel("step")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(self.out_dir, "contact_tracking.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        return [p]

    def _plot_box_orientation(self) -> List[str]:
        steps = self._steps()
        eulers = [r.get("box_euler_deg", [0, 0, 0]) for r in self.records]
        labels = ["yaw (Z)", "pitch (Y)", "roll (X)"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            vals = [e[i] for e in eulers]
            ax.plot(steps, vals, linewidth=1.2)
            ax.axhline(10, color="red", linestyle="--", alpha=0.4)
            ax.axhline(-10, color="red", linestyle="--", alpha=0.4)
            ax.set_ylabel(f"{label} (deg)")
            self._add_phase_bg(ax, steps)
        axes[-1].set_xlabel("step")
        fig.suptitle("Box Orientation", fontsize=11)
        fig.tight_layout()
        p = os.path.join(self.out_dir, "box_orientation.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_action_profile(self) -> List[str]:
        steps = self._steps()
        actions = [r.get("action", [0]*6) for r in self.records]
        labels = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
        fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            vals = [a[i] if i < len(a) else 0 for a in actions]
            ax.plot(steps, vals, linewidth=0.8)
            ax.set_ylabel(label, fontsize=8)
            self._add_phase_bg(ax, steps)
        axes[-1].set_xlabel("step")
        fig.suptitle("MPPI Action Profile", fontsize=11)
        fig.tight_layout()
        p = os.path.join(self.out_dir, "action_profile.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_phase_timeline(self) -> List[str]:
        steps = self._steps()
        phases = [r["phase"] for r in self.records]
        phase_names = list(dict.fromkeys(phases))  # unique, ordered
        phase_cmap = {n: i for i, n in enumerate(phase_names)}
        colors_map = plt.cm.Set2(np.linspace(0, 1, max(len(phase_names), 3)))

        fig, ax = plt.subplots(figsize=(10, 2))
        for i, (s, p) in enumerate(zip(steps, phases)):
            ax.barh(0, 1, left=s, height=0.6, color=colors_map[phase_cmap[p]])

        # Event markers
        for r in self.records:
            if r.get("phase_transition"):
                ax.axvline(r["step"], color="red", linewidth=1.5, alpha=0.8)
            if r.get("semantic_revision"):
                ax.axvline(r["step"], color="orange", linewidth=1, alpha=0.6,
                           linestyle="--")

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors_map[phase_cmap[n]], label=n)
                   for n in phase_names]
        handles.append(Patch(facecolor="red", label="phase transition"))
        handles.append(Patch(facecolor="orange", label="revision"))
        ax.legend(handles=handles, fontsize=7, ncol=len(handles), loc="upper center")
        ax.set_yticks([])
        ax.set_xlabel("step")
        ax.set_title("Phase Timeline")
        fig.tight_layout()
        p = os.path.join(self.out_dir, "phase_timeline.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        return [p]

    def _plot_stiffness_convergence(self) -> List[str]:
        """Σ̂_xx(t) vs ground-truth k — the spring_press headline figure."""
        steps = self._steps()
        # sigma_eigenvalues is the sorted spectrum of the 3x3 estimator matrix.
        # The press-axis component dominates after a few contact steps; we plot
        # the maximum eigenvalue as a proxy for Σ̂ along the press direction.
        sigmas = [
            (max(r["sigma_eigenvalues"]) if r.get("sigma_eigenvalues") else None)
            for r in self.records
        ]
        sigmas = [v if v is not None else float("nan") for v in sigmas]
        k_true = None
        for r in self.records:
            if r.get("button_stiffness"):
                k_true = float(r["button_stiffness"])
                break
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, sigmas, linewidth=1.5, label=r"$\hat{\Sigma}_{\max}(t)$")
        if k_true is not None:
            ax.axhline(k_true, color="red", linestyle="--", linewidth=1.5,
                       label=f"$k_{{true}}$ = {k_true:.0f} N/m")
            ax.axhline(0.9 * k_true, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.axhline(1.1 * k_true, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        self._add_phase_bg(ax, steps)
        ax.set_xlabel("step")
        ax.set_ylabel("stiffness (N/m)")
        ax.set_title("Riemannian estimator convergence")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(self.out_dir, "stiffness_convergence.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        return [p]

    def _plot_in_band_ratio(self) -> List[str]:
        """% of steps with F_n inside [F_min, F_max] over time."""
        steps = self._steps()
        cum_in = np.zeros(len(steps))
        running = 0
        for i, r in enumerate(self.records):
            if r.get("in_band") or r.get("regime") == "in-band":
                running += 1
            cum_in[i] = running / float(i + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, cum_in * 100.0, linewidth=1.5)
        ax.axhline(50, color="gray", linestyle=":", alpha=0.6)
        self._add_phase_bg(ax, steps)
        ax.set_xlabel("step")
        ax.set_ylabel("% steps in band (cumulative)")
        ax.set_ylim(0, 100)
        ax.set_title("Force-band tracking ratio")
        fig.tight_layout()
        p = os.path.join(self.out_dir, "in_band_ratio.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        return [p]

    def _add_phase_bg(self, ax: Any, steps: List[int]) -> None:
        """Add light phase-colored background bands to any plot."""
        phases = [r["phase"] for r in self.records]
        phase_colors = {"approach": "#e0f0ff", "push_to_wall": "#fff8e0", "lift": "#e0ffe0"}
        prev_phase = phases[0]
        start = steps[0]
        for i in range(1, len(phases)):
            if phases[i] != prev_phase or i == len(phases) - 1:
                end = steps[i] if phases[i] != prev_phase else steps[i]
                ax.axvspan(start, end, alpha=0.3,
                           color=phase_colors.get(prev_phase, "#f0f0f0"))
                start = steps[i]
                prev_phase = phases[i]

    # ── Video ──

    def _write_video(self) -> Optional[str]:
        if not self.save_video or not self.frames:
            return None
        h, w = self.frames[0].shape[:2]
        path = os.path.join(self.out_dir, "rollout.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        return path

    # ── Summary ──

    def _build_summary(self, *, video_path: Optional[str], plots: List[str]) -> Dict[str, Any]:
        if not self.records:
            return {"num_steps": 0, "video": video_path, "plots": plots}
        final = self.records[-1]

        # Phase transitions
        transitions = []
        for i, r in enumerate(self.records):
            if r.get("phase_transition") and r.get("phase_transition_to"):
                prev = self.records[i - 1]["phase"] if i > 0 else "none"
                transitions.append({
                    "step": r["step"], "from": prev,
                    "to": r["phase_transition_to"],
                })

        # Revisions
        revision_steps = [r["step"] for r in self.records if r.get("semantic_revision")]

        # Max tilt
        max_tilt = max((_max_tilt(r) for r in self.records), default=0.0)

        # Contact timing
        contact_step = next(
            (r["step"] for r in self.records if r.get("contact_latched")), None
        )

        # Peak costs
        cost_keys = ["cost_height", "cost_contact", "cost_pose",
                     "cost_energy", "cost_force_upper", "cost_force_lower"]
        peak_costs = {
            k: float(max(r.get(k, 0.0) for r in self.records))
            for k in cost_keys
        }

        return {
            "num_steps": len(self.records),
            "success": bool(final.get("success", False)),
            "final_height": float(final.get("box_height", 0.0)),
            "final_box_pos": final.get("box_pos"),
            "final_box_quat": final.get("box_quat"),
            "max_normal_force": float(max(
                r.get("measured_force_normal", 0.0) for r in self.records
            )),
            "max_tilt_deg": float(max_tilt),
            "phase_transitions": transitions,
            "num_revisions": len(revision_steps),
            "revision_steps": revision_steps,
            "contact_latched_at_step": contact_step,
            "height_at_contact": float(
                self.records[contact_step]["box_height"]
            ) if contact_step is not None and contact_step < len(self.records) else None,
            "time_in_band": int(sum(
                1 for r in self.records if r.get("regime") == "in-band"
            )),
            "num_drop_events": int(sum(
                1 for r in self.records if r.get("drop", False)
            )),
            "peak_costs": peak_costs,
            "max_eef_to_contact_error": float(max(
                r.get("eef_to_contact_error", r.get("eef_to_contact", 0.0)) for r in self.records
            )),
            "face_switch_count": int(final.get("face_switch_count", 0)),
            "video": video_path,
            "plots": plots,
        }
