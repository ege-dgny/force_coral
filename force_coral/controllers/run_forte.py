"""
FORTE Controller -- Energy-Aware MPPI with Riemannian Stiffness Estimation.

This controller extends the existing MPPI framework with:
1. Online stiffness estimation via Riemannian retraction on SPD(6)
2. VLM-based stiffness prior initialization
3. Energy cost: penalizes motion in stiff (contact) directions
4. Barrier cost: prevents force limit violations

Usage:
    python force_coral/controllers/run_forte.py

The original run_mppi.py and run_mppi_push_the_box.py are preserved unchanged.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import logging
logging.getLogger("robosuite").setLevel(logging.ERROR)

import numpy as np
import multiprocessing as mp
import atexit
import cv2
import json
import sys
from pathlib import Path

# Bootstrap force_coral
import force_coral

from force_coral.libero_ext.env_wrapper import SegmentationRenderEnv
from force_coral.libero_ext.init_loader import load_init_bundle_by_name
from force_coral.dynamics.estimator import RiemannianStiffnessEstimator
from force_coral.perception.vlm_interface import (
    TaskPhysicsParser,
    PhysicsConfig,
    default_physics_config,
)

import contextlib, io as _io
_silent = _io.StringIO()
with contextlib.redirect_stdout(_silent), contextlib.redirect_stderr(_silent):
    import robosuite as _robosuite


# ---------------------------------------------------------------------------
# Environment builder (reuses pattern from run_mppi_push_the_box.py)
# ---------------------------------------------------------------------------

def _canonical_bddl(problem_folder: str, task_name: str) -> str:
    return os.path.join(
        force_coral.get_data_path("bddl_files"),
        problem_folder,
        f"{task_name}.bddl",
    )


def build_inner_env(
    *,
    task_name: str,
    controller: str = "OSC_POSE",
    offscreen: bool = False,
    gui: bool = True,
    init_idx: int = None,
    problem_folder: str = "my_suite",
):
    """Build a MuJoCo environment from a saved initialization bundle."""
    if init_idx is None:
        raise ValueError("init_idx must be provided")

    bddl_file = _canonical_bddl(problem_folder, task_name)
    overrides, state = load_init_bundle_by_name(
        problem_folder=problem_folder,
        task_name=task_name,
        init_idx=init_idx,
    )

    env = SegmentationRenderEnv(
        bddl_file_name=bddl_file,
        robots=["Panda"],
        controller=controller,
        has_renderer=gui,
        has_offscreen_renderer=offscreen,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        camera_names=["frontview"],
        camera_heights=240,
        camera_widths=320,
        camera_depths=False,
        camera_segmentations="instance",
        **({"object_overrides": overrides} if overrides else {}),
    )
    env.robots[0].controller_config["control_ori"] = True
    env.seed(0)
    env.reset()
    env.set_init_state(state)
    return env


# ---------------------------------------------------------------------------
# ForteWrapper -- extends the simple wrapper with physics-aware costs
# ---------------------------------------------------------------------------

class ForteWrapper:
    """Environment wrapper with FORTE cost function (task + energy + barrier).

    Parameters
    ----------
    env : SegmentationRenderEnv
        MuJoCo environment.
    K : np.ndarray or None
        Current 6x6 stiffness matrix (updated from main process).
    force_limit : float
        Maximum allowed predicted force magnitude (N).
    lambda_E : float
        Weight for energy cost term.
    rho : float
        Weight for barrier cost term.
    task_cost_weights : tuple
        (w_dist, w_contact) weights for geometric task cost.
    """

    def __init__(
        self,
        env,
        K: np.ndarray = None,
        force_limit: float = 10.0,
        lambda_E: float = 1.0,
        rho: float = 100.0,
        task_cost_weights: tuple = (8.0, 2.0),
    ):
        self.env = env
        self.box_body_name = "block_1_main"
        self.box_body_id = env.sim.model.body_name2id(self.box_body_name)
        self.panda_eef_name = "gripper0_grip_site"
        geom_id = env.sim.model.body_geomadr[self.box_body_id]
        half_size = env.sim.model.geom_size[geom_id]
        self.half_extents = np.array(half_size)

        # FORTE-specific
        self.K = K
        self.force_limit = force_limit
        self.lambda_E = lambda_E
        self.rho = rho
        self.w_dist, self.w_contact = task_cost_weights

    def step(self, action, render=False):
        """Execute action: 6D (pos[3] + rot[3]) → 7D robot command."""
        action7 = np.zeros(7)
        action7[:3] = 8.0 * action[:3]       # position gain
        action7[3:6] = 0.5 * action[3:6]     # rotation gain
        action7[-1] = -1.0                    # gripper closed
        self.env.step(action7)
        return self.env.sim.data.body_xpos[self.box_body_id]

    def state_cost(self):
        """Compute FORTE cost: J_task + J_energy + J_barrier.

        Returns
        -------
        float
            Total cost for the current state.
        """
        box_pos = self.env.sim.data.body_xpos[self.box_body_id]
        eef_pos = self.env.sim.data.site_xpos[
            self.env.sim.model.site_name2id(self.panda_eef_name)
        ]

        # ---- J_task: geometric task cost (distance to wall + EEF alignment) ----
        dist_to_wall = 0.20 - (box_pos[1] + self.half_extents[1])
        dist_cost = dist_to_wall ** 2

        target_contact = box_pos + np.array(
            [0, -self.half_extents[1] - 0.025, -0.05]
        )
        contact_cost = np.linalg.norm(eef_pos - target_contact)

        task_cost = self.w_dist * dist_cost + self.w_contact * contact_cost

        # ---- J_energy + J_barrier (only if stiffness available) ----
        if self.K is None:
            return task_cost

        # Penetration vector: EEF position relative to object center
        delta_x = np.zeros(6)
        delta_x[:3] = eef_pos - box_pos

        # Energy cost: x^T K x (penalizes motion in stiff directions)
        energy_cost = self.lambda_E * float(delta_x @ self.K @ delta_x)

        # Predicted force from stiffness model
        F_pred = self.K @ delta_x
        F_norm = np.linalg.norm(F_pred[:3])  # force magnitude (ignore torque)

        # Barrier cost: soft CBF penalty
        barrier_cost = self.rho * max(0.0, F_norm - self.force_limit) ** 2

        return task_cost + energy_cost + barrier_cost

    def get_wrench(self) -> np.ndarray:
        """Read external contact wrench on the manipulated object.

        Returns
        -------
        np.ndarray, shape (6,)
            [Fx, Fy, Fz, τx, τy, τz] in world frame.
        """
        return self.env.get_body_wrench(self.box_body_name)

    def get_penetration_vector(self) -> np.ndarray:
        """Compute 6D penetration vector (EEF position - object position).

        Returns
        -------
        np.ndarray, shape (6,)
            [dx, dy, dz, 0, 0, 0] (rotation part zeroed for now).
        """
        eef_pos = self.env.sim.data.site_xpos[
            self.env.sim.model.site_name2id(self.panda_eef_name)
        ]
        box_pos = self.env.sim.data.body_xpos[self.box_body_id]
        delta_x = np.zeros(6)
        delta_x[:3] = eef_pos - box_pos
        return delta_x

    def update_inner_from_vision(self, pose, size=None):
        """Update inner world object pose from vision estimate."""
        joint_name = "block_1_joint0"
        qpos_addr, _ = self.env.sim.model.get_joint_qpos_addr(joint_name)
        self.env.sim.data.qpos[qpos_addr : qpos_addr + 7] = pose
        if size is not None:
            geom_id = self.env.sim.model.body_geomadr[self.box_body_id]
            self.env.sim.model.geom_size[geom_id] = np.array(size)
            self.half_extents = np.array(size)
        self.env.sim.forward()

    def sync_robot_from_real(self, real_env, *, include_vel=True, include_gripper=True):
        """Copy robot joint state from real env to this inner env."""
        try:
            robot = self.env.robots[0]
            if hasattr(robot, "_ref_joint_pos_indexes") and robot._ref_joint_pos_indexes is not None:
                idx = robot._ref_joint_pos_indexes
                self.env.sim.data.qpos[idx] = real_env.sim.data.qpos[idx]
            if include_vel and hasattr(robot, "_ref_joint_vel_indexes") and robot._ref_joint_vel_indexes is not None:
                idx = robot._ref_joint_vel_indexes
                self.env.sim.data.qvel[idx] = real_env.sim.data.qvel[idx]
            if include_gripper and hasattr(robot, "gripper") and robot.gripper is not None:
                if hasattr(robot.gripper, "_ref_gripper_joint_pos_indexes") and robot.gripper._ref_gripper_joint_pos_indexes is not None:
                    gidx = robot.gripper._ref_gripper_joint_pos_indexes
                    self.env.sim.data.qpos[gidx] = real_env.sim.data.qpos[gidx]
            self.env.sim.forward()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parallel MPPI with stiffness passthrough
# ---------------------------------------------------------------------------

_worker_env = None
_worker_wrapper = None


def _forte_init_worker(
    controller, control_freq, init_idx, problem_folder, task_name,
    force_limit, lambda_E, rho,
):
    """Initialize a ForteWrapper in each worker process."""
    global _worker_env, _worker_wrapper
    _worker_env = build_inner_env(
        task_name=task_name,
        controller=controller,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    _worker_wrapper = ForteWrapper(
        _worker_env,
        K=None,
        force_limit=force_limit,
        lambda_E=lambda_E,
        rho=rho,
    )


def _forte_evaluate_one(args):
    """Evaluate a single action sequence rollout with FORTE cost."""
    global _worker_env, _worker_wrapper
    action_sequence, initial_state, K_matrix = args

    # Set stiffness for this rollout batch
    _worker_wrapper.K = K_matrix

    # Reset to initial state
    _worker_env.sim.set_state(initial_state)
    _worker_env.sim.forward()

    total_cost = 0.0
    for a in action_sequence:
        _worker_wrapper.step(a)
        total_cost += _worker_wrapper.state_cost()
    return total_cost


class ForteParallelMPPI:
    """Parallel MPPI that passes the current stiffness matrix K to workers.

    Parameters
    ----------
    env_wrapper : ForteWrapper
        The inner-world wrapper (used only for its env reference).
    horizon : int
        Number of steps per rollout.
    num_samples : int
        Number of trajectories to sample.
    noise_scale : float
        Standard deviation of action noise.
    task_name : str
        Task name for building worker environments.
    force_limit, lambda_E, rho : float
        FORTE cost function parameters (passed to workers).
    """

    def __init__(
        self,
        env_wrapper: ForteWrapper,
        horizon: int = 10,
        num_samples: int = 64,
        noise_scale: float = 1.0,
        controller: str = "OSC_POSE",
        control_freq: int = 20,
        num_workers: int = None,
        seed: int = 0,
        init_idx: int = None,
        problem_folder: str = "my_suite",
        task_name: str = None,
        force_limit: float = 10.0,
        lambda_E: float = 1.0,
        rho: float = 100.0,
    ):
        self.envw = env_wrapper
        self.horizon = horizon
        self.num_samples = num_samples
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.scale7 = 8.0
        atexit.register(self.close)

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_forte_init_worker,
            initargs=(
                controller, control_freq, init_idx, problem_folder, task_name,
                force_limit, lambda_E, rho,
            ),
        )

    def close(self):
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass

    def compute_control(self, K: np.ndarray = None) -> np.ndarray:
        """Sample trajectories, evaluate with FORTE cost, return best first action.

        Parameters
        ----------
        K : np.ndarray or None, shape (6, 6)
            Current stiffness matrix to pass to all workers.

        Returns
        -------
        np.ndarray, shape (6,)
            Best first action.
        """
        initial_state = self.envw.env.sim.get_state()

        actions = (
            self.rng.uniform(-1, 1, size=(self.num_samples, self.horizon, 6))
            * (self.noise_scale / self.scale7)
        )

        tasks = [
            (actions[i], initial_state, K)
            for i in range(self.num_samples)
        ]
        costs = self.pool.map(_forte_evaluate_one, tasks)

        best_idx = int(np.argmin(costs))
        return actions[best_idx, 0]


# ---------------------------------------------------------------------------
# Main FORTE control loop
# ---------------------------------------------------------------------------

def run_forte(
    *,
    task_name: str,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    num_steps: int = 200,
    use_vlm: bool = False,
    show: bool = True,
    # Estimator parameters
    eta: float = 0.01,
    min_eigenvalue: float = 0.1,
    # Cost function parameters
    force_limit: float = 10.0,
    lambda_E: float = 1.0,
    rho: float = 100.0,
    # MPPI parameters
    horizon: int = 10,
    num_samples: int = 64,
    noise_scale: float = 1.0,
    # Sync parameters
    sync_robot: bool = True,
):
    """Run the FORTE controller.

    Parameters
    ----------
    task_name : str
        BDDL task name.
    init_idx : int
        Initialization state index.
    use_vlm : bool
        If True, query VLM for stiffness prior. Otherwise use default.
    show : bool
        If True, show OpenCV preview window.
    """
    print(f"[FORTE] Task: {task_name}")
    print(f"[FORTE] Init: {init_idx} | VLM: {use_vlm} | Steps: {num_steps}")

    # 1. Build environments
    inner_env = build_inner_env(
        task_name=task_name,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    inner_wrapper = ForteWrapper(
        inner_env, K=None,
        force_limit=force_limit, lambda_E=lambda_E, rho=rho,
    )

    real_env = build_inner_env(
        task_name=task_name,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    real_wrapper = ForteWrapper(
        real_env, K=None,
        force_limit=force_limit, lambda_E=lambda_E, rho=rho,
    )

    # 2. Initialize stiffness prior
    if use_vlm:
        try:
            from PIL import Image
            frame = real_env.sim.render(camera_name="frontview", width=512, height=512)
            frame = np.flipud(frame)
            img = Image.fromarray(frame)
            task_desc = task_name.replace("_", " ")

            parser = TaskPhysicsParser()
            physics_config = parser.parse_task(img, task_desc)
            print(f"[FORTE] VLM stiffness prior: {physics_config.stiffness_prior}")
            print(f"[FORTE] VLM constraints: {physics_config.constraints}")
        except Exception as e:
            print(f"[FORTE] VLM failed ({e}), using default config")
            physics_config = default_physics_config()
    else:
        physics_config = default_physics_config()

    # 3. Initialize Riemannian estimator from VLM prior
    estimator = RiemannianStiffnessEstimator.from_vlm_prior(
        physics_config.stiffness_prior,
        eta=eta,
        min_eigenvalue=min_eigenvalue,
    )
    print(f"[FORTE] Initial K diagonal: {np.diag(estimator.K).tolist()}")

    # Parse force limit from constraints (use first force constraint if available)
    for c in physics_config.constraints:
        try:
            if "force" in c.lower() and "<" in c:
                val = float(c.split("<")[1].strip())
                force_limit = val
                print(f"[FORTE] Force limit from VLM: {force_limit} N")
                break
        except (ValueError, IndexError):
            pass

    # 4. Initialize parallel MPPI
    mppi = ForteParallelMPPI(
        env_wrapper=inner_wrapper,
        horizon=horizon,
        num_samples=num_samples,
        noise_scale=noise_scale,
        controller="OSC_POSE",
        control_freq=20,
        num_workers=None,
        seed=42,
        init_idx=init_idx,
        problem_folder=problem_folder,
        task_name=task_name,
        force_limit=force_limit,
        lambda_E=lambda_E,
        rho=rho,
    )

    # 5. Logging setup
    log = {
        "step": [],
        "K_eigenvalues": [],
        "F_meas_norm": [],
        "cost_task": [],
        "cost_energy": [],
        "cost_barrier": [],
        "cost_total": [],
        "box_pos": [],
    }

    # 6. Control loop
    print(f"[FORTE] Starting control loop ({num_steps} steps)...")
    try:
        for t in range(num_steps):
            # a. Read contact wrench from real env
            F_meas = real_wrapper.get_wrench()

            # b. Compute penetration vector
            delta_x = real_wrapper.get_penetration_vector()

            # c. Update stiffness estimate
            K = estimator.update(F_meas, delta_x)

            # d. Set K on inner wrapper for MPPI cost evaluation
            inner_wrapper.K = K
            real_wrapper.K = K

            # e. Sync robot state from real → inner
            if sync_robot:
                inner_wrapper.sync_robot_from_real(real_env)

            # f. Sync object state from real → inner
            box_pos = real_env.sim.data.body_xpos[real_wrapper.box_body_id]
            box_quat = real_env.sim.data.body_xquat[real_wrapper.box_body_id]
            pose = np.concatenate([box_pos, box_quat])
            inner_wrapper.update_inner_from_vision(pose)

            # g. Compute optimal action via MPPI
            u = mppi.compute_control(K=K)

            # h. Execute on real env
            box_pos_t = real_wrapper.step(80.0 * u)

            # i. Compute cost breakdown for logging
            total_cost = real_wrapper.state_cost()

            # Log
            evals = np.linalg.eigvalsh(K)
            log["step"].append(t)
            log["K_eigenvalues"].append(evals.tolist())
            log["F_meas_norm"].append(float(np.linalg.norm(F_meas[:3])))
            log["cost_total"].append(total_cost)
            log["box_pos"].append(box_pos_t.tolist())

            if t % 10 == 0:
                print(
                    f"  Step {t:03d} | box={box_pos_t.round(3)} | "
                    f"cost={total_cost:.3f} | |F|={np.linalg.norm(F_meas[:3]):.2f} | "
                    f"K_eig=[{evals.min():.1f}, {evals.max():.1f}]"
                )

            # j. Render preview
            if show:
                frame = real_env.sim.render(
                    camera_name="frontview", width=320, height=240
                )
                frame = np.flipud(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("FORTE Simulation", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("[FORTE] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        real_env.close()
        mppi.close()

    # 7. Save log
    out_dir = os.path.join("my_runs", f"forte_{task_name}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "forte_log.json")
    # Convert numpy arrays to lists for JSON serialization
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[FORTE] Log saved to {log_path}")

    return log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TASK_NAME = "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side"
    run_forte(
        task_name=TASK_NAME,
        init_idx=0,
        problem_folder="my_suite",
        num_steps=200,
        use_vlm=False,
        show=True,
        # Estimator
        eta=0.01,
        min_eigenvalue=0.1,
        # Cost
        force_limit=15.0,
        lambda_E=0.5,
        rho=50.0,
        # MPPI
        horizon=10,
        num_samples=64,
        noise_scale=1.0,
    )
