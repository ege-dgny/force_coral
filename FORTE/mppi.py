"""Parallel MPPI planner for FORTE with warm-starting."""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import Any, Dict, Optional, Type

import numpy as np

from FORTE.env import ObjectCentricWrapper, build_inner_env


_worker_env = None
_worker_wrapper = None


def _init_worker(
    controller: str,
    control_freq: int,
    init_idx: int,
    problem_folder: str,
    task_name: str,
    wrapper_cls: Type[ObjectCentricWrapper],
    wrapper_kwargs: Dict[str, Any],
    action_mult: float = 10.0,
) -> None:
    del control_freq
    global _worker_env, _worker_wrapper, _worker_action_multiplier
    _worker_action_multiplier = action_mult
    _worker_env = build_inner_env(
        task_name=task_name, controller=controller,
        offscreen=True, gui=False,
        init_idx=init_idx, problem_folder=problem_folder,
    )
    _worker_wrapper = wrapper_cls(_worker_env, **wrapper_kwargs)


_worker_action_multiplier: float = 10.0


def _evaluate_rollout(args: Any) -> float:
    """Evaluate a rollout with matched action multiplier.

    Both workers and main loop use the SAME multiplier so MPPI cost
    is consistent with execution. The multiplier controls the
    exploration/precision trade-off:
    - Higher: faster robot movement, but less force magnitude control
    - Lower: finer force control, but slower approach
    """
    global _worker_env, _worker_wrapper, _worker_action_multiplier
    action_sequence, initial_state, runtime_data = args
    _worker_env.sim.set_state(initial_state)
    _worker_env.sim.forward()
    _worker_wrapper.configure_runtime(runtime_data)

    total_cost = 0.0
    for action in action_sequence:
        _worker_wrapper.step(_worker_action_multiplier * action)
        cost = _worker_wrapper.rollout_cost()
        total_cost += float(getattr(cost, "total", cost))
    return total_cost


class ParallelMPPI:
    """Multiprocessing MPPI with warm-starting."""

    def __init__(
        self,
        env_wrapper: ObjectCentricWrapper,
        *,
        wrapper_cls: Optional[Type[ObjectCentricWrapper]] = None,
        wrapper_kwargs: Optional[Dict[str, Any]] = None,
        horizon: int = 10,
        num_samples: int = 64,
        noise_scale: float = 1.0,
        action_multiplier: float = 10.0,
        controller: str = "OSC_POSE",
        control_freq: int = 20,
        num_workers: Optional[int] = None,
        seed: int = 0,
        init_idx: int = 0,
        problem_folder: str = "my_suite",
        task_name: str = "",
        warm_start_fraction: float = 0.5,
    ) -> None:
        if not task_name:
            raise ValueError("task_name is required")
        self.envw = env_wrapper
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.noise_scale = float(noise_scale)
        self.action_multiplier = float(action_multiplier)
        self.rng = np.random.default_rng(seed)
        self.position_scale = float(getattr(env_wrapper, "action_position_scale", 8.0))
        self.wrapper_cls = wrapper_cls or type(env_wrapper)
        self.wrapper_kwargs = dict(wrapper_kwargs or {})
        self.warm_start_fraction = float(warm_start_fraction)

        # Warm-start state: previous best trajectory
        self._prev_best_traj: Optional[np.ndarray] = None

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = int(num_workers)

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(
                controller, control_freq, init_idx, problem_folder,
                task_name, self.wrapper_cls, self.wrapper_kwargs,
                action_multiplier,
            ),
        )
        atexit.register(self.close)

    def compute_control(
        self,
        runtime_data: Optional[Dict[str, Any]] = None,
        action_prior: Optional[np.ndarray] = None,
        num_iters: int = 1,
    ) -> np.ndarray:
        """MPPI with optional iterative refinement.

        When num_iters > 1, runs multiple optimization passes. Each
        iteration uses the previous best trajectory as the mean for
        sampling, with progressively tighter noise. This implements
        CEM-like refinement within MPPI.
        """
        initial_state = self.envw.env.sim.get_state()
        scale = self.noise_scale / self.position_scale

        prior = np.zeros(6)
        if action_prior is not None:
            prior = np.asarray(action_prior, dtype=np.float64).ravel()[:6] * scale

        best_traj: Optional[np.ndarray] = None
        best_cost = float("inf")

        for iteration in range(max(1, int(num_iters))):
            # Noise shrinks each iteration (exploration → exploitation)
            iter_noise = 0.5 / (1.0 + 0.5 * iteration)

            if iteration == 0 and self._prev_best_traj is not None:
                # First iteration: warm-start from previous step
                n_warm = max(1, int(self.num_samples * self.warm_start_fraction))
                n_fresh = self.num_samples - n_warm

                shifted = np.zeros_like(self._prev_best_traj)
                shifted[:-1] = self._prev_best_traj[1:]

                warm_actions = np.tile(shifted, (n_warm, 1, 1))
                warm_noise = self.rng.normal(0, 0.15, size=warm_actions.shape) * scale
                warm_actions = np.clip(warm_actions + warm_noise, -scale, scale)

                fresh_noise = self.rng.normal(0, iter_noise, size=(n_fresh, self.horizon, 6)) * scale
                fresh_actions = np.clip(prior + fresh_noise, -scale, scale)
                actions = np.concatenate([warm_actions, fresh_actions], axis=0)
            elif best_traj is not None:
                # Subsequent iterations: refine around current best
                noise = self.rng.normal(0, iter_noise, size=(self.num_samples, self.horizon, 6)) * scale
                actions = np.clip(best_traj + noise, -scale, scale)
            else:
                # First iteration, no warm-start
                noise = self.rng.normal(0, iter_noise, size=(self.num_samples, self.horizon, 6)) * scale
                actions = np.clip(prior + noise, -scale, scale)

            tasks = [(actions[i], initial_state, runtime_data) for i in range(len(actions))]
            costs = self.pool.map(_evaluate_rollout, tasks)
            iter_best_idx = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_traj = actions[iter_best_idx].copy()

        self._prev_best_traj = best_traj
        return best_traj[0]

    def close(self) -> None:
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass
