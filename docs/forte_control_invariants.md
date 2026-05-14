# FORTE Control Invariants

These invariants are required to preserve CoRAL-style control semantics in `FORTE/`.

## Loop Ordering

Per control step:

1. Read current force/kinematics.
2. Update semantic phase and runtime configuration.
3. Select contact hypothesis (proposal -> feasibility filter -> force-consistency rerank).
4. Observe object pose (`ground_truth` or `foundationpose`).
5. Sync inner robot from real env.
6. Apply observed object pose to inner env via `update_inner_from_pose`.
7. Run MPPI on inner env.
8. Execute chosen action on real env with matched action multiplier.
9. Compute monitor/revision signals using post-action measurements.

## Action and Dynamics Consistency

- Worker rollouts and real execution use the same `action_multiplier`.
- Wrapper action map remains:
  - translational: `8.0 * action[:3]`
  - rotational: `0.5 * action[3:6]`
  - gripper: from runtime contact strategy (`gripper_command`).

## Dual-World State Contract

- Inner robot state must be copied from real robot each step (`sync_robot_from_real`).
- Inner object state must be updated from observed pose each step (`update_inner_from_pose`).
- `sync_box_from_real` is only a fallback utility; observed pose update is the primary path.

## Cost Contract

- `rollout_cost()` is scalar and uses `compute_cost_terms()` internally.
- Logged cost terms come from the same function used by rollout optimization.
- Force barriers are based on rollout sim force (`sim_force_normal_rollout_cost`), not a different auxiliary objective.

## Monitoring Contract

- Monitor and semantic revision inputs must use post-action force and post-action state from the same timestep.
