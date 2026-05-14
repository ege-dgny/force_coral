# FORTE Contact Inference Spec

## Objective

Improve contact-point selection robustness with a lightweight online belief layer that combines semantic proposals and force consistency.

## Runtime Pipeline

1. Start from semantic phase `contact_strategy`.
2. Generate top-K candidate strategies by perturbing:
   - `contact_standoff`
   - `contact_vertical_offset_scale`
3. Physics feasibility filter:
   - reject unreachable EEF-to-contact targets
   - reject overly low contact points
4. Score candidates using:
   - EEF-to-contact distance
   - force-band consistency error (`|measured_normal_force - band_mid|`)
   - wall-gap penalty
5. Pick best candidate and write it into active semantic config for this step.

## Contact Belief State

Belief fields:

- `mode`: `free`, `pre_contact`, `contact`
- `confidence`: `exp(-best_hypothesis_score)`
- `uncertain_steps`: increments while confidence is low

## Guarded Fallback Behavior

When `uncertain_steps >= 5`:

- Activate conservative action prior.
- Reduce execute action scale.
- Reduce MPPI refinement iterations.

This prevents aggressive unstable behavior while contact state is ambiguous.

## Artifacts Logged Per Step

- Selected and top contact hypotheses
- Contact belief mode/confidence
- Pose source (`foundationpose` or fallback source)
- Pre-action and post-action force measurements
- Cost terms from optimizer-consistent `compute_cost_terms()`
