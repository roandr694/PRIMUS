# PRIMUS: status and plan

*Updated 2026-11-17. History and reasoning behind each step: [experiments.md](experiments.md). Latest audit and
the v12 recipe: [v12_plan.md](v12_plan.md). Controller and roadmap details: [control_roadmap.md](control_roadmap.md).*

## Where we are

| area | state |
|---|---|
| CAD, robot built, MuJoCo model, LQR + VMC controller, automatic landing gear | done, validated in simulation |
| Residual RL for rough ground and pushes (blind) | trained on the old controller; replaced by the v12 always-on policy |
| Perception stack (D435i map, RPLIDAR, planner, edge guard, A/B buttons) | done in simulation and on the CPU stack |
| Stand-up after a fall | trained (98.4% of episodes), on CPU front/back falls work; some side falls stick |
| Skill teacher | **v11 = best teacher**: hops flush, walled 12-step flights 92–98% at 13–19 cm; asymmetric stance, never trained always-on |
| Always-on policy (assist + skills, one network) | **v12 training** since 2026-11-17 (`2026-11-17_12-08-25-skill-v12`, 9000 iterations, ~8.5 h; the first attempt's critic diverged, fixed, see [experiments.md §7](experiments.md)); implementation and checks in [v12_plan.md §8](v12_plan.md) |
| Camera student | v1 (v3 teacher) and a GRU student (v5) exist; rebuilt after v12 |
| Real robot | drivers, state estimation, actuator characterisation still open |

## v12 in one paragraph

One policy that always runs on top of the LQR. A `skill` flag picks the regime: **assist** (default; follows the
remote, helps balance, jump disabled, reduced authority, rewarded for riding at least as well as the LQR) or
**skill** (set by the planner after the pilot presses A; full authority and the jump). Mirror symmetry in PPO,
actions as common/differential pairs, falls priced by the height they come from, γ = 0.995, terrain variants with
nosing, landings and chained flights, per-type curriculum. Decisions taken for the owner are listed in
[v12_plan.md §6](v12_plan.md).

## Next, in order

1. v12 training: checks at 500 and 1500 iterations (hops alive in skill, none in assist, leg split falling), then
   the acceptance table of [v12_plan.md §1](v12_plan.md) with `skill_eval.py --suite skill / assist / cancel`,
   CPU videos, then a from-scratch run and a second seed with the same recipe.
2. **Student distillation** from v12 (camera map): short history vs ~1 s vs GRU, with the perception fixes P3–P6.
3. **Full-stack CPU tests**: rendered depth → map → planner → student → controller, plus guard, A/B and X.
4. **Real robot:** actuator characterisation (replace the estimates in `hardware/actuators.json`), weigh the robot,
   tyre radius, IMU behaviour during hops, real loop rate (S1–S5 of the plan); ONNX → TensorRT on the Jetson,
   moteus drivers and state estimation in `robot/`, reduced authority at first.
5. **Later:** banking in fast turns, front flip (needs measured actuators first).

## Decisions that shape the plan

| decision | reason |
|---|---|
| keep the realistic motor and hip model even though it costs performance | v7 numbers relied on motors the robot does not have |
| hop flush on the step with both wheels | more stable; one-leg climbing is not possible without hip abduction (confirmed by v7/v10/v11) |
| jump primitive stays, policy decides when/shape | pure learning never found the jump (v1, v8) |
| judge a policy only on 150+ robots per cell | v9's reported gain came from ~20 robots per cell and was noise |
| one always-on network with a skill flag (fallback: two networks) | the assist then also sees the terrain |
| docs in English, communication in Portuguese | project rule |

## Known limitations

- The jump clears 24–33 cm of wheel height with the realistic motors, with a small timing margin.
- Teacher results use a perfect height scan; the level camera sees the floor only from ~0.74 m ahead.
- Motor, mass, tyre and IMU numbers are estimates: every result is "simulation only" until measured.
- Experiments, checkpoints and videos live outside this repository (`experiments/`, `videos/`); the repository
  keeps code, docs and the small figures in `docs/figures/`.
