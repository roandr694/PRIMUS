# PRIMUS: status and plan

*Updated 2026-11-18. History and reasoning behind each step: [experiments.md](experiments.md). Latest audit and
the v12 recipe: [v12_plan.md](v12_plan.md). Controller and roadmap details: [control_roadmap.md](control_roadmap.md).*

## Where we are

| area | state |
|---|---|
| CAD, robot built, MuJoCo model, LQR + VMC controller, automatic landing gear | done, validated in simulation |
| Residual RL for rough ground and pushes (blind) | trained on the old controller; replaced by the v12 always-on policy |
| Perception stack (D435i map, RPLIDAR, planner, edge guard, A/B buttons) | done in simulation and on the CPU stack |
| Stand-up after a fall | trained (98.4% of episodes), on CPU front/back falls work; some side falls stick |
| Skill teacher | **v11 = best teacher**: hops flush, walled 12-step flights 92–98% at 13–19 cm; asymmetric stance, never trained always-on |
| Always-on policy (assist + skills, one network) | **v13b = current best** (`2026-11-19_02-03-03-skill-v13b/logs/model_17997.pt`; seed 1 `2026-11-18_09-39-32-skill-v12b-seed1/logs/model_12999.pt` kept as fallback), by the tile-only evaluation: assist 0.06 falls/min on flat with pushes (LQR 0.29); gaps to 45 cm 94-100%, overhang 88-100% with a leg command of at most 0.36 m, curbs 68-100%, hurdles to 19 cm 69-85% (24 cm: 41%), wide gaps to 0.6 m 77-100%; narrow stairs 0.5-1.4% falls per step. Needs the planner to enforce `max_level` (it no longer refuses 0.7-0.8 m gaps). Remaining limits partly physical (24 cm hurdles, 0.7-0.8 m gaps). See [experiments.md §7](experiments.md) |
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

1. **v13** (from the 2026-11-18 analysis, [experiments.md §7](experiments.md)): evaluation and training no longer
   leak into neighbouring tiles; with the corrected protocol seed 1 is the best skill policy. v13a (seed 1
   continued with tile truncation, capped levels for wide gaps and hurdles, overhang leg cap; evaluated 2026-11-19):
   hurdles +19 pts at 19 cm and wide gaps +20 pts at 0.60 m, but the 42 cm overhang drops from 91 % to 76 % and low
   narrow stairs and cancel get worse. The curriculum was still rising, so v13b continued it for 3000 iterations.
   v13b (`2026-11-19_02-03-03-skill-v13b/logs/model_17997.pt`) keeps the hurdle and wide-gap gains, recovers the
   overhang (88 % vs 91 %) and the narrow stairs, and is the new skill policy, **on condition that the planner
   enforces `max_level`**: v13b no longer refuses wide gaps of 0.70–0.80 m, it falls into them. Cancel at the top
   stairs level is worse (27.6 % vs 14.5 % on narrow L5, few robots). Then the
   planner rules (lower the leg for a bar; offer gaps_wide only to 0.60 m and hurdles to 19 cm), and a second seed
   of v13. Still open: the lr floor of the adaptive schedule (std floor, larger `desired_kl` or fixed lr).
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
