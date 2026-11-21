# PRIMUS: status and plan

*Updated 2026-11-20. History and reasoning behind each step: [experiments.md](experiments.md). Latest audit and
the v12 recipe: [v12_plan.md](v12_plan.md). Controller and roadmap details: [control_roadmap.md](control_roadmap.md).*

## Where we are

| area | state |
|---|---|
| CAD, robot built, MuJoCo model, LQR + VMC controller, automatic landing gear | done, validated in simulation |
| Residual RL for rough ground and pushes (blind) | trained on the old controller; replaced by the v12 always-on policy |
| Perception stack (D435i map, RPLIDAR, planner, edge guard, A/B buttons) | done in simulation and on the CPU stack |
| Stand-up after a fall | trained (98.4% of episodes), on CPU front/back falls work; some side falls stick |
| Skill teacher | **v11 = best teacher**: hops flush, walled 12-step flights 92–98% at 13–19 cm; asymmetric stance, never trained always-on |
| Always-on policy (assist + skills, one network) | **v13c = current best** (`2026-11-19_12-42-09-skill-v13c/logs/model_19496.pt`, v13b + more stair cancels; second v13 seed from the v12b lineage reproduces the recipe; v13b `2026-11-19_02-03-03-skill-v13b/logs/model_17997.pt`; seed 1 `2026-11-18_09-39-32-skill-v12b-seed1/logs/model_12999.pt` kept as fallback), by the tile-only evaluation: assist 0.06 falls/min on flat with pushes (LQR 0.29); gaps to 45 cm 94-100%, overhang 88-100% with a leg command of at most 0.36 m, curbs 68-100%, hurdles to 19 cm 69-85% (24 cm: 41%), wide gaps to 0.6 m 77-100%; narrow stairs 0.5-1.4% falls per step. It no longer refuses 0.7-0.8 m gaps: the detector must never offer them (it offers gaps ≤ 45 cm only). Remaining limits partly physical (24 cm hurdles, 0.7-0.8 m gaps). See [experiments.md §7](experiments.md) |
| Camera student | **student-v1** = default (from v13c; `2026-11-19_16-46-41-student-v1/logs/model_2999.pt`, exported): within a few points of its teacher on riding, stairs, overhang and gaps to 45 cm; behind on hurdles L5, wide gaps, narrow/long stairs up. student-v2 (from v13d, robot-order history) is a tie: better curbs/gaps/overhang/wide gaps, worse narrow stairs and cancel. The 1 s conv history beats a GRU student. Both run in the full CPU stack. See [experiments.md §7](experiments.md) |
| Home autonomy (semantic map, navigation, agent) | P0-P4 in the MuJoCo house + realistic Isaac apartments (branch `home-autonomy`): camera voxel map, EKF localisation (6 cm mean on the CMA tour), Isaac-rendered camera (detector recall 0 → 48 %), agent 11/16 in an apartment; navigation out of the furnished CMA living room is the open problem. See [home_autonomy_architecture.md §12](home_autonomy_architecture.md) |
| Real robot | drivers, state estimation, actuator characterisation still open |

## v12 in one paragraph

One policy that always runs on top of the LQR. A `skill` flag picks the regime: **assist** (default; follows the
remote, helps balance, jump disabled, reduced authority, rewarded for riding at least as well as the LQR) or
**skill** (set by the planner after the pilot presses A; full authority and the jump). Mirror symmetry in PPO,
actions as common/differential pairs, falls priced by the height they come from, γ = 0.995, terrain variants with
nosing, landings and chained flights, per-type curriculum. Decisions taken for the owner are listed in
[v12_plan.md §6](v12_plan.md).

## Next, in order

*RL policy work paused on 2026-11-20 by the owner's decision; the project moves to other areas. The policies below
are the ones to use until it resumes.*

**Policies to use now:** skill/always-on teacher v13c (`2026-11-19_12-42-09-skill-v13c/logs/model_19496.pt`,
trained with the duplicated history); camera student **student-v1** (`2026-11-19_16-46-41-student-v1/exported`,
obs_version 5) for the full stack and the robot. Kept as alternatives: v13d (robot-order history) and student-v2.

1. **Real robot:** actuator characterisation (replace the estimates in `hardware/actuators.json`), weigh the robot,
   tyre radius, IMU behaviour during hops, real loop rate (S1–S5 of the plan); ONNX → TensorRT on the Jetson,
   moteus drivers and state estimation in `robot/`, reduced authority at first.
2. **Full-stack CPU tests** with student-v1: guard, A/B and X, the house environment (`sim/house`), longer runs
   with a fall detector in `autonomy_sim.py` (it has none; results are judged from the final pose and the planner
   trace).
3. **Home autonomy** (semantic map of the house, room/object goals, navigation with replanning, LLM agent with
   voice, follow-me, carry mode): architecture and phases in [home_autonomy_architecture.md](home_autonomy_architecture.md)
   (approved 2026-11-20: ROS 2 on the Jetson, LLM only on board, phone/web voice first, the real robot maps
   with the Jetson + Pi alone).
4. **Later:** banking in fast turns, front flip (needs measured actuators first).

### When RL work resumes (notes from 2026-11-20)

- The student follows its teacher within a few points: gains must come from the teacher. Start from v13d (trained
  with the robot-order history; v13c is not), continue it with more weight on hurdles and narrow stairs, and keep
  v13d's gains (curbs, gaps, overhang, long stairs). Then distill student-v3 with `teacher_legacy_history=False`.
- Retrain with the measured actuators as soon as they exist (all numbers are on estimated motors).
- Keep the 1 s conv history for the student; the GRU student lost everywhere (experiments.md §7).
- Evaluation rules learnt the hard way: tile-only, leg cap 0.36, 150+ robots per cell, robot-order history; pick
  checkpoints by iteration number, never by a lexical sort; `skill_eval` must clear `p_cancel_by_type`.
- Still open: the lr floor of the adaptive schedule (std floor, larger `desired_kl` or fixed lr); hurdles above
  ~19 cm and gaps above 0.6 m are near the plant limit, do not chase them.
- Queue scripts and raw outputs: `experiments/analysis_2026-11-19/`, `analysis_2026-11-20/`, `student_v1_eval/`,
  `student_v2_eval/`, `student_gru_eval/`, `v13d_eval/`.

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
