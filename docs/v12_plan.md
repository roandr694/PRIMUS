# v12: everything that has to be right before the next training

*2026-11-16, decisions 2026-11-17. Audit of the whole skill-training stack (environment, rewards, curriculum, simulation, perception
hand-off, PPO setup) against the product the owner wants. Trained 2026-11-17; results against this plan in
[experiments.md §7](experiments.md#v12-result-evaluated-2026-11-18-seed-123--unseen-geometry-random-motors-150-robots-per-cell). Earlier diagnosis:
[diagnosis_2026-11-16.md](diagnosis_2026-11-16.md).*

Evidence tags: **[M]** measured in this audit (data in `experiments/analysis_2026-11-16/`), **[C]** read in the
code, **[R]** reasoning that has not been tested.

## 1. The target

One policy that is **always running on top of the LQR**, with two regimes chosen by one input:

| | Assist (default) | Skill (after the pilot presses A on an offered feature) |
|---|---|---|
| Purpose | the robot drives like the LQR, but falls less | cross the feature in front: stairs, curb, gap, hurdle, low bar |
| Who commands | the remote: speed, turn rate, height, (pitch, roll, responsiveness) | the planner: speed and heading across the feature |
| Jump primitive | disabled in the controller | available |
| What it must not do | hop, change the stance, override the pilot's acceleration setting | fall, stall, drift into a wall |
| Cancel (B) | n/a | stops and holds where it is, even on a step |

Later the same policy is distilled to the D435i map (student), so everything the teacher uses must be something
the camera and the map can really provide.

Proposed acceptance numbers (to be agreed; today's values in section 2):

| Regime | Metric | v12 target |
|---|---|---|
| Assist | falls per minute with 1.2 m/s pushes, flat / rough / slope | ≤ 0.25 / 0.30 / 0.55 (today's v11 level; LQR alone 0.9–1.6) |
| Assist | pitch rate and mechanical power, no skill | ≤ 1.2 × LQR alone |
| Assist | leg split on flat, body roll, height error | ≤ 2°, ≤ LQR + 0.3°, ≤ 1 cm |
| Assist | jump-primitive starts | 0 |
| Skill | stairs 13–17 cm, 12 walled steps, up and down, nominal motors | ≥ 97% done, ≤ 2% falls |
| Skill | falls per step on stairs (what decides 20–50 steps) | ≤ 0.15% (≈ 93% over 50 steps) |
| Skill | same with the weak motor draw | ≥ 90% done |
| Skill | curbs ≤ 17 cm, gaps ≤ 38 cm, low bars ≥ 0.46 m | ≥ 95% done |
| Skill | wheels at takeoff / landing, body roll during a hop | ≤ 20 ms / ≤ 40 ms apart, ≤ 6° |
| Both | unseen geometry (other treads, nosing, widths) and CPU MuJoCo | within 5 points of the training terrain |
| Both | cancel half way up the stairs | holds without falling in ≥ 98% |

## 2. Where v11 stands against that target

Skills **[M]** (about 190–280 robots per cell, random motors unless said):

| | 13 cm | 16 cm | 19 cm |
|---|---|---|---|
| Stairs up, done / fell | 93 / 7 | 95 / 5 | 91 / 2 |
| Walled 12-step flight up, done / fell | 96 / 4 | 95 / 5 | 92 / 3 |
| 8-step flight up, done / fell | 88 / 12 | 82 / 18 | 78 / 12 |
| Stairs up, weak motors, done / fell | 87 / 13 | 77 / 23 | 75 / 8 |

- A walled flight at 16 cm loses about 0.44% of the robots per step. Over 50 steps that is roughly 20% falls.
- Walled flights at 7 and 10 cm: 36% and 22% falls (chassis hits a step; the policy hops only half of the steps).
- Hurdles 53–87%, wide gaps 36–92%.

Running v11 all the time as the assist **[M]** (flat / rough / slope tiles, pilot-like commands, pushes; 900
robots per terrain, 20 s):

| Flat ground | LQR alone | v11 always on |
|---|---|---|
| Falls per minute, 0.6 m/s pushes | 0.01 | 0.01 |
| Falls per minute, 1.2 m/s pushes | 0.89 | 0.22 |
| Speed error [m/s] | 0.31 | 0.11 |
| Turn-rate error [rad/s] | 0.050 | 0.107 |
| Height error [cm] | 0.3 | 2.9 |
| Body roll [deg] | 0.4 | 2.2 |
| Pitch rate [rad/s] | 0.05 | 0.16 |
| Mechanical power [W] | 9 | 21 |
| Leg split [deg] | 0.2 | 11.7 |
| Jump-primitive starts per robot in 20 s | 0 | 0.06 (1.25 when the speed command goes to 2.5 m/s) |

v11 already makes the robot four times harder to push over, but it rides worse than the LQR in every other
respect, hops while just driving, and reaches the commanded speed faster than the pilot's acceleration setting
allows.

## 3. What is wrong, by area

### 3.1 Interface and architecture

| # | Problem | Fix in v12 |
|---|---|---|
| A1 | The policy has no "skill" input. It decides by itself to hop whenever the scan looks like an obstacle, and it hops in normal driving. **[M][C]** | Add a `skill` flag to the observation. Flag off: the controller ignores the jump trigger. The planner sets it in EXECUTE. |
| A2 | Deployment runs the policy only during a skill and resets its history at the start (`perception/autonomy.py`); training always starts standing at the foot of the feature. The always-on use was never trained. **[C]** | Episodes that drive in assist, switch to skill at the feature, and switch back after it. No history reset. |
| A3 | The policy sees 3 of the 7 commands (speed, turn rate, height). Pitch reference (the planner uses −0.32 rad to look down), roll, and the responsiveness level are always 0 / 1.5 in training. **[C]** | Put all commands in the observation and draw them over the remote's real ranges. |
| A4 | On skill tiles the commands are forward 0.3–1.2 m/s with a heading hold; raw turn-rate commands, reverse, and fast driving exist only on the three easy tiles. **[C]** | Assist tiles: the full remote envelope (speed range, raw turn rate, reverse, stops). |
| A5 | The speed reward compares with the raw command, so the policy learns to out-accelerate the controller's ramp (speed error 0.31 → 0.11). The pilot's responsiveness setting stops meaning anything. **[M][R]** | In assist, reward tracking of the controller's ramped target. Owner decides (section 6). |
| A6 | Cancel during a skill is not defined for the policy. Stops are drawn at random times, never "flag off while on a step". **[C]** | Train it: flag off mid-feature means hold position on the current step. |
| A7 | Residual authority is the same everywhere (±150 N per leg, ±4 N·m per wheel, ±17 cm, ±20°). In assist that is far more than balance help needs, and on flat ground v11 uses half of it constantly. **[M]** | Assist: authority scaled down (about 0.3–0.4 of the skill ranges). Optional: an authority knob (0–1) drawn at random in training, for a careful first run on the robot. |
| A8 | The planner's skill speeds (stairs up 0.6, gap 1.0, no hurdle entry) and alignment tolerance (4°) differ from training (hurdles and wide gaps 0.8–1.6 m/s, spawn yaw ±17°). **[C]** | One table shared by planner and environment: speed and approach angle per feature, and the largest feature the policy is validated for (the planner offers nothing beyond it). |

### 3.2 Rewards

| # | Problem | Fix in v12 |
|---|---|---|
| R1 | Nothing ties left and right. Every teacher holds a fixed staggered stance (leg split v7 +19°, v8 −17°, v10 +16°, v11 +15°), on flat ground too. The LQR alone is symmetric (0.06°). **[M]** | Mirror symmetry in PPO (augmentation + mirror loss, RSL-RL has it) and a leg-split cost with no free zone. |
| R2 | The residual fights the base loops. On flat ground v11 asks a leg-length difference of 12 cm (a 40° roll reference, which saturates the roll loop at its 50 N limit) and cancels it with a 105 N force difference. The roll loop is left with almost no range in one direction. **[M][C]** | Actions as common-mode and differential pairs (height + roll reference, lean + split, wheel torque sum + difference, leg force sum + difference), small ranges on the differential ones, and a cost on them. |
| R3 | A fall costs 10; a 12-step flight pays about 90 for climbing and up to 60 for speed tracking. With a 2 s horizon (γ = 0.99) a fall near the end is cheap. The policy accepts 5–15% falls. **[C][R]** | Fall cost that grows with the height fallen from and with the skill flag on; γ = 0.995; a bonus for finishing a feature. |
| R4 | During a jump and 0.3 s after it, power, vertical speed and extension speed cost nothing. Hops are free, so the policy also hops on slopes and at speed. **[C][M]** | A small fixed cost per jump start; A1 removes them from assist. |
| R5 | `climb` pays for any higher ground under the wheels, whatever the command. **[C]** | Pay only with the skill flag on and in the commanded direction. |
| R6 | Nothing rewards landing square on the tread. Landing lag between wheels is about 100 ms; one wheel is on a different level 13–28% of the time on features. **[M]** | Cost on the two wheels standing on different levels and on the lag between touchdowns; bonus for landing with margin from the step edge. |
| R7 | Wheel spin is free. At the falls on low walled stairs the wheel-speed estimate reads 2.5–3.6 m/s with a command ≤ 1.0. Spin also corrupts the odometry the map depends on. **[M][R]** | Cost on wheel surface speed minus true speed while grounded. |
| R8 | Landing loads are barely priced (`impact` is 0.0005 per second of reward). **[M]** | Cost on the peak leg force and on the legs bottoming out at touchdown. |
| R9 | `upright` punishes the pitch the planner asks for when it looks down. **[C]** | Measure it relative to the pitch reference. |
| R10 | In assist nothing asks for "do no harm": no cost on leaving the commanded height (2.9 cm error), on pitch rate, or on power relative to the LQR. **[M]** | Assist terms: height error, pitch rate, residual magnitude (larger weight than in skill). |

### 3.3 Environment, terrain, curriculum

| # | Problem | Fix in v12 |
|---|---|---|
| E1 | One fixed terrain: 72 tiles, each a single shape. Treads are exactly 0.30 m (walled flights 0.28 / 0.30 / 0.32), hurdles always 10 cm thick, bars and trenches at fixed places. Evaluation uses the same shapes. **[C]** | Several variants per type and level (tread 0.25–0.34 m, rise jitter ±1 cm inside a flight, widths, thicknesses) and a held-out terrain seed for evaluation only. |
| E2 | No step nosing, no open risers, no landing between flights, no U-turn stairwell. A wheel can catch under a nosing; a camera sees through open risers. **[C][R]** | Nosing 0–3 cm; a share of flights with landings; U-turn landings (two flights joined by a 1 × 2 m landing). Open risers only in the student stage. |
| E3 | Longest flight is 12 steps; the requirement is 20–50. **[C]** | Chained flights with landings, and the falls-per-step metric. |
| E4 | The curriculum level belongs to the robot, but the terrain type is redrawn at every reset. A level earned on flat ground is used on hurdles. There is no real per-skill curriculum; hurdles end training with 55% falls. **[C][M]** | A level per robot and per type, plus a fixed share of uniformly random levels so the easy ones are never abandoned. |
| E5 | Assist terrain is three tiles (flat, ±3 cm rough, slope ≤ 15°). No cross slope, speed bumps, cables, 5–10 cm drops met with the guard off, or slippery patches; friction 0.5–1.25 only. **[C]** | Add them; friction down to 0.3. |
| E6 | Pushes in the skill environment are 0.3 m/s. **[C]** | Assist 0.6–1.2 m/s; skill 0.3–0.5 m/s, including on the stairs. |
| E7 | Spawn yaw on open stairs is ±17°, on walled flights ±9°; the planner aligns to 4°. **[C]** | ±10° everywhere (margin over the planner), fed from the A8 table. |
| E8 | Top levels of hurdles (24 cm) and wide gaps (80 cm) may be beyond the jump (it clears 24–33 cm of wheel height). They burn samples and teach falling. **[R]** | Measure the feasible envelope with the controller's best jump per motor draw; cap the levels there. |
| E9 | Stops on a step, reverse on a feature, and turning on a landing are barely or never sampled. **[C]** | Covered by A6 and E2. |

### 3.4 Simulation fidelity

| # | Problem | Fix |
|---|---|---|
| S1 | The 1 kHz base controller reads the true attitude and rates. Noise is added only to the policy's observation, and it is white. A real attitude filter is disturbed by the accelerations of a hop and a landing. **[C][R]** | Attitude error model in the loop: bias (±1°), error that grows with acceleration, 2–5 ms delay. Needs the real IMU's behaviour (owner). |
| S2 | Motor numbers are estimates and decide the result (16 cm stairs: 77% weak, 94% strong). No thermal model: a flight is 12+ maximum-effort pushes. **[M][C]** | Bench measurement (torque–speed, peak current, battery sag); log RMS torque per flight against the motor's continuous rating. |
| S3 | Mass 15.7 kg from CAD (the owner says the robot is lighter); tyre radius 0.1075 m in the sim, 0.113 m in CAD; rigid tyre. **[C]** | Weigh the robot and legs; measure the tyre; add tyre compliance to the randomisation. |
| S4 | No encoder offsets, no backlash. A leg-length calibration error shifts every hop. **[C]** | Per-episode hip offset (±0.5°) and a small backlash. |
| S5 | The real control loop rate and CAN latency are not known to this model (1 kHz ideal, 1–4 ms torque lag, 0–10 ms policy delay). **[C]** | Measure on the robot; randomise the loop period if it is slower than 1 kHz. |
| S6 | Policies are checked on CPU MuJoCo only through single videos. **[C]** | A standing test: 50 CPU episodes per key skill; it is the cheapest proxy of a different physics. |
| S7 | Contact buffers (48 contacts, 200 constraints per world) were sized before the walls existed; overflow was not checked. **[R]** | Count active contacts on walled flights before training. |

S1–S5 need data from the real robot. v12 can train on wider ranges without them, but the acceptance numbers stay
"simulation only" until they are measured.

### 3.5 Perception and the camera student

| # | Problem | Fix |
|---|---|---|
| P1 | The camera is level and about 0.41 m high: the floor is first visible about 0.74 m ahead. The teacher's scan covers −0.30 to 1.20 m, so half of it can only come from map memory. **[C][R]** | Teacher actor gets only what camera + memory could know (simulated visibility, no noise); the critic keeps the full scan. Closes most of the teacher–student gap before distillation. |
| P2 | Climbing, steps above the camera show only their risers; the teacher sees every tread 1.2 m ahead. **[R]** | Same fix as P1. First measure how much v11 loses with the scan masked that way. |
| P3 | The student's map uses a constant offset per episode (2 cm) and the true height. Real drift grows with distance and jumps at every hop, in x, yaw and height. **[C]** | Drift model: random walk per metre plus a step per hop; heights relative to the wheel contact. R7 helps here. |
| P4 | The student's map is built from the floor without the side walls, while the teacher's scan includes them. **[C]** | Build the map from the same surface (bug, small). |
| P5 | No camera latency, no depth holes. **[C]** | 1–3 policy steps of delay, random holes. |
| P6 | Assist must stay safe when the map is empty or wrong. **[R]** | Episodes with the scan fully or partly unknown (mask in the observation). |

### 3.6 Training setup and process

| # | Problem | Fix |
|---|---|---|
| T1 | γ = 0.99 at 50 Hz is a 2 s horizon for tasks of 12–20 s. **[C]** | γ = 0.995. |
| T2 | The lineage is v7 (old motors) → v10 → v11. From scratch on the realistic plant, v8 never found the hop. The recipe is not reproducible. **[M]** | v12 warm-starts from v11 (weights remapped to the new inputs and actions). After it passes, one from-scratch run with the same recipe. |
| T3 | One seed per run; the spread between seeds is unknown. **[C]** | Two seeds for the final recipe. |
| T4 | Evaluation has no assist suite, no asymmetry numbers, no falls per step, no unseen terrain, no CPU check. **[C]** | Add them to `skill_eval.py`; the acceptance table of section 1 becomes one command. |
| T5 | The mirror map (which observation and action entries swap or change sign) is unverified. A wrong sign would wreck the run. **[C]** | Test before training: two robots in mirrored states with mirrored actions must keep mirrored observations. |
| T6 | Everything since commit a1313b5 is uncommitted (v10 and v11 code, new evaluation, walled stairs); `experiments.md` and `status_and_plan.md` still carry wrong v9 claims. **[C]** | Commit and correct the docs before v12, so the run points to a revision. Needs the owner's approval. |

## 4. The v12 recipe

Observation: history frame + all 7 commands + `skill` flag + authority; privileged values; scan limited to what
is knowable (actor) and full (critic); ceiling scan.

Actions (12, same count): height, roll reference, lean, split, wheel torque sum, wheel torque difference, leg
force sum, leg force difference, jump trigger, extension, push, tuck. The trigger is ignored with the flag off.

Episodes: 60% assist-only on everyday terrain with the full remote envelope and strong pushes; 40% skill tiles
that start in assist, switch the flag at the feature, switch back after it, with a share of cancels half way.

Rewards: section 3.2. Terrain and curriculum: section 3.3. PPO: γ 0.995, mirror augmentation + mirror loss,
warm start from v11.

Order of work, with what stops the line at each step:

| Step | Work | Gate |
|---|---|---|
| 0 | Commit the current state; correct the docs (T6) | owner's approval |
| 1 | Mirror map + its test (T5); contact-buffer check (S7) | test passes |
| 2 | Evaluation suite: assist, asymmetry, falls per step, held-out terrain, CPU (T4, S6) | v11 and LQR baselines recorded |
| 3 | Scan-visibility test on v11 (P2) and feasible-jump envelope (E8) | decides how P1 and E8 are built |
| 4 | Interface: flag, commands, action pairs, authority, controller gate (A1–A8, R2); parity test of the controller | zero action still equals the LQR; v11 remapped reproduces its own numbers |
| 5 | Terrain variants, nosing, landings, chained flights, per-type curriculum, assist tiles (E1–E7) | LQR alone and remapped v11 evaluated on the new terrain |
| 6 | Rewards (R1–R10), γ (T1) | 300-iteration smoke run: every term has a sensible size, no term dominates |
| 7 | v12 training, about 10 000 iterations (~7 h) | at 500 and 1500 iterations: hops alive in skill, zero in assist, leg split falling; otherwise stop |
| 8 | Full acceptance (section 1), videos, then the from-scratch and second-seed runs (T2, T3) | section 1 table |

Steps 1–6 are code and short checks; only step 7 is a long training.

If v12 misses the assist ride-quality targets while the skills pass, the fallback is two networks (a small blind
assist policy and the skill policy) with a blended hand-over. One network is preferred because the assist then
also sees the terrain.

## 5. Not in v12

- Camera student (distillation) and the student comparison (short history, 1 s history, GRU): after the teacher
  passes, using P3–P6.
- Stand-up after a fall stays a separate policy on the X button.
- Banking in fast turns; front flip.
- Bench measurements (S1–S5): owner's side, in parallel.

## 6. Decisions (taken 2026-11-17; the owner delegated them)

1. **Speed tracking in assist (A5): follow the controller's ramp.** The reward compares the speed with the
   controller's ramped target, so the responsiveness level (smooth 0.5, normal 1.5, sport 5.0 m/s²) keeps its
   meaning. In skill mode the target is the planner's command through the same ramp.
2. **Remote envelope (A3, A4)**, from `sim/scripts/pilot.py`: speed ±1.5 m/s (trained to ±1.8 for margin), turn
   rate ±1.8 rad/s, the three responsiveness levels, leg height over the full range. Roll is not commanded by the
   remote (always 0). Pitch comes only from the planner (look down, −0.32 rad) and is drawn in training in
   0 … −0.35 rad.
3. **Stairs requirement: 13–17 cm is the requirement, 19 cm is a margin level.** The real stairs are about 17 cm
   and with the weak motor draw 19 cm may be a hardware limit. "Done" on long flights is measured as falls per
   step (target ≤ 0.15%) on chained flights.
4. **Cancel on the stairs (A6): hold on the current step.** Rolling back down a flight blind is riskier than
   stopping; the pilot can then reverse or ask for a descent.
5. **Hurdles and wide gaps (E8): keep them, capped at the jump's feasible envelope** (measured with the
   controller's best jump; levels above it are not sampled for training).
6. **Warm start (T2): from v11**, with the new inputs and actions remapped so that the remapped network
   reproduces v11 at the start. A from-scratch run with the same recipe follows only after v12 passes.
7. **Commit (T6): approved.** v10/v11 code and the corrected docs are committed and pushed before v12.

Scope cut for the first v12 run (to keep the change testable): S1 (attitude error model) is added as a simple
bias + delay; S2–S5 stay as today's randomisation until the robot is measured; P1–P2 (camera-visible scan for the
actor) are measured first and only applied if the loss is small (step 3); E2 open risers and P3–P6 belong to the
student stage.

## 7. Reproducing the measurements

```bash
cd experiments/analysis_2026-11-16
~/test/isaac-rl/bin/python asym_probe.py --checkpoint <run>/logs/model_N.pt --per 100      # left-right asymmetry, stair falls
~/test/isaac-rl/bin/python asym_probe.py --checkpoint <...> --symmetrize                   # equal left/right actions
~/test/isaac-rl/bin/python drive_probe.py --push 1.2 [--checkpoint <...>]                  # everyday driving, LQR vs policy
```

Results: `asym_v11.txt`, `asym_more.txt`, `drive_probe.txt` in the same folder.

## 8. What was implemented (2026-11-17)

Code: `control/policy_interface.py` (the interface, one place), `sim/warp_rl/skill_env.py` (rewritten),
`mirror.py` (symmetry + v11 remap), `skill_terrain.py`, `controller.py` (sensor model), `skill_eval.py`
(rewritten), `train.py`, `export_policy.py`, `control/skill_policy.py`, `perception/autonomy.py`, `planner.py`,
`student_env.py`, `sim/scripts/skill_world.py`, `record_skill_video.py`; tests `tests/test_mirror.py`,
`tests/test_remap.py`.

| # | Status | How |
|---|---|---|
| A1 | done | `skill` flag in the frame; the environment (and the robot wrapper) send the trigger as -1 when it is off |
| A2 | done | skill episodes start in assist for 0.5-2.5 s in 35% of them, go back to assist after an open flight is climbed (3 s hold on top); the robot wrapper never resets the history for a v12 policy |
| A3 | done | frame = speed, yaw rate, leg length, pitch, responsiveness / 5, ramped speed target, skill flag; roll is not commanded (decision 2) |
| A4 | done | assist commands: speed +-1.8, raw yaw rate +-1.8, leg 0.20-0.48, pitch 0..-0.35 (30%), the three responsiveness levels, every 2-6 s |
| A5 | done | `track_speed` compares with the ramped target (`policy_interface.ramp_accel`, measured on the controller) |
| A6 | done | 12% of skill episodes cancelled 1.5-9 s after the start (flag off, speed 0); `--suite cancel` measures it |
| A7 | done | assist: continuous actions x 0.5 (`ASSIST_AUTHORITY`); no random authority knob (not needed for the first robot run: the assist regime is already the reduced one) |
| A8 | done | `policy_interface.SKILLS` (speed, training speeds, validated level) used by the planner and the environment; spawn yaw +-10 deg |
| R1 | done | mirror augmentation + mirror loss (0.5); `leg_split` = split^2 with no free zone (-5 skill, -20 assist) |
| R2 | done | actions as pairs; differential ranges 0.08 m (12 deg roll ref), 0.20 rad, 2 N.m, 60 N; `action_diff` cost |
| R3 | done | fall = -15 (skill) / -25 (assist) - 20 per metre of ground height; gamma 0.995; +10 for finishing a flight |
| R4 | done | `jump_start` -0.5 per primitive start |
| R5 | done | `climb` only with the flag on and a forward command |
| R6 | partial | `wheel_levels` (wheels on levels > 3 cm apart while grounded); no touchdown-lag term (measured in the evaluation instead) |
| R7 | done | `wheel_slip` = (odometry speed - true speed)^2 while grounded |
| R8 | done | `leg_load` above 400 N estimated per leg; `bottom_out` |
| R9 | done | `upright` relative to the pitch command |
| R10 | done | assist weights: height error, pitch/roll rate, action size, power, split, all several times the skill ones |
| E1 | done | treads per side 0.26-0.34 m (long 0.25-0.31, walled 0.26-0.34 per flight), rise jitter +-1 cm, hurdle thickness 3-15 cm; evaluation on terrain seed 123 (training 0) |
| E2 | partial | nosing 1-3 cm (half of the open stairs, the first walled flight); a 0.9 m landing in the middle walled flight; no U-turn stairwell, no open risers (student stage) |
| E3 | partial | falls per step on the walled flights, projected over 50 steps; no physically chained 24-step flight yet |
| E4 | done | a level per robot and per type; 20% of episodes at a random level |
| E5 | done | `bumps` tile (cross slope 2-10 deg, speed bumps 2-10 cm, cables, plateau drops 3-8 cm); friction per episode 0.3-1.25 (assist), 0.5-1.25 (skill) |
| E6 | done | pushes +-1.0 m/s per axis (assist), +-0.4 (skill, stairs included), every 2-6 s |
| E7 | done | +-10 deg everywhere (`APPROACH_DEG`) |
| E8 | decided | not capped: v11 already crosses the top levels (hurdles 24 cm 53%, gaps 80 cm 36%), so they are feasible |
| S1 | partial | attitude bias +-1 deg per robot, in the controller and the observation; no acceleration-dependent error, no delay |
| S4 | done | hip encoder offsets +-0.5 deg per joint |
| S7 | done | contact buffer use logged (`Metrics/contact_buffer_used`): 6-7% on the training terrain |
| S2, S3, S5, S6 | open | need the robot (bench data) / a CPU test run |
| P1, P2 | deferred | the teacher keeps the full scan; handled at distillation with the map memory |
| P4 | done | the student's map sees the walls (as the teacher's scan) |
| P3, P5, P6 | deferred | student stage |
| T1 | done | gamma 0.995 |
| T2 | done | `--init-from-v11` (exact, `tests/test_remap.py`: 4e-6) |
| T4 | done | `skill_eval.py --suite skill / assist / cancel`, split and one-wheel columns, falls per step, held-out seed |
| T5 | done | `tests/test_mirror.py`: static permutation checks + mirrored rollouts (map error 0.17-0.36 of the wrong-map error) |
| T6 | done | commit 7adf7f8 |
| (new) | done | success ends are terminal, not time-outs (the first v12 run's critic diverged on them) |

Findings on the way:
- The robot model is mirror symmetric about the plane x = -0.0117 m of the base frame (masses, inertias, sites),
  but the cylinder-tyre contacts are not exactly symmetric in simulation: two mirrored robots drift apart after
  ~0.5 s of violent actions. The mirror is a near-symmetry; augmentation is still the right tool.
- v11 depends on its staggered stance: remapped with the narrow differential ranges it falls more (8 s on mixed
  levels: gaps 8 -> 17-23%, hurdles 26 -> 40-49%). v12 has to relearn the skills symmetric; the warm start keeps
  the hop timing.
