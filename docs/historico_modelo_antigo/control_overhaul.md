# Control Overhaul — Investigation & Fixes

This document explains everything that was wrong with the simulation (swinging, not
stopping, only forward/backward working, links visibly disconnecting from the wheel) and
how each problem was diagnosed and fixed. It is meant to be read top-to-bottom by someone
who did not watch the work happen.

The project's goal is to run the **exact same control logic** as the reference robot
[`Skythinker616/foc-wheel-legged-robot`](https://github.com/Skythinker616/foc-wheel-legged-robot)
but on a **bigger robot** (≈13 kg, 0.215 m wheels, L0/L1/L2 = 0.14 / 0.22 / 0.35 m). The
control *structure* (LQR balance + VMC legs + cascade PIDs for yaw/roll/leg-length/leg-angle
+ ground detection) is faithfully reused; what had to change is everything that depends on
the robot's physical size.

---

## TL;DR — root causes

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Swinging, never settles, sluggish | LQR gains were the **small robot's**, copied verbatim and used at L0≈0.37 m (a meaningless cubic extrapolation of a curve fit over 0.04–0.14 m) | Re-derived the gain schedule for this robot (`scripts/lqr_design.py`), validated against the reference to 1e-8 |
| 2 | Half the leg did nothing; leg-height/leg-angle dead | VMC used `mj_jacSite`, which ignores the closed-loop constraint, so the **front hip got ~0 torque** | Constraint-projected Jacobian in `_side_vmc_torques` so **both** hips actuate |
| 3 | Robot too springy / oscillatory | Large **passive joint springs** (hip stiffness 90, knee 25) — a spring-mass oscillator fighting the controller | Removed the springs; motors do the work (as on the real robot) |
| 4 | Weight/feed-forward wrong | Decorative geoms had no `mass`, so MuJoCo filled them from default density → **15.7 kg** | `mass="0"` on hints/hubs → **12.96 kg** |
| 5 | Falls immediately when the crutches are removed | Wheel-torque **sign** opposite to the reference (this model's wheel spins about +Y) | `lqr_t_ratio = -1` |
| 6 | **Links disconnect from the wheel**, and robot **drifts backward** after a few seconds | The `connect` equality **anchor was given in world coordinates**, but MuJoCo expects it in **body1's local frame** → the solver held the *wrong* material points | Anchor = `(LOWER_LINK_LENGTH, 0, 0)` in `front_lower`'s frame |
| 7 | Yaw turned the wrong way and destabilized | Yaw differential sign + windup of the integrated yaw target | `yaw_dir = +1`, gentle yaw gains, and a `max_yaw_lead` clamp |
| 8 | "Lots of inertia", sluggish, big speed overshoot | LQR penalized control effort heavily and weighted position ≫ velocity | Re-weighted `Q` toward velocity, lowered `Ruu`, faster command ramp |
| 9 | **Legs swing in opposite directions when turning** | Yaw wheel-differential reacts back through the legs; the leg-angle loop was too weak to hold them aligned | Much stronger `leg_angle_pid` (splay 0.52 → 0.006 rad) |
| 10 | Falls when turning **while driving** | State used **world-X** position/velocity; once yawed ~90° it regulated the wrong axis | Wheel-encoder odometry (heading-independent), matching the reference |
| 11 | Commanding **roll** tips it over | Roll feedback **sign inverted** (only hidden because two wheels are laterally stable) | `roll_dir = -1`; roll kept as a robust auto-leveler |

Bug **#6 was the big one you first reported** — it caused *both* the visible linkage
separation *and* the backward drift. Bugs **#8–#11** are the second round ("still not great,
inertia, yaw/roll/height, legs move oppositely when turning"): the controls now work and the
robot feels responsive.

---

## 1. The LQR gains were for the wrong robot

The reference computes its balance gains in `matlab/sys_calc.m`: for each leg length `L0`
it linearizes a planar "wheel + variable-length inverted pendulum" model, discretizes it,
solves a discrete LQR, and fits each of the 12 gains as a cubic in `L0`. Those cubics
(`lqr_k`) were fit over **L0 = 0.04–0.14 m** for a **0.9 kg** robot with **0.02 m** wheels.

The original port copied those cubic coefficients verbatim and evaluated them at our
**L0 ≈ 0.37 m** — far outside the fit range. Cubic extrapolation there is physically
meaningless (the position gain alone jumps ~10× for no real reason), and the coefficients
still encode the *small* robot's masses, inertias, and wheel radius. That is the primary
reason the robot swung and never settled.

**Fix:** `scripts/lqr_design.py` reproduces `sys_calc.m` in Python (numeric linearization →
`c2d` → discrete LQR via SciPy → cubic fit) and runs it with **this robot's** parameters.

- `python3 scripts/lqr_design.py --validate` runs the pipeline with the *small* robot's
  parameters and checks that it reproduces the published `lqr_k.m` coefficients. It matches
  to ~1e-8, which proves the derivation is faithful — not an ad-hoc retune.
- `python3 scripts/lqr_design.py --write` regenerates `scripts/lqr_gains.py`, which
  `repo_control_policy.py` imports. Edit the `BIG_ROBOT` parameters and the `Q`/`Ruu`
  weights at the top of `lqr_design.py`, re-run `--write`, and the controller picks up the
  new gains automatically.

This is the "same control logic, re-derived for a bigger robot" the project intends.

## 2. The VMC could only drive half of each leg

Each leg is a closed **5-bar**: a front branch (hip → upper → lower) and a rear branch
(hip → upper → lower → carrier → wheel), joined by a MuJoCo `connect` equality between the
front-lower tip and the carrier. In the kinematic *tree*, the wheel hangs off the **rear**
branch only.

The old VMC mapped the desired leg force to hip torques with `mj_jacSite(wheel)`. But the
site Jacobian only knows the tree — it does not know about the equality constraint — so its
column for the **front** hip is ≈0. The front hip motor therefore received ≈0 torque, and
half of every leg was effectively dead. This is why "many controller functions didn't work":
leg-length and leg-angle control had no authority through the front hip.

**Fix:** `_side_vmc_torques` now builds a **constraint-projected Jacobian** — the MuJoCo
equivalent of the reference's analytic `leg_conv`:

1. Wheel site Jacobian `Js` (3×nv) via `mj_jacSite`.
2. Loop-closure constraint Jacobian `C = mj_jac(front_lower@anchor) − mj_jac(carrier@anchor)`.
3. Treat the hips as independent and the knees as dependent: `dknee/dhip = −lstsq(C[:,knees], C[:,hips])`.
4. Constrained wheel Jacobian `J = Js[:,hips] + Js[:,knees] @ dknee/dhip`, then `τ_hips = Jᵀ·F`.

Now both hips get correct torque, and the legs hold the body up (verified: with the
gravity feed-forward the body stays at z≈0.41 with no balance loop running).

## 3 & 4. Spring-mass oscillator and phantom mass

The MJCF had large passive joint springs (hip `stiffness=90`, knee `stiffness=25`), added
earlier as a crutch. A spring plus the leg mass is literally an oscillator, and it fought
the leg controllers. They were removed (`generate_mjcf.py`); damping was reduced to small,
motor-friendly values. The real robot has no such springs — the motors hold the pose.

Separately, the decorative geoms (deck plates, wheel hubs) had no `mass` attribute, so
MuJoCo assigned mass from the default density (1000 kg/m³) — inflating the robot to 15.7 kg
and corrupting the gravity feed-forward. Setting `mass="0"` on those geoms brings the model
to **12.96 kg**, matching the real robot.

## 5. Wheel-torque sign

With the crutches gone, the bare LQR/VMC fell immediately. The reference applies
`wheelTorque = −lqrOutT`; in this MuJoCo model the wheel joints spin about **+Y**, the
opposite sense, so the balance term needs the opposite sign. `lqr_t_ratio = -1.0` captures
exactly this frame difference (verified: with `+1` it falls in 0.26 s; with `−1` it
balances). The LQR itself is derived in the reference frame; this one knob absorbs the
convention mismatch.

## 6. The constraint anchor bug (the one you saw)

After the above, the robot balanced for ~2 s and then slowly drifted backward, and you
reported the **links visibly disconnecting from the wheel**. Logging the gap between the
front-lower tip and the carrier told the story:

```
 t     gap(mm)   efc_residual
0.00     0.1       1e-5
1.50     6.3       6e-4
2.00    24.0       7e-4
2.50   169.4       5e-4
2.75   293.3       3e-4     <- 29 cm apart, but solver says "satisfied"
```

The loop gap blew up to **293 mm** while MuJoCo's own constraint residual (`efc_pos`)
stayed tiny. That contradiction is the tell: **the constraint was holding the wrong pair of
material points.**

MuJoCo's `connect` `anchor` is specified **in body1's local frame**. The generator was
passing **world coordinates** (with a comment that wrongly assumed world). At `qpos0` the
mismatched points happen to nearly coincide (so the model loads and the residual is ~0), but
as the linkage moves, the constraint dutifully holds its (wrong) internal points together
while the *physical* tip and carrier separate. Because the simulated linkage no longer
matched the rigid linkage the VMC assumes, this produced **both** the visible separation
**and** the backward drift / slow oscillation.

**Fix** (`generate_mjcf.py`, `add_equality`): the connection point is the tip of the front
lower link, which in `front_lower`'s own frame is `(LOWER_LINK_LENGTH, 0, 0)` — exactly
where the loop site sits. With the correct anchor the loop gap stays **< 0.1 mm** for the
entire run, and the drift/oscillation disappears entirely (pitch holds within ~0.01 rad,
position within ~0.02 m). One fix, both symptoms gone.

## 7. Yaw

Yaw initially turned the wrong way and threw the robot into a near-fall. Two things:

- **Sign / stability:** the yaw differential's stable-feedback sign for this model is
  `yaw_dir = +1`. (The opposite sign turns yaw into positive feedback that diverges even at
  rest.) The earlier "wrong direction" was an artifact of an over-aggressive differential
  tumbling the robot, not correct tracking.
- **Windup:** a held yaw command integrates `target_yaw` far faster than a 13 kg robot can
  actually turn, so the error saturates the loop into a balance-fighting differential. Gentle
  yaw gains plus a `max_yaw_lead = 0.5` clamp (cap how far the yaw target may lead the actual
  heading) keep it stable.

Yaw now turns the **correct** direction and stays balanced.

## 8. Responsiveness ("lots of inertia")

The robot felt heavy and over-shot speed commands (a 0.3 m/s step peaked at 0.63 m/s and
oscillated). Cause: the LQR weighted **position (x) far above velocity (dx)** and penalized
control effort heavily (`Ruu=2`), so it lumbered toward a position target and overshot
velocity. Re-weighting `Q` toward velocity (`x:30, dx:500`) and lowering `Ruu` to 0.4 cut
the overshoot to ~37% and the settling time roughly in half, with lower pitch. The
acceleration ramp was also doubled so the heavier robot still starts/stops briskly.

## 9. Legs swinging in opposite directions when turning

This is the specific thing you saw. Logging a turn showed the **left and right leg angles
diverging to ±0.26 rad** (left swings forward, right back) while roll stayed ~0. Cause: the
**yaw wheel-differential reacts back through each leg** — driving the left wheel forward
torques the left leg backward, and vice versa. The leg-angle controller (which keeps the two
legs parallel) was far too weak and saturated. Strengthening `leg_angle_pid` (outer 12→60,
inner 2→12, limit 40→240) drops the splay from **0.52 rad to 0.006 rad** — the legs stay
put during turns.

## 10. Falling when turning *while* driving

Driving and turning together fell after ~6 s. Cause: the controller's forward
position/velocity state used **world-X** (`qpos[0]`/`qvel[0]`). That is fine going straight,
but once the robot has yawed ~90° its "forward" is world-Y, so the controller was regulating
the wrong axis and fought itself into the ground.

The reference never has this problem because it derives forward distance/speed from **wheel
encoders** (`stateVar.x = wheel_angle × R`) — the average wheel rotation is how far the robot
rolled along *its own* heading, regardless of which way it points. The port had substituted
`qpos[0]`. Restoring wheel-encoder odometry (`_forward_odometry`) fixes drive-while-turning
completely (now stable through full-circle turns).

## 11. Roll command tipped the robot over

Commanding a roll lean drove roll the **wrong way** (cmd +0.10 → actual −0.93, runaway).
The roll feedback **sign was inverted**; it only looked OK at roll=0 because the two
laterally-separated wheels are passively stable and hid the wrong sign. `roll_dir = -1`
fixes the sign. Roll is kept as a robust **auto-leveler** (it holds the body level through
turns and would level on a slope); precise *commanded* roll tracking on a laterally-stable
base is a knife-edge for a pure P-cascade, so that is intentionally gentle (add integral
action if you ever need to hold a large commanded lean).

## 12. Coordinated banking into turns (no more centripetal tip)

Aggressive curves tipped the robot over: centripetal force leaned it outward and the
**outside leg gave in**, which compounded the tip. Fix: **coordinated-turn banking** — the
robot now leans *into* the turn by the physically-correct angle `tan(φ)=a_lat/g`
(`a_lat = forward speed × yaw rate`, low-passed, capped at ~15°), which balances the
centripetal force and drives the **outside leg longer** so it resists instead of collapsing.

The lean is produced by feeding the bank angle as the **roll setpoint** to the (P-D, no
integral) roll loop. Two earlier attempts failed and inform the final design: an integral
roll loop **wound up and ran away**, and an open-loop differential-force feed-forward
**overshot** (roll ran to −0.6 rad and tipped *inward*). Setpoint tracking with rate (D)
damping settles at the bank without either failure mode. With it, turns that used to fall
(0.6–0.7 m/s at 1.5–2.0 rad/s) now survive and visibly bank (≈5° into the turn, symmetric
left/right, scaling with how hard you turn). `banking_enabled` toggles it.

## 13. Responsiveness + snappier yaw

Re-weighted the LQR further toward velocity and lowered the control penalty
(`Q.x 30→45, Q.dx 500→550, Ruu 0.4→0.25`) for quicker speed response. With banking now
protecting the turn envelope, yaw authority was roughly doubled (`yaw_pid` limits up,
`max_yaw_lead 0.6→0.7`): yaw reaches 90 % of a step in **~0.5 s** instead of ~1.2 s, and
still survives the hardest turns.

## 14. Controller mapping (yaw axis + latching height)

- **Yaw axis** moved to the right-stick **horizontal** axis (SDL/macOS Xbox `axis 2`) instead
  of the vertical `axis 3`, which felt wrong for steering. (`run_xbox_viewer.py --yaw-axis`).
- **Ride height now latches**: the triggers integrate a **persistent** height offset (hold
  RT to raise / LT to lower, release to hold) instead of only adjusting while held.
  Start/Menu resets it. (`xbox_controller.py`, `height_rate`).

## 15. Terrain / obstacle course

The world is a 20 m arena (`generate_mjcf.py`): a large **flat play disc** (radius 4.6 m)
around the spawn, blended into a ring of **rolling hills** and rougher far-field, all from a
fine height-field (`assets/terrain_heightmap.png`, regenerated with the model). On top of
the flat disc are explicit obstacles:

- **Ramp → platform → ramp** hill ahead (drivable; the robot climbs the 0.30 m rise).
- **Staircase** (6 concrete steps) off to one side.
- **Surface-type patches** with genuinely different friction and colour: **ice** (μ≈0.04,
  slippery), **grass** (μ≈1.7, grippy), **sand** (μ≈0.85, high torsional), **metal**
  (μ≈0.30). These exercise the controller on varied grip.
- **Speed humps** (washboard) and **boulders**.

Plus a sky gradient, haze, and shadowed lighting for a nicer-looking scene. Contact detection
was made terrain-agnostic (`_touching_ground` now treats any tire contact as grounded, not
only the geom named `ground`), so the balancer keeps wheel control on ramps, stairs, and
patches. The robot drives the course (climbs the ramp, crosses the friction patches) at
nominal/low ride height; raised height (higher CoM) is less stable on the rough far-field.
Regenerate the model **and** heightmap with `python3 scripts/generate_mjcf.py`.

## 16. Responsiveness, suspension, and controller feel

Round-4 changes from driving feedback ("too much inertia, slow to accelerate/brake, can't
push through terrain; legs should suspend bumps; yaw inverted; height triggers dead"):

- **Less inertia / snappier drive.** The reference's tight target clamps were the main cause
  of the sluggish feel — they only let the position/speed *targets* lead the actual state by
  ±0.1 m / ±0.3 m/s, so the controller never commanded much accel/brake torque. Widened to
  `pos_lead 0.25→0.18 m` (relative to before this round it went 0.12→0.18) and
  `speed_lead 0.4→0.7 m/s`, tripled the command ramp (`accel_step`), and lowered the LQR
  control penalty (`Ruu 0.25→0.22`). Result: 0→0.63 m/s in **~0.68 s** (was ~1.2 s), and it
  brakes in ~0.85 s.
- **Snappier yaw.** Yaw authority and lead raised (`yaw_pid` limits up, `max_yaw_lead 0.7→1.0`):
  a yaw step reaches 90 % in **~0.4–0.7 s** (was ~1 s). Pushed only as far as the turn stays
  stable — hotter yaw made hard turns fall.
- **Legs as active suspension.** The leg-length *target* is trimmed to hold the chassis at a
  slowly-tracked reference height (`susp_*` knobs): fast bumps make the strong leg loop
  shorten/extend to keep the body steady, while slow height changes (ramps) are followed.
  (A plain vertical-force damper did almost nothing — the stiff leg-length loop just absorbed
  it; trimming the *target* uses that strong loop to actually hold height.) Cuts chassis
  bounce over the speed humps by ~15–25 %, on top of the roll auto-leveler's lateral
  stabilization. `susp_height_gain = 0` disables it.
- **Controller:** yaw is **inverted by default** (right-stick-right turns right;
  `--no-invert-yaw` to undo); **ride height now also works on the D-pad** (up/down), summed
  with the triggers so it works regardless of trigger-axis mapping, with a larger range and
  faster rate.

> Trade-off note: the aggressive settings raise the resting pitch jitter on the bumpy
> height-field from ~0.06 to ~0.09 rad (it does not fall) — deliberately biased toward
> responsiveness per the request. Back off `accel_step`/`speed_lead`/`Ruu` for a calmer feel.

---

## Verification (current behavior)

All headless, via `repo_control_policy` defaults (flat spawn zone unless noted):

| Test | Result |
|------|--------|
| Balance (on height-field spawn disc) | upright; ~0.09 rad pitch jitter from contact noise, no drift |
| Accelerate 0 → 0.63 m/s | ~0.68 s (was ~1.2 s) |
| Brake 0.7 → 0 m/s | ~0.85 s |
| Yaw step | reaches 90 % in ~0.4–0.7 s (was ~1 s), correct direction |
| Hard turn L / R (0.6 m/s, 1.5 rad/s) | **survives, banks ∓0.08 rad into the turn**, legs aligned |
| Very hard turn (0.7 m/s, 2.0 rad/s) | survives, banks into the turn |
| Speed-hump washboard | chassis bounce cut ~15–25 % by leg suspension |
| Ramp / obstacle course | climbs the 0.30 m ramp, crosses friction patches, upright |

## Files changed / added

- `scripts/lqr_design.py` — **new.** Re-derives the LQR schedule (port of `sys_calc.m`).
  `--validate` checks it against the reference; `--write` regenerates the gains.
- `scripts/lqr_gains.py` — **new, generated.** The big-robot gain schedule (do not edit by
  hand; regenerate with `--write`).
- `scripts/repo_control_policy.py` — table-driven `lqr_k`, constraint-projected VMC,
  wheel-encoder forward odometry (`_forward_odometry`), strong leg-angle loop, coordinated
  banking (setpoint-tracked roll), snappier yaw, retuned LQR/PIDs, frame-convention sign
  knobs (`lqr_t_ratio`, `yaw_dir`, `roll_dir`, `wheel_odom_sign`, `bank_sign`),
  `max_yaw_lead`/`max_bank` clamps, `sim_stabilizer` off.
- `scripts/generate_mjcf.py` — corrected `connect` anchor, removed joint springs,
  zeroed decorative-geom mass, updated dimensions/masses, **height-field terrain** + heightmap.
- `scripts/xbox_controller.py` / `run_xbox_viewer.py` — yaw on right-stick horizontal axis,
  latching (persistent) trigger ride-height.
- `scripts/diagnose.py` — **new.** Headless trajectory logger for tuning.
- `docs/model_notes.md` — updated; this file (`docs/control_overhaul.md`) added.

## How to tune further (when you have measured values)

1. Put measured masses/inertias/wheel radius into `BIG_ROBOT` in `scripts/lqr_design.py`;
   adjust `Q` (state weights: θ, dθ, x, dx, pitch, dpitch) and `Ruu` (control effort).
2. `python3 scripts/lqr_design.py --write` to regenerate `lqr_gains.py`.
3. `python3 scripts/diagnose.py --seconds 8 --speed 0.3` (and `--yaw`, `--leg`) to inspect.
4. Re-tune the cascade PIDs in `repo_control_policy.py` only after the LQR is set.
5. The `sim_stabilizer_*` knobs remain as an optional fallback but are off by default and
   should stay off now that the LQR is correct.

---

## 17. The "still swinging a lot" round — measured & fixed

Reported symptom: *"the robot is still swinging a lot."* This round replaced eyeballing
with measurement, found three concrete swing sources, fixed them, and confirmed the result
quantitatively. Two new tools made this possible:

- `scripts/analyze_balance.py` — runs a scenario headless and reports the **swing metrics**:
  pitch RMS / peak-to-peak, body-z bounce, **airborne fraction** (wheels off ground →
  balance torque off), hip/wheel torque RMS & peak, oscillation frequency, and drift. One
  number per change instead of watching the viewer.
- `scripts/generate_mjcf.py --flat` → `models/pro_wheel_legged_robot_flat.xml` — a clean,
  obstacle-free plane. **Essential for tuning:** the demo arena's ramp starts only 2 m ahead
  of spawn, so a forward-drive test climbs it and pollutes every metric (see below).

### Method note: don't measure the controller on the obstacle course

The first "drive forward" baseline showed a **151 mm vertical "bounce."** A physical probe
(`whlZ` climbing 0.108 → 0.244 m starting exactly when `x ≈ 2.0 m`) showed it was **not a
controller fault at all** — the robot was correctly *climbing the `ramp_up` geom* that begins
at x = 2.0 m. All drive measurements now use the flat model. Lesson: separate "does it
balance/track well" (flat) from "does it handle terrain" (arena).

### Root cause A — coordinated banking was a pitch limit cycle (biggest)

The Round-12 coordinated-turn banking estimated lateral accel as `v·yaw_rate`, low-passed it,
and tracked it as a roll setpoint. Both `v` (wheel odometry) and `yaw_rate` are oscillatory,
and the resulting differential leg force couples roll → **pitch** on the closed 5-bar leg.
The loop self-sustained a ~0.8 Hz pitch oscillation:

| drive + turn (0.4 m/s, 1.0 rad/s), flat | pitch peak-to-peak |
|---|---|
| banking ON (old default) | **9.8°** |
| banking OFF | 1.6° |

Crucially, banking provided **no fall protection** any more: hard turns
(0.5–0.8 m/s @ 1.5–2.0 rad/s) stay upright *without* it (this is a laterally-stable 0.37 m
track two-wheel robot, not a single-track that tips). **Fix: `banking_enabled = False` by
default** (knob kept for a future narrow-track variant).

### Root cause B — the balance controller switched OFF on every terrain micro-hop

On the faceted height-field terrain the two tires micro-hop (often on alternate steps). The
old `_touching_ground` required **both** tires in contact *this step*, and the code zeroed the
wheel balance torque whenever that was false — an on/off controller. At rest on the real
arena that meant **12% airborne** and **20 Nm RMS hip chatter** at ~0° pitch (the visible
leg buzz). **Fix: per-tire ground hysteresis** (`air_grace_steps`, ~60 ms): a tire counts as
grounded if it touched within the grace window, so transient hops don't toggle the
controller; a genuine sustained lift-off still gates the torque as a safety.

### Root cause C — the LQR barely damped the leg-swing mode

After any push/bump the legs rang fore/aft (`theta`) at ~0.7 Hz for ~4.5 s (7 oscillations).
The Round-13 "responsiveness" weights `Q = [1,1,45,550,600,1]` had cranked `dx` to 550 for a
snappy speed response but left **`dTheta` at 1** — the leg-swing mode was almost undamped.
Re-balanced to `Q = [8, 50, 70, 250, 800, 4]`, `Ruu = 0.4` (damp the leg swing via
theta/dTheta, ease the excessive dx, add pitch-rate damping, smoother torque). Verified by an
offline Q/R sweep scoring post-push settling **and** speed step-response so responsiveness was
not sacrificed.

### Before → after (settled-window metrics)

| Scenario (real terrain arena) | Before | After |
|---|---|---|
| Rest — hip-torque RMS / airborne | 20.4 Nm / 12.5 % | **6.8 Nm / 0 %** |
| Rest — hip-torque peak | 54 Nm | **8 Nm** |
| Turn 1.0 rad/s — pitch p2p / airborne | 0.78° / 21 % | **0.30° / 0 %** |
| Drive + turn — pitch p2p | 9.8° | **0.40°** |
| Push 120 N·0.1 s — settle / oscillations | 4.58 s / 7 | **3.34 s / 4** |
| Push — pitch peak | 4.05° | **3.17°** |
| Hard turn 0.7 m/s @ 2.0 rad/s — pitch p2p | 13–16° | **0.5°**, upright |
| Speed step 0→0.6 m/s — overshoot / rise | 0.57 / 0.71 s | **0.41 / 0.85 s** |

### Files changed / added (this round)

- `scripts/analyze_balance.py` — **new.** Quantitative swing metrics (used for every number
  above). `--flat` measures on the clean plane.
- `scripts/generate_mjcf.py` — **new `--flat`** flat-plane model for measurement; the arena
  model is unchanged.
- `scripts/repo_control_policy.py` — `banking_enabled = False`; per-tire ground hysteresis
  (`air_grace_steps`, `_air_steps`) in `_touching_ground`.
- `scripts/lqr_design.py` — re-balanced `BIG_ROBOT["Q"]`/`["Ruu"]`; `scripts/lqr_gains.py`
  regenerated (`--write`). Pipeline still validates against the reference (`--validate`, 5.7e-8).

### Reproduce

```bash
python3 scripts/generate_mjcf.py --flat                 # one-time: build the flat model
python3 scripts/analyze_balance.py --speed 0.4 --yaw 1.0          # terrain
python3 scripts/analyze_balance.py --flat --speed 0.6 --yaw 1.5   # flat, hard turn
```

---

## 18. Capability round — top speed, yaw feel, height range, hard brake

After the anti-swing round, four capability questions were measured (all on the flat model
unless noted) and addressed.

### Speed — was artificially capped, raised

On flat ground (no aero/rolling drag in sim) the robot tracks **every** commanded speed up
to 5 m/s with no fall, using only **~2 Nm** of the 140 Nm wheel-torque budget — it was never
near a dynamic limit, just the controller's `max_speed = 1.0`. Raised the Xbox cap to
**2.5 m/s** (`--max-speed`); it tracks well beyond that if you raise the flag. Acceleration is
governed by the LQR speed loop, *not* the command ramp — sweeping `accel_step` 0.010→0.080
barely moved the 0→0.8 m/s rise (0.97→0.86 s), so the ramp was left alone (raising it just
adds overshoot/lurch without real benefit). 0→0.8 m/s in ~0.9 s, 0→1.5 in ~1.3 s.

### Yaw — was 185 % overshoot, now feedforward + damped rate loop

Yaw is a *rate* command but the old controller integrated it into a *heading* target and
chased it with a stiff P (outer kp 5, heading lead 1.0 rad). Commanding 1.0 rad/s **overshot
to 2.85 rad/s** and the steady rate drifted high (1.11). Rebuilt as **rate feedforward + a
damped rate loop**: the commanded rate goes straight to the rate-loop setpoint, the heading
loop is pinned to actual heading *while turning* (so its correction can't stack on the
feedforward — that double-count was the overshoot) and only **holds heading when released**;
a small rate-loop integral (`ki = 0.02`) removes the steady-state error.

| yaw cmd | old overshoot / steady | new overshoot / steady / rise |
|---|---|---|
| 1.0 rad/s | +185 % / 1.11 | **+2 % / 1.00 / 0.43 s** |
| 1.8 rad/s | +58 % / 1.80 | **+8 % / 1.80 / 0.05 s** |

### Ride height — range nearly doubled

The robot was measured upright and length-tracking from leg **0.24 m → 0.52 m** (base height
**0.33 → 0.62 m**, close to the real Ollie's 0.33–0.7 m), far past the old ±0.09 m command cap
(0.28–0.46). Widened the LQR gain schedule sweep to `L0 ∈ [0.24, 0.52]` (cubic fit error
0.016, pipeline still validates) so the whole range has real gains instead of edge-clamped
extrapolation, made the leg-force cap a parameter (`max_leg_length = 0.52`, separate from the
airborne-retract length), and raised the Xbox ride-height range to **±0.15 m**
(`--max-leg-adjust`).

### Hard brake — new button

The "stop" button (A) only zeroed the command and let the speed coast down (the `speed_lead`
anti-windup clamp pins the target near the actual speed, so from 1.5 m/s it coasted **1.56 m
in 1.38 s**). Added a dedicated **hard brake** (`RobotCommand.brake`, Xbox **B**): it snaps the
speed target straight to 0, pegs the position target to the current odometry (no stale
position push), and widens the anti-windup lead (`brake_speed_lead = 3.0`) so the full
velocity error reaches the LQR → maximum decel torque. From 1.5 m/s it now stops in **1.02 m /
0.73 s** (vs 1.56 m / 1.38 s) without tipping.

### Files changed (this round)

- `scripts/repo_control_policy.py` — `RobotCommand.brake`; yaw feedforward + heading-hold +
  rate-loop integral; brake logic in `_update_target`; parametrized `max_leg_length` /
  `air_retract_length` / `brake_speed_lead`.
- `scripts/lqr_design.py` — `L0_sweep` widened to [0.24, 0.52]; `scripts/lqr_gains.py` regenerated.
- `scripts/xbox_controller.py`, `scripts/run_xbox_viewer.py` — B = hard brake, `max_speed`
  1.0→2.5, `max_leg_adjust` 0.09→0.15, corrected the stale "auto-banks" help text.

### Verify

```bash
python3 scripts/lqr_design.py --validate && python3 scripts/lqr_design.py --write
python3 scripts/analyze_balance.py --flat --speed 2.0          # fast drive, rock-steady
python3 scripts/diagnose.py --seconds 6 --yaw 1.5              # clean yaw, no overshoot
```
