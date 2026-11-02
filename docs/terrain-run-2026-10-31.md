# First Isaac terrain PPO run (2026-10-31) — post-mortem

Run `2026-10-31_15-16-02-terrain-cuda-0`: task `Primus-Terrain-Direct`, 128 environments, 3000 updates, action =
raw torque (±18 N·m) on the 6 motors. The run artifacts were removed in the workspace cleanup; the task code is
kept at the Git tag `isaac-raw-torque-2026-10-31`. It was replaced by the residual RL in `sim/warp_rl/`.

**What happened:** the policy did not learn to balance. The recorded trajectory shows the landing gear coming
down (−97/+97° → 0) and the robot sitting on it, tilted 12–56°, base at ~0.28 m (nominal 0.38 m).

**Causes:**
- the gear servo position target was never set (Isaac Lab starts `joint_pos_target` at zero = gear down);
- lenient termination (height < 0.20 m, tilt > 60°) and a `cos(tilt)` reward + large alive bonus: sitting
  tilted still paid ~80% of the reward;
- a spawn-point curriculum left almost every environment after the stairs;
- raw torque with a growing exploration std (0.30 → 0.52) and no gravity compensation;
- the actor lacked the states the LQR uses (dx, θ, L0); default PhysX friction (0.5) on the terrain; the wheel
  as a cylinder without exact geometry.

These lessons apply when returning to Isaac (large steps + camera). Current direction:
[control_roadmap.md](control_roadmap.md).
