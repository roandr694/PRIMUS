# Perception and autonomy

Everything between the pilot and the controller. The same code runs in the simulator (rendered depth, ray-cast
lidar) and is meant to run on the robot (librealsense / rplidar frames); only the sensor drivers differ.

```text
D435i depth (15 Hz) ──► elevation map (ground + ceiling, with memory) ──► feature detector (ahead / behind)
RPLIDAR A1 (5.5 Hz) ──► obstacles in the travel corridor                        │
pilot (sticks, buttons) ─────────────────────────────────────────────► skill planner + edge guard (50 Hz)
                                                                                │ speed, turn rate, brake
                                               skill policy (only during a skill) │ + residual (9 values)
                                                                                ▼
                                                             RepoControlPolicy (1 kHz, control/)
```

| file | what it does |
|---|---|
| `d435i.py` | camera mount, intrinsics, noise model (σ ∝ depth², ~1.4 cm at 2 m; dropouts; no matches at depth jumps), depth image → points |
| `elevation_map.py` | robot-centric 6 m × 6 m map, 4 cm cells, ring buffer (no copying when moving); per cell a Kalman-fused ground height and the lowest overhead surface (a ceiling must be seen in 3 frames) |
| `features.py` | samples the map on a corridor as wide as the robot: stairs (up/down: steps, rise, tread), curbs, drops, cliffs, gaps (width), overhangs (free height), walls; which skill crosses each one; heading perpendicular to an edge |
| `lidar.py` | RPLIDAR A1 model, scan → points, terrain returns removed with the map, nearest obstacle in the corridor, 2-D occupancy grid |
| `planner.py` | states PILOT / ALIGN / EXECUTE; edge guard; skill offers |
| `autonomy.py` | wires everything for the viewers and the robot |

## Using it

| button | Quest | Xbox (desktop browser) | keyboard |
|---|---|---|---|
| start the offered skill | A | A | Enter |
| brake / cancel the skill | B | B | space |
| edge guard on / off | hold B 1 s | Back (or hold B) | G |

- **Edge guard** (on by default): caps the pilot's speed so the robot stops 0.35 m (from the wheel axle) before
  any drop deeper than 6 cm, step higher than 5 cm, stairs, gap, wall, overhang lower than the robot, unseen ground
  close ahead, or lidar obstacle, in the direction of travel (the map remembers what is behind). It plans with the
  measured speed and 0.7 s of latency (perception + the controller's speed lag). It brakes only while moving: the
  controller's brake drops the position hold, so a robot held with it creeps.
- **Skills**: the nearest crossable feature is shown on the pilot panel ("stairs up 5x16 cm at 1.2 m - A to
  climb"). A: the robot stops, turns perpendicular to the edge, and crosses at the skill speed holding that heading;
  the skill policy runs only during this (normal driving stays model-based). It ends after the feature (odometry)
  or after 15 s; B cancels at any time.

```bash
cd sim
MUJOCO_GL=egl python3 scripts/perception_check.py                     # detection accuracy on every tile
MUJOCO_GL=egl python3 scripts/autonomy_sim.py --tile stairs --level 4 --start top      # edge guard
MUJOCO_GL=egl python3 scripts/autonomy_sim.py --tile stairs --level 3 --start top --press-a --policy <exported>
python3 scripts/run_quest_viewer.py --world skills --tile gaps --level 2 --start top --policy <exported>
```

## Measured (simulator, rendered D435i with noise, robot gliding to 0.6 m from the feature)

| feature (level 0 → 5) | true | detected |
|---|---|---|
| stairs up, rise | 4 / 7 / 10 / 13 / 16 / 19 cm | 4 / 7 / 10 / 13 / 16 / 19 cm |
| stairs down, rise | 4 / 7 / 10 / 13 cm | 4 / 7 / 10 / 12 cm |
| stairs down, 16 / 19 cm (5 steps) | total 80 / 95 cm | "stairs down ~78 / ~93 cm total" (estimated, see below) |
| drop | 4 … 13 cm; 17 / 20 cm | exact; reported as estimated stairs down of 17 / 20 cm |
| gap width | 10 / 17 / 24 / 31 / 38 / 45 cm | 10 / 14 / 26 / 26 / 38 / 46 cm |
| overhang (only reported when lower than the robot) | 46 / 42 cm | 45 / 42 cm |

With the real mount the camera is level and about 0.38 m above the ground, so it cannot see the treads just past
a stair edge going down. When lower ground is seen further on at a stair-like slope, the detector reports "stairs
down, ~X cm total" (estimated); otherwise "unseen ground" (the guard stops, nothing is offered). Bars higher than
the robot can show as a "wall" far away until the floor under them has been seen; this resolves as the robot
approaches. Edge distances are within ~4 cm. With the guard on, the robot stopped before every feature tested
(stairs up and down, gaps, curbs, bars), also at 1.5 m/s, 0.2–0.4 m from the edge.

## Sensor mounts (CAD export of 2026-10-30) and what to check on the robot

- **D435i**: depth origin (left imager) at (−0.0292, 0.1643, −0.0003) m in the base frame, looking straight
  forward, no tilt (±2 mm; estimated from the body centre + datasheet offsets). It sees the ground from ~0.85 m
  ahead of the axle; right after start-up the area closer than that has not been seen yet: start on open ground.
  The front arms / knees fall in the side corners of the image closer than the minimum range and drop out.
- **RPLIDAR A1**: rotation axis at (−0.0116, 0.0942) m, scan plane 0.066 m above the base origin (±4 mm), parallel
  to the chassis; nothing of the robot blocks the beam. Check on the robot whether the SDK's 0° points forward or
  to the motor side (here: 0° = forward, counter-clockwise; the SDK turns clockwise).
- **Robot height**: the top of the lidar (+0.0767 m) is the highest point: leg length + wheel radius + 0.077 m
  (0.364 m crouched). The CAD tyre is 226 mm (radius 0.113, the simulator uses 0.1075): measure it.
- **Pose**: the simulator gives the true pose; on the robot the map needs odometry + IMU yaw (drift accumulates,
  the map is local and forgets).
- **Real depth artefacts** not modelled: black or shiny floors and sunlight give holes — the guard treats unseen
  ground close ahead as a possible edge and stops.
