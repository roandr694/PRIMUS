# home_sim: the house as the robot sees it

The CMA house (`sim/house`) with the robot, its controller and the Pi side running in-process, producing what the
Jetson gets on the real robot (odometry with realistic drift, rendered D435i depth and aligned RGB-D, ray-cast
RPLIDAR scans). The Jetson stack (`brain/`) runs unchanged against it. Every phase of the home-autonomy plan
([docs/home_autonomy_architecture.md](../../docs/home_autonomy_architecture.md)) is scored here against the
ground truth of the house generator (`sim/house/semantics.py` → `sim/models/house/house_gt.json`).

| file | what it does |
|---|---|
| `world.py` | `HomeSim`: physics + controller + `robot/pi_runtime.py`; `step(command, patch)` advances 20 ms and returns `state` (+ depth / RGB-D / scan when due); drifting odometry; doors open by default; optional walking person |
| `runner.py` | closed loops (`run` with a bare `Brain`, `run_stack` with the `HomeStack`), collision monitor, video recorder, scripted driver |
| `gt_grid.py` | ground-truth traversability grid of a floor (ray casting + collider rasterisation), cached |
| `tour.py` | the owner's tour, simulated: a pilot who knows the house visits every indoor room (A* on the truth) |
| `person.py` | a person (mocap mannequin) walking waypoints, seen by the camera and the lidar |
| `eval_mapping.py` | P1: tour with SLAM; pose error, map precision and coverage |
| `eval_nav.py` | P2: random point-to-point goals on the tour map (localisation mode) |
| `eval_semantic.py` | P3: semantic tour; object recall / precision, room segmentation and naming; saves the house |
| `eval_agent.py` | P4/P5: Portuguese requests executed in the house, judged on the true final state |
| `run_home.py` | live demo: real-time sim + stack + agent + web page (http://localhost:8090) |

```bash
cd PRIMUS
python3 sim/house/build.py                                        # once: the house + house_gt.json
MUJOCO_GL=egl python3 sim/home_sim/eval_semantic.py               # ~8 min: tour, map, objects, rooms -> robot_map_sem
MUJOCO_GL=egl python3 sim/home_sim/eval_nav.py --goals 20         # navigation on that map
MUJOCO_GL=egl python3 sim/home_sim/eval_agent.py [--llm Qwen/Qwen3-4B-Instruct-2507]
MUJOCO_GL=egl python3 sim/home_sim/run_home.py --person [--llm Qwen/Qwen3-4B-Instruct-2507]
```

Use the simulation Python environment (`~/test/isaac-rl/bin/python`: MuJoCo 3.11, torch, transformers, numba).
The first run of the semantic tour downloads OWL-ViT B/32 and CLIP B/32 (~1.2 GB); `--llm` downloads the model.

Known simulation gaps: lawn and garden foliage is visual only (the robot sees it as obstacles, the ground truth
calls it free), so tours and goals stay indoors; glass is seen by the lidar here, unlike on the real robot.
