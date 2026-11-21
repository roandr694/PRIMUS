# Home autonomy: architecture and plan

*Drafted 2026-11-20. Status: P0-P4 implemented in simulation (MuJoCo, CMA house), results in
[§12](#12-implementation-status-simulation); not yet on the robot. The owner's decisions are in
[§10](#10-decisions); D5 is still open. Code: `brain/`, `robot/link.py`, `robot/pi_runtime.py`, `sim/home_sim/`.*

## 1. Goal

PRIMUS already handles most of what a house puts in its way: small obstacles, stairs up and down, ramps, curbs,
low furniture (v13c teacher, student-v1 on the camera). What is missing is **autonomy**: today a pilot drives all
the time and the robot only protects itself (edge guard) and offers skills (A to cross).

Target behaviour, in the owner's words turned into requirements:

| request | needs |
|---|---|
| "vai para a cozinha", "vai para a sala de TV" | a map of the house split into named rooms; localisation; route planning |
| "vai para a sala e fica do lado do sofá" | objects in the map (open vocabulary), a goal pose next to an object |
| path blocked → find another way; object in the way → go around | replanning on a live costmap; local avoidance |
| "sobe aquela escada" | the route can include skills; the existing skill planner runs them without the A button |
| talk to it, it answers, knows who it is and what it can do | speech in/out, an LLM agent with a self-description and a tool API |
| "fica mais baixo / mais alto" | body commands as tools |
| "vou botar esse copo em você, leva para tal lugar" | carry mode: payload detection, gentle driving, no jumps |
| "vem me seguindo" | person detection and tracking, follow behaviour |

Main reference: **OK-Robot** (Liu et al., NYU + Meta, RSS 2024): scan the home once, build a voxel map with
open-vocabulary features (OWL-ViT detections + CLIP embeddings), answer "where is X" with a text query, plan with
A* on a 2-D obstacle map, then act. 58.5 % pick-and-drop success in 10 homes, 82 % in tidy ones; its main
weakness is a static map. We take its *pipeline* (scan → semantic memory → query → navigate) and fix the weakness
with ideas from three related works:

- **HOV-SG** (Werby et al., Freiburg, RSS 2024): hierarchical open-vocabulary scene graph *floors → rooms →
  objects* with a cross-floor navigation graph. We need exactly this hierarchy (the CMA house has 3 levels plus
  the pier and parking). Its construction is slow, so it is an offline step, as here.
- **ConceptGraphs** (Gu et al., 2023/ICRA 2024): object-centric graph built by fusing 2-D foundation-model
  detections across views; the graph is easy to serialise as text for an LLM planner.
- **DynaMem** (Liu et al., NYU, 2024): online memory that adds *and removes* objects as the robot sees the scene
  again; 70 % vs 30 % for a static memory on non-stationary objects.

We do not grasp (no arm), so OK-Robot's manipulation half is replaced by navigation, body commands and carrying.

## 2. Hardware split

From `README.md`: **Jetson Orin Nano** (8 GB shared CPU/GPU memory) and **Raspberry Pi 5**, linked by Ethernet.
The split follows one rule: **everything that keeps the robot standing runs on the Pi and must survive the Jetson
crashing; everything that thinks runs on the Jetson.**

| | Raspberry Pi 5 ("spinal cord") | Jetson Orin Nano ("brain") |
|---|---|---|
| rate | 1 kHz control, 50 Hz policy | 5-30 Hz perception, 50 Hz planner, event-driven agent |
| devices | moteus drivers (CAN-FD), WitMotion IMU, landing-gear servos, battery monitor, **gamepad** (Bluetooth) | **D435i** (USB 3, needs the Jetson for depth + RGB bandwidth and the GPU), **RPLIDAR A1** (USB), microphone + speaker (see §10) |
| software | LQR + VMC (`control/`), state estimator and wheel-inertial odometry (`robot/`), RL policy student-v1 (ONNX on CPU, ~1 ms), safety supervisor | elevation map, features, skill planner and edge guard (`perception/`), SLAM, costmaps, navigator, semantic memory, detectors, agent, speech |
| no ROS | plain Python/C loop, fixed binary messages | ROS 2 Humble for mapping and navigation (D1) |

**Everything runs on board (D2, D4).** The real robot builds its semantic map, plans and talks with no PC and no
internet: no cloud LLM, no desktop processing. The desktop GPU is used only to run the simulation (and to run the
*same* Jetson-sized models faster there), so a result in sim says something about the robot.

Why the camera goes on the Jetson: the D435i needs USB 3 bandwidth, the semantic pipeline needs RGB on the GPU,
and the elevation map already runs there. The policy on the Pi still needs the map: see the elevation patch in §3.

### Safety supervisor (Pi)

- Pilot over autonomy, always: any stick input above a dead band, or B, takes over immediately.
- Jetson heartbeat: commands older than 150 ms → ramp the speed to zero and hold; older than 2 s → park (landing
  gear). The robot never keeps driving on a stale command.
- Elevation patch older than 300 ms → the policy sees the map as unknown (`known = 0`, as in training when a cell
  is unseen) and the speed is capped at 0.3 m/s.
- Fall detector (pitch/roll beyond limits, wheel-ground contact lost while not in a jump) → torque off / recovery
  policy on the stand-up button, as today.
- Voice "pára"/"stop" is handled by a keyword spotter on the Jetson that sends `brake` directly, below the LLM.

## 3. Jetson ↔ Pi interface

UDP on the Ethernet link, fixed little-endian structs with a sequence number and a timestamp; clocks synced with
chrony (Jetson = server). One small Python module (`robot/link.py`) shared by both ends and by the simulator.

| message | direction | rate | content |
|---|---|---|---|
| `State` | Pi → Jetson | 100 Hz | odom pose (x, y, z, yaw), body pitch/roll, speed, yaw rate, leg length, gear state, contact, battery, payload estimate, policy regime, faults, gamepad axes and buttons |
| `Command` | Jetson → Pi | 50 Hz | the `RobotCommand` fields (speed, yaw rate, leg length, pitch_ref, brake, park, max_accel), skill flag, source (pilot / nav / agent), heartbeat |
| `ElevationPatch` | Jetson → Pi | 15 Hz | 2.4 m × 2.4 m around the robot, 4 cm cells, ground height + known mask + ceiling, in the **odom** frame with its origin (~35 kB, 0.5 MB/s) |
| `Correction` | Jetson → Pi | 1-10 Hz | map → odom transform from localisation (for logging; the Pi does not need it to balance) |

The Pi samples the patch with its *own* 1 kHz odometry pose for the policy's height and ceiling scans (exactly what
`SkillPolicy.act` does with `ground_fn` / `ceiling_fn` today), so map latency does not turn into scan misalignment
while the robot moves.

## 4. Software layers

```text
                    ┌──────────────────────── Jetson Orin Nano ─────────────────────────────────┐
 voice ─► wake word ─► ASR ─► AGENT (LLM + tools) ─► BEHAVIOURS (go_to, follow, carry, ...) ─┐   │
 phone/web chat ──────────────┘   ▲  robot card, scene graph as text                       │   │
                                  │                                                        ▼   │
                        SEMANTIC MEMORY (floors → rooms → objects, places)        NAVIGATOR    │
                          ▲ online detector + CLIP (few Hz)                      global A* on  │
                          │                                                      nav graph +   │
 D435i RGB-D ─► VIO ──────┼─► ELEVATION MAP (existing) ─► features (existing) ─► costmap; local│
              (pose)      │                                  │                  planner       │
 RPLIDAR ─► 2-D SLAM / localisation per floor ───────────────┴──► COSTMAP ◄─────┘            │
                                                                       │                        │
                                     SKILL PLANNER + EDGE GUARD (existing, gets nav commands)   │
                                                                       │ Command 50 Hz          │
                    └──────────────────────────────────────────────────┼────────────────────────┘
                                                                       ▼ Ethernet
                    Raspberry Pi 5: safety supervisor ─► student-v1 policy ─► LQR + VMC 1 kHz ─► motors
```

### 4.1 Localisation and geometric maps

- **Frames:** `odom` (Pi wheel-inertial odometry, continuous, drifts), `map_<floor>` (one per floor, from SLAM),
  `base`. The elevation map stays in `odom`, as it is today.
- **2-D lidar SLAM per floor** (`slam_toolbox`, mapping during the tour, localisation afterwards) is the backbone:
  cheap on CPU and robust indoors. The scan must be projected with the body pitch/roll and height: PRIMUS pitches
  the body ±20° to look with the camera and changes height by ~0.35 m, so the scan plane tilts and hits the floor.
  Scans with |pitch| > 8° are dropped for SLAM (they still feed obstacle avoidance after `drop_ground`, as today).
- **Visual-inertial odometry** from the D435i (cuVSLAM in Isaac ROS, or RTAB-Map's odometry as the fallback) fused
  with wheel odometry in an EKF (`robot_localization`). It helps where the lidar sees little (open decks, glass).
- **Floors:** the active floor comes from the odom z and from the transition the robot just executed (stairs edge,
  lift); each floor has its own map and the nav graph links them.
- **Glass:** the CMA house is full of glazed pivot doors and glass rails; lidar and stereo both see through or
  reflect off glass. Mitigation: mark glass in the map during the tour (VLM or the owner: "isto é vidro"), keep
  the edge guard's depth check, and in simulation make the glass visible to the lidar as it is on the real robot
  (to verify per material).
- **3-D:** the existing elevation map (ground + ceiling) gives traversability; nvblox (TSDF/ESDF on the GPU) is an
  option if we need a full 3-D occupancy for the semantic fusion — on the Orin Nano it runs one camera in
  static-scene mode only (Isaac ROS docs), so it is not on the critical path.

### 4.2 Costmap and traversability

One 2-D costmap per floor at 5 cm, layers:

| layer | source | meaning |
|---|---|---|
| static | SLAM map from the tour | walls, fixed furniture |
| obstacles | lidar (after `drop_ground`) + depth obstacles | live, with decay, so a moved chair clears |
| terrain | elevation map + `features.py` | free / **skill-crossable** (step, curb, stairs, gap ≤ 45 cm: high cost + the skill it needs) / forbidden (drop, gap > 45 cm, stairs > 19 cm, overhang lower than the robot can crouch) |
| keep-out | owner ("não entra no banheiro"), glass, pool edge | forbidden |
| people | person tracker | inflated, moving |

The terrain layer is the PRIMUS-specific part: a wheel-legged robot can cross things a wheeled robot cannot, and
the limits come straight from `features.Limits` and `policy_interface.SKILLS`, so the planner never routes through
something the policy cannot do.

### 4.3 Navigation

- **Nav graph** (built offline from the tour, updated online): nodes = rooms, doorways, stair ends, lift stations,
  named places ("perto do sofá"); edges = traversals with a cost, a type (`drive`, `stairs_up`, `stairs_down`,
  `curb`, `gap`, `lift`, `door_closed_ask`) and the skill they need.
- **Global planner:** Dijkstra on the nav graph for the room sequence, A* (Nav2 Smac 2-D) on the floor costmap for
  each leg.
- **Local planner:** Nav2 MPPI or a small custom DWA tuned to PRIMUS dynamics (speed ramp `max_accel`, the speed
  lag of the LQR, 0.7 s planning latency already used by the edge guard). Output: speed + yaw rate, the same two
  numbers the pilot gives today.
- **Skills on the route:** when the next edge is a skill, the navigator drives to the edge's approach pose and
  calls the existing `SkillPlanner` (ALIGN → EXECUTE) programmatically instead of the A button, then checks with
  the detector and odometry that it is on the other side. The **edge guard stays between the navigator and the
  controller**: the navigator is just another pilot, so all the existing protection still applies.
- **Blocked route:** if the local planner makes no progress for 5 s, or a corridor is closed in the costmap, the
  edge is marked blocked (cost ∞, decays after a few minutes) and the route is replanned. No alternative → the
  agent tells the owner ("a porta do corredor está fechada, pode abrir?") and waits.
- **Doors:** PRIMUS has no arm. Closed doors are edges of type `door_closed_ask`. Pushing a door open is a later
  option (light interior doors only, needs tests in the sim, the doors there already have damping/friction).

### 4.4 Semantic memory (the OK-Robot part)

Built in two stages, both testable in the CMA house sim:

**Stage A, the tour (once per house, processed on the Jetson).** The owner drives the robot through the house (or
the robot explores frontiers on its own, later). It records RGB-D keyframes (every ~0.3 m or 15°), poses, lidar and
short notes ("esta é a cozinha") to a log on the Jetson's NVMe. The log is processed **on the Jetson** (D4), while
the robot is parked: the agent's LLM is unloaded during this "mapping mode" so the vision models get the memory,
and the job may take a while (target: under 1 h for the CMA house social + bedroom levels; it can resume after an
interruption). In simulation the same pipeline, with the same models, runs on the desktop GPU.

Model choice follows from the 8 GB budget, not from what is best on a workstation:

1. Open-vocabulary detection per keyframe with a TensorRT model sized for the Orin Nano (YOLO-World-S/M or
   NanoOWL/OWLv2-B), masks from the box + depth (cluster the depth inside the box) or a light segmenter
   (EfficientSAM / MobileSAM class) only where the box is ambiguous, and one embedding per object crop from a
   small CLIP/SigLIP (ViT-B/16 class) in fp16.
2. Multi-view fusion into 3-D object instances (ConceptGraphs-style: merge by 3-D overlap + embedding similarity),
   each with centroid, extent, floor, embedding, best crops, a short caption from a VLM, first/last seen.
3. **Rooms:** segment each floor's 2-D map into regions (door detection on the occupancy map + Voronoi /
   morphological segmentation, as HOV-SG does with its floor/room layers), then name each region from (a) the
   owner's voice notes when present, (b) a VLM over the region's keyframes ("which room is this?"), (c) object
   co-occurrence (bed → bedroom, stove → kitchen). The VLM is the agent's local model (§4.6), loaded after the
   detector is unloaded. The owner reviews and renames on a web page served by the robot, or by voice.
4. Output, versioned per house: `house_map/` with the floor maps, `scene_graph.json` (floors → rooms → objects,
   places, glass, keep-outs), `nav_graph.json`, object crops and embeddings.

**Stage B, live (on the Jetson).** A light open-vocabulary detector (YOLO-World or NanoOWL in TensorRT) and a small
CLIP run at 1-3 Hz on the camera. DynaMem-style updates: objects seen where the memory has them refresh
`last_seen`; memory objects whose position is in view and clearly visible but not detected for several frames are
removed; new detections are added with low confidence until seen again. The owner can teach places at any time
("lembra deste lugar como 'carregador'").

**Queries.** Text → embedding → top candidates among objects (filtered by room if given), re-ranked by the LLM
or a VLM on the stored crops when the score is ambiguous. Then a goal pose: the nearest free, reachable costmap
cell at 0.5-0.8 m from the object's footprint, facing it. If the object is not in memory: explore the rooms
where it is likely (LLM prior), looking for it with the live detector.

**Known limit:** the camera sits ~0.38 m above the floor, level. Table tops and counters (~0.75-0.9 m) are seen
only from far or by raising the leg (to 0.54 m) and pitching up (+10°, already supported). Small objects on tables
are therefore weakly mapped; furniture, doors, appliances and floor-level objects map well.

### 4.5 Behaviours

A behaviour executor (state machines, cancellable, each reports progress and failures as text for the agent). The
agent starts behaviours; it never sends speeds itself.

| behaviour | what it does |
|---|---|
| `go_to(target)` | target = room, object (+ relation), named place, or "here" (the speaker); plans, executes, crosses skills, replans; arrives facing the target |
| `follow(person)` | person detector (TensorRT) + depth for 3-D position + appearance re-ID + track; keeps 1.0-1.5 m, slows near the person, goes to the last seen position and calls out when the track is lost |
| `come_here` | finds the speaker: last known position from the phone/app, else turns towards the voice (only with a mic array), else looks around for the person |
| `set_body(height, pitch)` | leg 0.18-0.52 m (low / medium / high), look up / down |
| `carry(on/off)` | detects the payload from the hip torques (state estimator mass change) and confirms by voice; while on: `max_accel` 0.5 m/s², yaw rate and speed capped, body level, **no jump skills** (stairs, curbs, gaps forbidden in the costmap), park gently |
| `cross(kind)` | "sobe aquela escada": the detector's nearest feature of that kind, then the skill |
| `look_and_describe` | VLM on the current frame ("o que você está vendo?") |
| `find(object)` | memory query, else search, then `go_to` |
| `remember_place(name)` / `label_room(name)` | write to the semantic memory |
| `explore` | frontier exploration to extend the map (later) |
| `patrol(route)` / `check(thing)` | go, look, report with a photo to the phone ("a porta da varanda está aberta?") |
| `park` / `stand` | landing gear |
| `wait(s)`, `stop` | |

### 4.6 Agent and speech

- **Speech in:** wake word (openWakeWord, "Primus") → VAD → ASR (faster-whisper small, int8, Portuguese) on the
  Jetson GPU. The "pára/stop" keyword spotter runs always and bypasses everything.
- **Speech out:** Piper TTS with a pt-BR voice (CPU, real time on the Jetson). Short answers; the robot can also
  nod (pitch) or crouch as body language.
- **LLM agent with tool calling.** The system prompt is generated at runtime from:
  - a **robot card** (`agent/robot_card.md`): what PRIMUS is (two-wheeled legged robot, 0.18-0.52 m legs, top
    speed, battery), what it can cross (stairs ≤ 19 cm, curbs ≤ 20 cm, gaps ≤ 45 cm, ramps, crouch under bars),
    what it cannot do (open doors, pick things up, glass, carry while hopping), safety rules;
  - the **scene graph as text** (floors, rooms, notable objects, named places);
  - the **current state** (room, battery, height, carrying, running behaviour).
  Tools = the behaviours of §4.5 plus `where_am_i`, `status`, `list_rooms`, `query_memory`, `say`. The agent
  plans multi-step requests ("leva este copo para a cozinha e volta") as a sequence of tool calls, confirms risky
  ones (stairs while carrying → refuses and explains), and reports results.
- **The LLM runs only on the Jetson (D2).** The Orin Nano has 8 GB shared by everything. A 2-4 B vision-language
  model in 4-bit (Qwen3-VL-2B/4B class via llama.cpp, ~2-3 GB) fits next to ASR, detectors and SLAM, at tens of
  tokens/s for a 2 B model per community benchmarks (to measure on our board; candidates are compared in P4 on
  the same scripted requests). Such a model is good at tool calling and short Portuguese answers and weak at long
  reasoning, so the design keeps its job small:
  - **constrained output:** tool calls are generated under a JSON grammar (llama.cpp GBNF) from the tool schema, so
    they always parse;
  - **a deterministic intent layer first:** common commands ("pára", "vai para a <cômodo>", "fica mais
    baixo/alto", "me segue", "volta") are matched by rules against the room and place names and never wait for
    the LLM; the LLM handles the rest;
  - **short context:** the scene graph goes in as a compact summary (rooms + notable objects); details come from
    `query_memory` calls instead of the prompt;
  - **multi-step plans are checked by code** (each step validated against the map and the robot card) before
    execution.
- **Phone/web first (D3):** the agent is reached from a chat + push-to-talk page served by the Jetson (phone,
  Quest viewer); audio is sent to the robot and ASR/TTS still run on the Jetson. This is also how it is tested in
  the simulator. An onboard mic array and speaker come later.

## 5. What we reuse

| existing | role in the new stack |
|---|---|
| `perception/elevation_map.py`, `features.py`, `lidar.py` | terrain layer of the costmap; source of skill edges; elevation patch for the Pi |
| `perception/planner.py` (`SkillPlanner`, edge guard) | executes skills on the route; last safety layer under the navigator |
| `perception/autonomy.py` | grows a `nav` input next to the pilot; the A button becomes one caller among others |
| `control/` (LQR, VMC, `RobotCommand`), student-v1 | unchanged, moved to the Pi |
| `sim/house` (CMA house, doors, lift, 3 levels) | the test world for everything above; needs ground-truth rooms and object labels (§6, P0) |
| `sim/scripts/skill_world.py` `SimSensors` | rendered D435i depth and lidar; add RGB output |
| Quest / web viewer | chat panel, map view, room review page |

## 6. Simulation first

The house sim is the test bench for every phase before the real robot drives itself.

- **Sim bridge:** MuJoCo house + robot + `SimSensors` publish the same topics/messages as the robot (RGB-D, lidar,
  `State`) and accept `Command`; it runs the Pi side (controller + policy) in-process. The Jetson stack runs
  unchanged on the desktop against it.
- **Ground truth:** the house generator tags every room polygon (name, floor) and every furniture item (class
  label, room) when it builds the scene → `house_gt.json`. This scores localisation (ATE), room segmentation (IoU,
  naming accuracy), object recall/precision, and navigation success.
- **Rendering realism:** open-vocabulary detectors on MuJoCo renders may underperform real images. If they do,
  run the semantic tour through the Isaac Sim USD of the same house (already generated, never opened in Isaac).
- **Fall detector + scenario runner:** `autonomy_sim.py` has none today; add it, then batch scenarios
  ("living room → kitchen", "suite 1 → sea deck via stairs", "blocked corridor", "follow a scripted person",
  "carry a 300 g mass to the varanda") with success / collision / fall / time metrics.

## 7. Phases

Each phase ends with a measurable test in the house sim. Real-robot items (§2 hardware) progress in parallel and
are listed in `status_and_plan.md`.

| phase | content | done when |
|---|---|---|
| **P0 foundations** | `robot/link.py` messages; sim bridge; RGB in `SimSensors`; `house_gt.json` from the generator; fall detector + scenario runner; ROS 2 workspace skeleton (D1) | a scripted drive in the house runs through the bridge with the Pi side in-process and is scored |
| **P1 localisation + 2-D maps** | pitch-aware scan projection; slam_toolbox tour mapping per floor; EKF with wheel odometry (+ VIO); floor switching | tour of the social level and bedrooms; ATE < 5 cm; relocalises after a kidnap |
| **P2 navigation** | costmap layers incl. terrain; nav graph; global + local planner; navigator as a pilot into `Autonomy`; skills on routes; blocked-route replanning | 20 random point-to-point goals per floor ≥ 90 % success, 0 falls; a route across floors via the living-room stair; a closed door triggers a replan |
| **P3 semantic memory** | tour log; Jetson-sized pipeline (detection, fusion, rooms, naming) with its memory budget enforced in sim; review page; scene graph + queries; live detector + DynaMem updates | rooms named correctly ≥ 90 % on the CMA house; "sofá", "geladeira", "cama", "mesa de jantar" etc. found; a moved chair updates the memory; peak memory of the pipeline < 6 GB |
| **P4 agent (text)** | robot card; intent layer; tool API over the behaviours; local LLM with a JSON grammar (compare 2-3 candidate models); chat page | 30 scripted Portuguese requests ("vai para a cozinha", "fica do lado do sofá na sala", "fica mais baixo", "sobe a escada") executed correctly in sim with the chosen local model |
| **P5 behaviours** | follow (sim person: a scripted walking body), carry mode with payload estimation, find, check/patrol | follow a person through 3 rooms; carry 300 g without a fall or a hop; refuses stairs while carrying |
| **P6 voice** | push-to-talk from the phone page, then wake word; ASR, TTS, stop keyword, all on the Jetson | voice round trip < 2 s for simple commands on the Jetson |
| **P7 on the robot** | Jetson deployment, memory/latency profile, Pi loop with the safety supervisor, tour of the real house | the same P2/P4 tests on the real robot at reduced speed |

P0-P2 are the critical path; P3 and P4 can start in parallel once P1 exists (P4 can use the ground-truth scene
graph until P3 delivers the real one).

## 8. Jetson memory budget (Orin Nano 8 GB, estimates to measure in P7)

| item | GB |
|---|---|
| OS, ROS 2, drivers, maps, Python | 1.5 |
| SLAM + EKF + Nav2 | 0.5 |
| elevation map, features, planner | 0.2 |
| live detector + CLIP (TensorRT, fp16) | 1.0 |
| person detector + re-ID | 0.4 |
| faster-whisper small int8 + wake word + Piper | 0.8 |
| LLM 2-4 B, 4-bit, short context | 2.0-3.0 |
| margin | ~1 |

This is "service mode". In "mapping mode" (tour processing, robot parked) the LLM, ASR and person detector are
unloaded and the tour models (detector, light segmenter, CLIP) get ~4-5 GB.

Tight but plausible. Fallbacks in order: run the live detector and person detector as one model; load the VLM
vision tower only on demand; a 2 B instead of a 4 B LLM; upgrade to a Jetson Orin NX 16 GB (same module format).
nvblox is left out of this budget on purpose. In simulation the desktop enforces the same per-process limits so a
pipeline that does not fit is caught before the robot.

## 9. Risks

| risk | mitigation |
|---|---|
| lidar plane tilts with body pitch and height | project with the full 3-D pose; drop tilted scans for SLAM |
| glass doors and rails invisible to lidar/stereo | glass layer in the map, edge guard, owner keep-outs |
| camera too low for table tops | raise leg + pitch up when searching; accept weak small-object mapping |
| Orin Nano memory | §8 fallbacks |
| sim renders too clean for detectors | Isaac Sim USD tour; real tour as soon as the robot drives |
| payload changes balance (LQR tuned for nominal mass) | payload estimate from hip torques; carry-mode limits; test masses in sim; later a mass-scheduled gain |
| a top tray would block the lidar (it is the highest point of the robot) | mechanical decision before carry mode on the real robot (§10) |
| LLM hallucinates actions | tools validate everything against the map and robot card; physical safety never depends on the LLM |
| a 2-4 B local model misunderstands longer requests | deterministic intent layer for common commands; JSON grammar; asks back when unsure; compare models in P4 |
| tour processing on the Jetson too slow | keyframe subsampling, resumable job, run while parked/charging |
| cuVSLAM tracking loss with a D435i (reported on forums) | lidar SLAM is the backbone; VIO is optional |

## 10. Decisions

| id | question | decision (owner, 2026-11-20) |
|---|---|---|
| D1 | ROS 2 on the Jetson, or a pure-Python stack like today | **ROS 2 Humble** on the Jetson and the desktop sim; no ROS on the Pi |
| D2 | where the LLM runs | **only on the Jetson**, no cloud |
| D3 | microphone/speaker | **phone/web page first**, onboard mic array + speaker later |
| D4 | where the semantic tour is processed | **in sim on the desktop GPU; the real robot does everything on the Jetson + Pi** (so the sim uses the same models) |
| D5 | where a payload goes (top tray vs lidar) | open: decide with the CAD before P5 on the real robot |

## 11. Camera-first perception and realistic houses in Isaac Sim (plan of 2026-11-20)

The owner's review of the first implementation (§12): the lidar sits at the top of the robot and sees one plane,
so anything lower that the robot can hit is invisible to it; the D435i sees in 3-D and must also build the semantic
map. The camera therefore becomes the primary map sensor and the lidar assists it (long walls, 360°, fast and stable
localisation). Pose comes from a filter, not from odometry corrected by one matcher. And the generated CMA house is
too clean for detectors trained on photos (on the MuJoCo tour 42 of 57 objects lie on a real object, only ~25 %
with the right class), so the tests move to ready-made realistic houses rendered by Isaac Sim's RTX camera.

Not everything is the house's fault: the P2 navigation failure (0.9 m dining gap, pivot-door channels) is geometry
and control and would happen in a real house too, so it is fixed in parallel (phase E).

| phase | content | done when |
|---|---|---|
| **A realistic houses** | compare 2-3 ready-made USD homes (GRScenes / InternUtopia, Infinigen Indoors, HSSD converted) on licence, collision quality, 12 GB GPU fit; ground truth (rooms, objects with class and box) generated from the scene's own labels | 2+ houses load in Isaac Sim 6.1 with collision and a `house_gt.json` each |
| **B robot + sensors in Isaac** | robot in PhysX with the LQR in the loop (student-v1 optional: a sim-to-sim test of its own), RTX D435i (colour, depth with noise, IR stereo pair), IMU, RTX 2-D lidar; a bridge that feeds `brain/` the same messages as `HomeSim`. Fallback if the wheel-legged robot is not stable in PhysX: MuJoCo physics with the house's collision meshes converted, Isaac renders only | a scripted drive through a realistic house with the full brain stack, no fall |
| **C camera 3-D map** | voxel occupancy map from depth (log-odds, 5 cm, GPU/NumPy; the same role nvblox plays on the Jetson); the 2-D costmap = everything between the floor and the robot's current height (it changes with the leg), plus lidar walls and 360°; semantic objects attached to the 3-D map and removed when the map shows them gone | table tops, chair legs and steps appear in the costmap with no lidar contribution; no collision with low furniture |
| **D fused localisation** | EKF over wheel odometry, gyro (IMU), lidar scan-to-map matches and camera depth-to-voxel-map matches, each with its covariance; slip detected as disagreement between sources; visual-inertial odometry (PyCuVSLAM on the D435i IR pair + IMU) as an extra input if it runs; loop closure for the mapping tour | mapping tour mean pose error < 10 cm, max < 30 cm; localisation 2-5 cm as today |
| **E tight spaces** | planner prefers wide corridors (clearance cost), door-channel alignment manoeuvre, guard corridor while turning in place | ≥ 8/10 random goals out of a furnished room, in MuJoCo and in Isaac |
| **F evaluations in Isaac** | mapping, navigation, semantic memory and agent evaluations in the realistic houses, compared with MuJoCo | numbers in §12 per house |

Then the LLM in 4-bit next to the vision models, and voice (P6). Risks: the 12 GB GPU may not hold Isaac rendering,
the vision models and the LLM together (measure; run the LLM apart for navigation tests if needed); Isaac is
slower than MuJoCo (longer tests); the robot's stability in PhysX is the least certain part (fallback above).

## 12. Implementation status (simulation)

*2026-11-21, branch `home-autonomy`, MuJoCo CMA house, student-v1 policy on the Pi side, social level.*

| phase | state | measured |
|---|---|---|
| P0 foundations | done | link messages (UDP structs), Pi runtime (supervisor + policy on the elevation patch), HomeSim (drifting odometry, D435i depth + aligned RGB-D, lidar, fall detector, walking person), ground truth (24 rooms, 175 objects), scenario runner |
| P1 localisation + maps | works, drifts in long tours | matcher alone ~1 cm; full indoor tour: mean pose error 26 cm (max 1.6 m after a wheel slip), map precision 79 %, coverage 83 %; localisation on the saved map 2-7 cm |
| P2 navigation | **partial** | goals inside a room reach 0.1-0.2 m; leaving the furnished living room (dining set, sofas, pivot doors) fails: 2/10 random goals. The plans are right, execution stalls in ~0.9 m gaps and at the pivot-door channels (guard stops, route switching). Main next work |
| P3 semantic memory | geometry ok, classes weak on renders | 42/57 confirmed objects lie on a real object, ~25 % with the right class (OWL-ViT B/32 on MuJoCo renders); rooms: 5/9 named from objects; with the owner's tour voice notes the main rooms resolve by name. CLIP scene classification of keyframes is uninformative on these renders |
| P4 agent | rules done, LLM tested offline | Portuguese intents cover the simple commands; 8/16 scripted requests succeed in the house (failures: the P2 limit, objects never detected); Qwen3-4B answers valid JSON in Portuguese in 0.5-0.9 s with few-shot examples; Qwen3-1.7B 0.3-0.7 s but answers partly in English; the in-house LLM run ran out of GPU memory next to the vision models (needs the 4-bit build, as on the Jetson) |
| P5 behaviours | implemented, not scored | go_to, set_body, park/stand, stop, turn, cross (skills), follow (person tracker), find, carry mode (gentle limits, no skills), sequences |
| P6 voice | not started | text chat page (`brain/ui/server.py`, `sim/home_sim/run_home.py`) is the entry point speech will use |
| Isaac camera (§11) | see §12.1 | |

### 12.1 Camera-first perception, EKF and realistic apartments (phases A-F of §11, 2026-11-21)

| phase | state | measured |
|---|---|---|
| A realistic houses | done | 3 InteriorAgent apartments (72, 127, 169 m²; 8-14 rooms; 29-44 labelled objects) converted by `sim/apartments/convert.py` (voxel-box colliders, doors removed, ground truth from the scene's labels and room polygons); every evaluation takes `--world` |
| B robot + sensors in Isaac | **co-simulation** (the fallback of §11) | MuJoCo keeps physics, controller, student-v1, lidar and the brain; Isaac renders the D435i (both streams) with RTX from the robot's camera pose (`sim/isaac_home/`): 31 ms per depth frame, 39 ms per colour frame, depth within 3 cm of MuJoCo's. The robot itself in PhysX was not attempted: the controller and the policy are validated in MuJoCo, and the camera, which was the fidelity problem, is now rendered by Isaac |
| C camera 3-D map | works | `brain/mapping/voxels.py`: log-odds voxels (5 cm x 5 cm x 10 cm, int8) from every depth frame, carved by free rays; the costmap takes every column with something between 6 cm and the robot's top. On the apartment tour it shows the dining chairs, coffee table, bed and desk the lidar misses. The voxel map is saved with the house. Room segmentation now uses lidar + camera free space |
| D fused localisation | works (lidar measurements) | EKF over (x, y, heading, gyro bias), lidar matches with the covariance of their score surface, gating + slip recovery. CMA social tour, same run: mean pose error **6.1 cm, max 23 cm** (previous corrector 29.7 cm / 2.47 m), map precision 93 % (67 %); apartment tour 1.5 cm mean. Camera depth-to-voxel matches are implemented but off by default: with a voxel map built live they bias the pose (self-matching), and on the apartment they gave no gain and more collisions. Loop closure not done |
| E tight spaces | partial | narrow-passage mode (line up before a passage, 0.18 m/s, short lookahead, guard corridor 0.31 m, back off and retry three times); owner's voice-note positions seed rooms and a watershed puts room borders at doorways (apartment rooms named 5-6/8, was 2/8). Fixed an old navigator bug: after a failed plan it reported "arrived" (earlier navigation numbers may contain false arrivals). Apartment kujiale_0026: **5/5 reachable goals** with and without the camera layer (0.11-0.17 m), the other 5 random goals fall on a balcony the tour mapped badly. CMA house: still 2/10 without the camera layer; with a camera layer built live (the old CMA map has none) 0/10: the robot stalls among the living-room furniture and the stall marks close the way. **Still the main open problem** |
| F evaluations | done for kujiale_0026 | semantic tour with the Isaac camera: object recall **48 %** (0 % with MuJoCo renders of the same apartment), precision 24 % (30 % on the classes the ground truth labels; many false "wardrobe" on white walls, per-class thresholds added after this run, not yet re-measured); agent with the Isaac camera: **11/16** Portuguese requests (failures: kitchen and bathroom not yet labelled because the tour pilot did not reach them, dining table never detected, office route blocked) |

Next, in order: the CMA living-room stall (decaying stall marks, wider-corridor preference, recover by backing out
of the furniture cluster), re-run the semantic tour with the per-class thresholds and the full tour of the larger
apartments (kujiale_0003, kujiale_0004), loop closure, then the LLM in 4-bit next to the vision models and voice.
The person (follow-me) and the RTX lidar are not mirrored in Isaac yet.

## References

- OK-Robot: Liu et al., *OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics*, RSS 2024 —
  [arXiv 2401.12202](https://arxiv.org/abs/2401.12202), [ok-robot.github.io](https://ok-robot.github.io)
- HOV-SG: Werby et al., *Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation*,
  RSS 2024 — [arXiv 2403.17846](https://arxiv.org/abs/2403.17846), [code](https://github.com/hovsg/HOV-SG)
- ConceptGraphs: Gu et al. — [arXiv 2309.16650](https://arxiv.org/abs/2309.16650)
- DynaMem: Liu et al., *Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation* —
  [arXiv 2411.04999](https://arxiv.org/abs/2411.04999)
- Isaac ROS nvblox (Orin Nano support matrix) —
  [docs](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html)
- Orin Nano Super LLM benchmarks (community, llama.cpp) —
  [hokwangchoi/jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
