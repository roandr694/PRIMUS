# brain: the Jetson stack (home autonomy)

What runs on the Jetson Orin Nano: maps, localisation, navigation, the semantic memory of the house and the agent
you talk to. The plan and the reasons are in [docs/home_autonomy_architecture.md](../docs/home_autonomy_architecture.md);
this folder is the code. Everything is plain Python (numpy, scipy, numba, transformers) so the same code runs in
the simulator (`sim/home_sim`) and on the robot. ROS 2 wrappers for the robot will be added later.

```text
sensors (odom frame) ─► runtime.Brain: elevation map, features, skill planner, edge guard (perception/) ─► Command, ElevationPatch ─► Pi
        │
        ├─► mapping.FloorSLAM   2-D lidar SLAM / localisation per floor (one map frame, a grid per floor)
        ├─► nav.Costmap         static (tour) + live lidar + camera terrain / low overhangs + keep-out, inflated
        ├─► nav.Navigator       A* (numba) + path follower, replanning, fallbacks, stall and guard recovery
        ├─► semantic.SemanticMemory   OWL-ViT detections + CLIP embeddings fused into 3-D objects (add / remove)
        ├─► semantic.SceneGraph       rooms from the map, named from their objects; places; goal resolution
        ├─► semantic.PeopleTracker    people to follow
        └─► behaviors + agent         go_to, follow, set_body, carry, cross, find, ...; intents + local LLM
```

| file | what it does |
|---|---|
| `runtime.py` | `Brain`: the robot-centric perception (`perception/`) and the 50 Hz loop that turns a pilot input into a `link.Command`; the navigator and the agent are pilots, the edge guard stays underneath |
| `stack.py` | `HomeStack`: everything together (SLAM, costmaps, navigator, memory, people, behaviours); `on_state / on_depth / on_scan / on_rgbd`, `step(now)` |
| `persist.py` | save / load a house: floor maps, the camera layer of the tour, object memory, scene graph labels and places |
| `mapping/grid.py` | log-odds occupancy grid with ray clearing (keeps wall surfaces) |
| `mapping/matcher.py` | correlative scan matching on a likelihood field (coarse-to-fine, bilinear) and global relocalisation |
| `mapping/slam.py` | multi-floor scan-to-map SLAM: odometry trusted within a dead band, slips detected, floors switched by height |
| `nav/astar.py` | A* on a cost grid (numba), line-of-sight simplification |
| `nav/follower.py` | pure pursuit with turn-in-place and speed limits for a balancing robot |
| `nav/costmap.py` | layered costmap (see the architecture §4.2) |
| `nav/navigator.py` | goals, replanning every 2 s or when blocked, fallback plans, stall and edge-guard recovery |
| `semantic/models.py` | OWL-ViT B/32 detector (NanoOWL's model) and CLIP B/32 embedder, lazy, unloadable |
| `semantic/memory.py` | object memory: fusion, confirmation, DynaMem-style removal, text queries |
| `semantic/rooms.py` | room segmentation (distance transform, seeds, geodesic growth) and naming from objects |
| `semantic/scene_graph.py` | floors → rooms → objects + places; resolves "cozinha", "do lado do sofá na sala" into a goal |
| `semantic/people.py` | person tracks (position, velocity, appearance) for follow-me |
| `behaviors.py` | `GoTo`, `SetBody`, `Park`, `Stop`, `Turn`, `Cross`, `Follow`, `Find`, `Sequence` |
| `agent/` | `robot_card.py` (who PRIMUS is), `intents.py` (Portuguese rules), `tools.py`, `llm.py` (local LLM), `agent.py` |
| `ui/server.py` | the owner's web page: chat, house map, camera, status |

## Running it

Everything is driven from the simulator for now (`sim/home_sim/README.md`):

```bash
cd PRIMUS
MUJOCO_GL=egl python3 sim/home_sim/eval_semantic.py          # tour the house, build and save the semantic map
MUJOCO_GL=egl python3 sim/home_sim/run_home.py --person       # live: open http://localhost:8090 and talk to it
```
