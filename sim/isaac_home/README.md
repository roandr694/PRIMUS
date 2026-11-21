# Isaac Sim camera for the home sim

MuJoCo keeps the physics, the controller and policy (Pi side), the lidar and the whole Jetson stack; Isaac Sim
renders the D435i with RTX in the same world, from the robot's camera pose (docs/home_autonomy_architecture.md
§11). Both streams come from Isaac: the decimated depth for the elevation map and the camera voxel map
(212 x 120, 15 Hz) and the colour + aligned depth for the semantic memory (640 x 360).

| file | what it does |
|---|---|
| `isaac_server.py` | the render process: opens the stage, hides the door leaves the MuJoCo world removed, one USD camera + Replicator render product per stream (`rgb`, `distance_to_image_plane`), answers render requests over `multiprocessing.connection` |
| `camera_bridge.py` | `IsaacCamera(world)`: starts the server, `depth(sim, cam)` / `rgbd(sim, cam)` for `HomeSim(camera=...)`; D435i intrinsics with square pixels at the model's fx, cropped to its height |

Why a separate process: `SimulationApp` owns its process and GPU context, MuJoCo's EGL renderer and the vision
models stay in the client. Isaac only renders (no physics, timeline stopped).

Measured on the RTX 5070 (`RaytracedLighting`, apartment kujiale_0026): start 12 s, first colour frame ~34 s
(shaders and textures), then 31 ms per depth frame and 39 ms per colour frame, depth medians within 3 cm of
MuJoCo's on the same view (the MuJoCo collision is the voxelised copy of the same meshes).

Not mirrored yet: the walking person (follow-me tests stay in MuJoCo), articulated CMA-house doors, the lidar
(MuJoCo ray casts on the voxel colliders; RTX Lidar is a later option), the robot's own body in the view.

```bash
MUJOCO_GL=egl ~/test/isaac-rl/bin/python sim/home_sim/eval_semantic.py --world kujiale_0026 --camera isaac
```

Needs Isaac Sim 6.1 in `~/test/isaac-rl` (or `ISAAC_PYTHON`) and the NVIDIA Omniverse licence accepted
(`OMNI_KIT_ACCEPT_EULA=YES` is set by the bridge; the owner accepted it on 2026-11-20).
