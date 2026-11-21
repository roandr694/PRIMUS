# Realistic apartments for the home-autonomy tests

Ready-made Isaac Sim interiors from **InteriorAgent** (Manycore Tech / SpatialVerse,
[Hugging Face](https://huggingface.co/datasets/spatialverse/InteriorAgent)): 25 furnished apartments in USD with
MDL materials, every object named by class (`sofa_0000`, `bed_0001`, ...) and grouped by room, and a `rooms.json`
with the room polygons. **Licence: non-commercial research and education only, no redistribution, cite the
dataset** ([terms](https://kloudsim-usa-cos.kujiale.com/InteriorAgent/InteriorAgent_Terms_of_Use.pdf), accepted by
the owner on 2026-11-20). Nothing from the dataset goes into Git: the USD files live in `sim/isaac/usd/` (ignored)
and everything generated from them in `sim/models/` (ignored).

| scene | area | rooms | objects in the ground truth |
|---|---|---|---|
| `kujiale_0026` | 72 m² | 8 (living, dining, kitchen, bedroom, office, bathroom, 2 balconies) | 29 |
| `kujiale_0004` | 127 m² | 14 | 35 |
| `kujiale_0003` | 169 m² | 12 | 44 |

## Download and convert

```bash
cd PRIMUS
~/test/isaac-rl/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('spatialverse/InteriorAgent', repo_type='dataset', local_dir='sim/isaac/usd/interioragent', allow_patterns=['kujiale_0026/*','kujiale_0003/*','kujiale_0004/*'])"
OMNI_KIT_ACCEPT_EULA=YES ~/test/isaac-rl/bin/python sim/apartments/convert.py kujiale_0026 kujiale_0003 kujiale_0004
```

`convert.py` writes, per apartment:

- `sim/models/cena_<name>.xml`: the robot in the apartment, keyframes `parked` / `home` (living room) and one per
  room (the free spot with the most clearance);
- `sim/models/apartments/<name>/world.xml`: MuJoCo colliders. MuJoCo collides meshes by their convex hull, so a
  table or a wall with a doorway cannot be a mesh collider: every mesh is sampled on its surface, rasterised in
  4 cm voxels and merged into boxes (7-8 k boxes per apartment). The same boxes, coloured per class, are the MuJoCo
  visuals (fallback renders);
- `sim/models/apartments/<name>/gt.json`: rooms (polygon, type, Portuguese names), objects (class mapped to the
  detector's vocabulary, room, box), spawn points, the removed doors.

Choices: **door leaves are removed** (doors open; the robot has no arm), curtains are visual only, ceilings and
everything above 2.2 m are dropped, door sills (1 cm) and rugs are not obstacles, the floor is the plane z = 0.
Some sliding balcony doors keep a fixed panel, so a few balconies are closed off (`kujiale_0026` balcony 2).

## Use

Every `sim/home_sim` evaluation takes `--world <name>` (default `cma`, the generated CMA house), and the semantic
tour takes `--camera isaac` to render the D435i in Isaac Sim instead of MuJoCo (`sim/isaac_home/`).

```bash
MUJOCO_GL=egl ~/test/isaac-rl/bin/python sim/home_sim/eval_mapping.py --world kujiale_0026
MUJOCO_GL=egl ~/test/isaac-rl/bin/python sim/home_sim/eval_semantic.py --world kujiale_0026 --camera isaac
```
