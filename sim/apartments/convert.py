"""Realistic apartments (InteriorAgent USD scenes, Isaac Sim ready) -> MuJoCo scene + ground truth.

    OMNI_KIT_ACCEPT_EULA=YES ~/test/isaac-rl/bin/python sim/apartments/convert.py kujiale_0026 [--voxel 0.04]

Input:  sim/isaac/usd/interioragent/<name>/<name>.usda + rooms.json (download: sim/apartments/README.md)
Output (generated, ignored by Git):
    sim/models/apartments/<name>/world.xml     static colliders (merged voxel boxes, geom group 4, cast by the lidar)
                                               + the same boxes as visuals (group 1, coloured per class) so the
                                               MuJoCo cameras see something when Isaac does not render
    sim/models/apartments/<name>/gt.json       rooms (polygons, type, Portuguese names), objects (class in the
                                               detector's vocabulary, room, box), spawns, removed doors
    sim/models/cena_<name>.xml                 robot + apartment, keyframes "parked"/"home" + one per room

Why voxels: MuJoCo collides meshes by their convex hull, so a table or a wall with a doorway cannot be a mesh
collider. Every mesh is sampled on its surface, rasterised at `voxel` metres and merged into boxes (per object, so
each box keeps its class colour). Door leaves are removed (doors open), curtains are visual only, ceilings and
everything above 2.2 m are dropped, and the floor is the plane z = 0.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMUS = HERE.parents[1]
MODELS = PRIMUS / "sim" / "models"
SRC = PRIMUS / "sim" / "isaac" / "usd" / "interioragent"

ZMAX = 2.2
ZMIN = 0.025                       # door sills (1 cm) and rugs are not obstacles for the wheels
SKIP_COLLISION = ("door", "door_handle", "curtain")       # removed from collision (doors open; cloth)
SKIP_ALL = ("door", "door_handle")                        # removed from the stage too (see isaac_stage.py)

# InteriorAgent room group / rooms.json type -> our room type + Portuguese names
ROOM_TYPES = {
    "livingroom": ("living room", ["sala", "sala de estar", "sala de tv"]),
    "living room": ("living room", ["sala", "sala de estar", "sala de tv"]),
    "diningroom": ("dining room", ["sala de jantar"]),
    "dining room": ("dining room", ["sala de jantar"]),
    "kitchen": ("kitchen", ["cozinha"]),
    "bedroom": ("bedroom", ["quarto"]),
    "bathroom": ("bathroom", ["banheiro"]),
    "balcony": ("balcony", ["varanda", "sacada"]),
    "studyroom": ("office", ["escritório"]),
    "study room": ("office", ["escritório"]),
    "storage": ("storage", ["depósito", "despensa"]),
    "unknown": ("hallway", ["corredor"]),
}
CLASS_PT = {
    "sofa": "sofá", "armchair": "poltrona", "chair": "cadeira", "stool": "banqueta", "bed": "cama",
    "dining table": "mesa de jantar", "coffee table": "mesa de centro", "side table": "mesa lateral",
    "desk": "escrivaninha", "television": "televisão", "bookshelf": "estante", "wardrobe": "guarda-roupa",
    "refrigerator": "geladeira", "stove": "fogão", "kitchen sink": "pia da cozinha",
    "bathroom sink": "pia do banheiro", "toilet": "vaso sanitário", "washing machine": "máquina de lavar",
    "potted plant": "planta", "nightstand": "criado-mudo", "sideboard": "aparador",
    "kitchen cabinet": "armário da cozinha", "floor lamp": "luminária", "bathtub": "banheira",
}
COLOURS = {   # visual colour per class family (MuJoCo fallback renders only)
    "wall": (0.86, 0.84, 0.80), "window": (0.6, 0.75, 0.85), "sofa": (0.45, 0.47, 0.55), "bed": (0.75, 0.7, 0.62),
    "chair": (0.55, 0.38, 0.25), "table": (0.62, 0.45, 0.3), "cabinet": (0.7, 0.62, 0.5), "appliance": (0.8, 0.8, 0.82),
    "bath": (0.92, 0.93, 0.95), "plant": (0.3, 0.55, 0.3), "other": (0.6, 0.58, 0.55),
}


def classify(raw, room, lo, hi):
    """InteriorAgent object name (without the _NNNN) + room + box -> detector class or None."""
    h = hi[2] - lo[2]
    w = sorted(hi[:2] - lo[:2])
    area = w[0] * w[1]
    if raw in ("sofa",):
        return "sofa" if w[1] > 1.1 else "armchair"
    if raw in ("chair",):
        return "chair"
    if raw == "stool":
        return "stool"
    if raw == "bed":
        return "bed"
    if raw == "dining_table":
        return "dining table"
    if raw == "desk":
        return "desk"
    if raw == "table":
        if hi[2] < 0.55 and lo[2] < 0.1:
            return "coffee table" if area > 0.35 else "side table"
        if hi[2] < 0.9 and lo[2] < 0.1:
            return "dining table" if room in ("dining room", "kitchen") or area > 0.9 else (
                "desk" if room in ("office", "bedroom") else "side table")
        return None
    if raw == "television":
        return "television"
    if raw == "shelf":
        return "bookshelf" if h > 0.8 else None
    if raw in ("cabinet", "storage"):
        if lo[2] > 0.3:                                    # wall-hung
            return None
        if room == "kitchen":
            return "kitchen cabinet"
        if room == "bathroom":
            return None
        if h > 1.5:
            return "wardrobe"
        if h < 0.75 and w[1] < 0.7 and room == "bedroom":
            return "nightstand"
        return "sideboard" if h < 1.2 else "bookshelf"
    if raw == "fridge":
        return "refrigerator"
    if raw == "other_cooker":
        return "stove"
    if raw == "basin":
        return "kitchen sink" if room == "kitchen" else "bathroom sink"
    if raw == "closestool":
        return "toilet"
    if raw == "washing_machine":
        return "washing machine"
    if raw == "flower" and lo[2] < 0.1 and h > 0.4:
        return "potted plant"
    return None


def colour_family(raw):
    for k, fam in (("wall", "wall"), ("window", "window"), ("sofa", "sofa"), ("bed", "bed"), ("chair", "chair"),
                   ("stool", "chair"), ("table", "table"), ("desk", "table"), ("cabinet", "cabinet"),
                   ("storage", "cabinet"), ("shelf", "cabinet"), ("fridge", "appliance"),
                   ("washing", "appliance"), ("cooker", "appliance"), ("basin", "bath"), ("closestool", "bath"),
                   ("flower", "plant")):
        if k in raw:
            return fam
    return "other"


# ------------------------------------------------------------------------------------------------ USD reading
def read_meshes(usda):
    """-> list of dict(name, raw, group, tris [n, 3, 3] world)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(usda))
    cache = UsdGeom.XformCache()
    out = []
    root = st.GetPrimAtPath("/Root/Meshes")
    for grp in root.GetChildren():
        gname = grp.GetName()
        if gname == "ceiling":
            continue
        for obj in grp.GetChildren():
            name = obj.GetName()
            raw = re.sub(r"_\d+$", "", name)
            tris = []
            for p in Usd.PrimRange(obj):
                if p.GetTypeName() != "Mesh":
                    continue
                m = UsdGeom.Mesh(p)
                pts = m.GetPointsAttr().Get()
                fvc = m.GetFaceVertexCountsAttr().Get()
                fvi = m.GetFaceVertexIndicesAttr().Get()
                if pts is None or fvc is None or len(pts) == 0:
                    continue
                P = np.asarray(pts, np.float64)
                M = np.asarray(cache.GetLocalToWorldTransform(p), np.float64)
                P = P @ M[:3, :3] + M[3, :3]
                fvc = np.asarray(fvc)
                fvi = np.asarray(fvi)
                starts = np.concatenate([[0], np.cumsum(fvc)[:-1]])
                for k in np.unique(fvc):                     # fan triangulation per polygon size
                    if k < 3:
                        continue
                    s = starts[fvc == k]
                    for j in range(1, k - 1):
                        tris.append(P[np.stack([fvi[s], fvi[s + j], fvi[s + j + 1]], 1)])
            if tris:
                out.append(dict(name=name, raw=raw, group=gname, tris=np.concatenate(tris)))
    return out


def sample_surface(tris, h):
    """Points on the triangles with spacing <= h."""
    e = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1), np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                  np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)], 1).max(1)
    n = np.clip(np.ceil(e / h).astype(int), 1, 400)
    pts = [tris.reshape(-1, 3)]
    for k in np.unique(n):
        if k == 1:
            continue
        T = tris[n == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        a, b = i[keep] / k, j[keep] / k
        c = 1.0 - a - b
        pts.append((a[None, :, None] * T[:, None, 0] + b[None, :, None] * T[:, None, 1]
                    + c[None, :, None] * T[:, None, 2]).reshape(-1, 3))
    return np.concatenate(pts)


# ------------------------------------------------------------------------------------------------ voxel boxes
def merge_boxes(lab):
    """Greedy merge of a 3-D label grid (0 = empty) into boxes of one label: [(label, i0, j0, k0, i1, j1, k1)]."""
    from numba import njit

    @njit(cache=True)
    def run(lab):
        nx, ny, nz = lab.shape
        used = np.zeros(lab.shape, np.bool_)
        out = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    L = lab[i, j, k]
                    if L == 0 or used[i, j, k]:
                        continue
                    i1 = i
                    while i1 + 1 < nx and lab[i1 + 1, j, k] == L and not used[i1 + 1, j, k]:
                        i1 += 1
                    j1 = j
                    ok = True
                    while ok and j1 + 1 < ny:
                        for ii in range(i, i1 + 1):
                            if lab[ii, j1 + 1, k] != L or used[ii, j1 + 1, k]:
                                ok = False
                                break
                        if ok:
                            j1 += 1
                    k1 = k
                    ok = True
                    while ok and k1 + 1 < nz:
                        for jj in range(j, j1 + 1):
                            for ii in range(i, i1 + 1):
                                if lab[ii, jj, k1 + 1] != L or used[ii, jj, k1 + 1]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            k1 += 1
                    used[i:i1 + 1, j:j1 + 1, k:k1 + 1] = True
                    out.append((L, i, j, k, i1, j1, k1))
        return out
    return run(lab)


def polygon_contains(poly, x, y):
    from matplotlib.path import Path as MPath
    return MPath(np.asarray(poly)).contains_points(np.stack([x, y], -1))


# ------------------------------------------------------------------------------------------------ main
def convert(name, voxel=0.04):
    src = SRC / name
    usda = src / f"{name}.usda"
    out = MODELS / "apartments" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"reading {usda} ...")
    meshes = read_meshes(usda)
    rooms_raw = json.loads((src / "rooms.json").read_text())

    # rooms: unique names, our types, Portuguese names
    rooms, count = [], {}
    for r in rooms_raw:
        typ, pt = ROOM_TYPES.get(r["room_type"].lower(), ("hallway", ["corredor"]))
        count[typ] = count.get(typ, 0) + 1
        rooms.append(dict(type=typ, names_pt=list(pt), polygon=[list(map(float, p)) for p in r["polygon"]],
                          floor="main", z=0.0, indoor=True))
    for typ, n in count.items():
        k = 0
        for r in rooms:
            if r["type"] == typ:
                k += 1
                r["name"] = typ if n == 1 else f"{typ} {k}"
                if n > 1:
                    r["names_pt"] = [f"{p} {k}" for p in r["names_pt"]] + r["names_pt"]

    def room_of(x, y):
        for r in rooms:
            if polygon_contains(r["polygon"], np.array([x]), np.array([y]))[0]:
                return r
        return None

    # objects + voxels
    allpts = np.concatenate([m["tris"].reshape(-1, 3) for m in meshes if m["raw"] not in SKIP_COLLISION])
    keep = (allpts[:, 2] > -0.1) & (allpts[:, 2] < ZMAX)
    lo = allpts[keep].min(0) - 0.2
    hi = allpts[keep].max(0) + 0.2
    lo[2], hi[2] = 0.0, ZMAX
    shape = np.ceil((hi - lo) / voxel).astype(int)
    lab = np.zeros(shape, np.int32)
    labels = [None]
    objects, doors = [], []
    for m in meshes:
        T = m["tris"]
        P = T.reshape(-1, 3)
        olo, ohi = P.min(0), P.max(0)
        if m["raw"] in SKIP_COLLISION:
            if m["raw"] == "door":
                doors.append(dict(name=m["name"], bbox=[olo.round(3).tolist(), ohi.round(3).tolist()]))
            continue
        if m["group"] not in ("wall", "floor", "other"):
            c = 0.5 * (olo + ohi)
            r = room_of(c[0], c[1])
            rtype = r["type"] if r else ROOM_TYPES.get(re.sub(r"_\d+$", "", m["group"]), ("hallway",))[0]
            cls = classify(m["raw"], rtype, olo, ohi)
            if cls is not None:
                objects.append(dict(id=m["name"], cls=cls, name_pt=CLASS_PT.get(cls, cls), raw=m["raw"],
                                    center=c.round(3).tolist(),
                                    bbox=[olo.round(3).tolist(), ohi.round(3).tolist()], floor="main",
                                    room=r["name"] if r else None))
        if m["group"] == "floor":
            continue
        pts = sample_surface(T, 0.5 * voxel)
        pts = pts[(pts[:, 2] > ZMIN) & (pts[:, 2] < ZMAX)]
        if len(pts) == 0:
            continue
        idx = np.floor((pts - lo) / voxel).astype(int)
        idx = np.clip(idx, 0, shape - 1)
        labels.append(m)
        lab[idx[:, 0], idx[:, 1], idx[:, 2]] = len(labels) - 1
    print(f"{len(meshes)} meshes, grid {shape.tolist()} at {voxel} m, {int((lab > 0).sum())} occupied voxels")
    boxes = merge_boxes(lab)
    print(f"{len(boxes)} boxes")

    # 2-D obstacle map (0.03-0.9 m, the robot's height band) -> spawn points with clearance
    from scipy import ndimage
    occ2 = (lab[:, :, : int(math.ceil(0.9 / voxel))] > 0).any(2)
    inside = np.zeros_like(occ2)
    gx = lo[0] + (np.arange(shape[0]) + 0.5) * voxel
    gy = lo[1] + (np.arange(shape[1]) + 0.5) * voxel
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    spawns = {}
    clear = ndimage.distance_transform_edt(~occ2) * voxel
    for r in rooms:
        m = polygon_contains(r["polygon"], X.ravel(), Y.ravel()).reshape(X.shape)
        inside |= m
        c = np.where(m, clear, 0.0)
        if c.max() < 0.4:
            continue
        i, j = np.unravel_index(np.argmax(c), c.shape)
        key = r["name"].replace(" ", "_")
        spawns[key] = [float(gx[i]), float(gy[j]), 0.0, 0.0, round(float(c[i, j]), 2)]
    living = next((k for k in spawns if k.startswith("living")), next(iter(spawns)))

    # world.xml
    F = lambda v: " ".join(f"{x:.4g}" for x in v)          # noqa: E731
    lines = ["<mujoco>", "  <worldbody>"]
    lines.append('    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.62 0.55 0.47 1" friction="1.0 0.02 0.002"'
                 ' contype="1" conaffinity="1" group="0"/>')
    for n, (L, i0, j0, k0, i1, j1, k1) in enumerate(boxes):
        bl = lo + np.array([i0, j0, k0]) * voxel
        bh = lo + np.array([i1 + 1, j1 + 1, k1 + 1]) * voxel
        bh[2] = min(bh[2], ZMAX)
        c, s = 0.5 * (bl + bh), 0.5 * (bh - bl)
        col = COLOURS[colour_family(labels[L]["raw"])]
        lines.append(f'    <geom name="c{n}" type="box" pos="{F(c)}" size="{F(s)}" friction="0.95 0.02 0.002" '
                     f'contype="1" conaffinity="1" group="4" rgba="0.95 0.45 0.1 0.35"/>')
        lines.append(f'    <geom name="v{n}" type="box" pos="{F(c)}" size="{F(s)}" contype="0" conaffinity="0" '
                     f'group="1" rgba="{F(col)} 1"/>')
    lines += ["  </worldbody>", "</mujoco>"]
    (out / "world.xml").write_text("\n".join(lines) + "\n")

    gt = dict(source=f"InteriorAgent {name} (Manycore Tech, non-commercial research use)", frame="Isaac world frame",
              floors={"main": 0.0}, rooms=rooms, objects=objects, doors_removed=doors,
              spawns={k: v[:4] for k, v in spawns.items()}, spawn_clearance={k: v[4] for k, v in spawns.items()},
              usd=str(usda.relative_to(PRIMUS)), voxel=voxel, bounds=[lo.tolist(), hi.tolist()])
    (out / "gt.json").write_text(json.dumps(gt, indent=1, ensure_ascii=False))

    # scene with the robot and keyframes
    import mujoco
    flat = (MODELS / "cena_plana.xml").read_text()
    rk = {k: [float(x) for x in v.split()] for k, v in re.findall(r'<key name="(\w+)" qpos="([^"]+)"', flat)}
    scene = MODELS / f"cena_{name}.xml"

    def yaw_quat(heading):
        a = (heading - math.pi / 2) / 2
        return [math.cos(a), 0.0, 0.0, math.sin(a)]

    def write(keys):
        L = [f'<mujoco model="robo_bipedal_{name}">',
             f"  <!-- GENERATED by sim/apartments/convert.py from InteriorAgent {name}. Do not edit. -->",
             '  <include file="robo_bipedal.xml"/>',
             f'  <include file="apartments/{name}/world.xml"/>',
             "  <visual>",
             '    <global azimuth="-60" elevation="-30" offwidth="1920" offheight="1080" fovy="50"/>',
             '    <headlight ambient="0.45 0.45 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>',
             '    <map znear="0.01" zfar="30"/>',
             "  </visual>",
             f'  <statistic center="{F(0.5 * (lo + hi))}" extent="{float(np.max(hi - lo)):.2f}"/>',
             '  <worldbody><light name="sun" directional="true" pos="0 0 10" dir="0.2 0.3 -1" castshadow="false"'
             ' diffuse="0.6 0.6 0.6"/></worldbody>',
             "  <keyframe>"]
        for k in keys:
            L.append(f'    <key name="{k[0]}" qpos="{F(k[1])}"/>')
        L += ["  </keyframe>", "</mujoco>"]
        scene.write_text("\n".join(L) + "\n")

    write([])
    model = mujoco.MjModel.from_xml_path(str(scene))
    keys = []
    order = [living] + [k for k in spawns if k != living]
    for n, sp in enumerate(order):
        x, y = spawns[sp][:2]
        for base in (("parked", "home") if n == 0 else ("parked",)):
            q = model.qpos0.copy()
            q[:len(rk[base])] = rk[base]
            q[0:3] = [x, y, rk[base][2]]
            q[3:7] = yaw_quat(0.0)
            keys.append((base if n == 0 else sp, q))
    write(keys)
    model = mujoco.MjModel.from_xml_path(str(scene))
    print(f"wrote {scene}: ngeom {model.ngeom}, {len(rooms)} rooms, {len(objects)} objects, spawns {order}")
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--voxel", type=float, default=0.04)
    a = ap.parse_args()
    for n in a.names:
        convert(n, a.voxel)


if __name__ == "__main__":
    main()
