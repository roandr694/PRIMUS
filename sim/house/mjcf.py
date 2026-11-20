"""Writes a `Scene` (geom.py) as a MuJoCo model: meshes (.msh), textures, materials, colliders, articulated
bodies, lights, the HDRI skybox and keyframes that include the robot.

Output layout (all generated, not in Git):
    sim/models/house/meshes/*.msh        visual meshes (static ones merged per material)
    sim/models/house/textures/*.png      baked textures (textures.py) + sky.png
    sim/models/house/house_assets.xml    <asset> block
    sim/models/house/house_world.xml     <worldbody> content of the house (included by the scene)
    sim/models/cena_casa_cma.xml         scene = robot + house + keyframes
"""
import math
from pathlib import Path

import numpy as np

from geom import MOVING_CA, MOVING_CT, STATIC_CA, STATIC_CT, mat2quat, write_msh

F = lambda x: f"{x:.6g}"                                     # noqa: E731
# House colliders live in geom group 4: hidden in the renderers (default visible groups 0-2), cast by the
# simulated lidar (skill_world.SimSensors includes group 4), shown in the MuJoCo viewer with the group-4 toggle.
# (mj_ray ignores alpha-0 geoms, so invisible group-0 colliders would be missed by the lidar.)
COLLISION_GROUP = 4


def vec(a):
    return " ".join(F(float(x)) for x in np.atleast_1d(a))


def _quat_attr(R):
    if R is None or np.allclose(R, np.eye(3), atol=1e-9):
        return ""
    return f' quat="{vec(mat2quat(R))}"'


# ----------------------------------------------------------------------------------------------------------------
# sky
# ----------------------------------------------------------------------------------------------------------------
def make_skybox(hdr_path, out_png, face=1024, sun_azimuth_deg=-150.0, exposure=1.0):
    """Equirectangular HDRI -> MuJoCo skybox (3x4 cross, '.U..LFRB.D..'), rotated so the HDRI's brightest point
    (the sun) sits at the requested azimuth (degrees, from +X towards +Y). Returns the sun direction (unit vector
    pointing from the scene to the sun) for the directional light."""
    import cv2
    from PIL import Image
    hdr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
    H, W, _ = hdr.shape
    lum = hdr.mean(axis=2)
    small = cv2.GaussianBlur(lum, (0, 0), 3)
    sy, sx = np.unravel_index(np.argmax(small[: H // 2]), small[: H // 2].shape)
    sun_lon = (0.5 - sx / W) * 2 * math.pi          # longitude convention used below
    sun_lat = (0.5 - sy / H) * math.pi
    rot = math.radians(sun_azimuth_deg) - sun_lon    # azimuth rotation applied to the sky

    def lookup(d):
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        lon = np.arctan2(y, x) - rot
        lat = np.arcsin(np.clip(z, -1, 1))
        u = (0.5 - lon / (2 * math.pi)) % 1.0
        v = np.clip(0.5 - lat / math.pi, 0, 0.9999)
        return hdr[(v * H).astype(int), (u * W).astype(int) % W]

    # OpenGL cube-map face directions, then rotated +90 deg about X (MuJoCo renders the skybox with Z up)
    s = (np.arange(face) + 0.5) / face * 2 - 1
    sc, tc = np.meshgrid(s, s)
    one = np.ones_like(sc)
    gl = {"R": (one, -tc, -sc), "L": (-one, -tc, sc), "U": (sc, one, tc), "D": (sc, -one, -tc),
          "F": (sc, -tc, one), "B": (-sc, -tc, -one)}
    faces = {}
    for k, (x, y, z) in gl.items():
        d = np.stack([x, -z, y], axis=-1)            # Rx(+90): (x, y, z) -> (x, -z, y)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        c = lookup(d) * exposure
        c = c / (1 + c * 0.35)                        # soft Reinhard so the sun disc saturates gently
        faces[k] = np.clip(c ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
    grid = np.zeros((4 * face, 3 * face, 3), np.uint8)
    layout = [".U.", "LFR", ".D.", ".B."]            # gridsize 4 3 rows x cols -> '.U.LFR.D..B.'
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch != ".":
                grid[r * face:(r + 1) * face, c * face:(c + 1) * face] = faces[ch]
    Image.fromarray(grid).save(out_png)
    sun_az = math.radians(sun_azimuth_deg)
    make_skybox.rotation_deg = math.degrees(rot)               # for the USD dome light
    return np.array([math.cos(sun_lat) * math.cos(sun_az), math.cos(sun_lat) * math.sin(sun_az),
                     math.sin(sun_lat)]), math.degrees(sun_lat)


# ----------------------------------------------------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------------------------------------------------
def write(scene, models_dir, robot_xml="robo_bipedal.xml", scene_name="cena_casa_cma.xml", keyframes=(),
          sky_png=None, sun_dir=None, reflections=True):
    models_dir = Path(models_dir)
    hdir = models_dir / "house"
    mdir = hdir / "meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("*.msh"):
        old.unlink()
    mats = scene.materials
    used = set()

    asset = ["<mujoco>", "  <asset>"]
    if sky_png:
        asset.append(f'    <texture name="sky" type="skybox" file="house/textures/{Path(sky_png).name}" '
                     f'gridsize="4 3" gridlayout=".U.LFR.D..B."/>')
    # textures / materials (only those used are emitted later; collect first)
    mesh_lines, static_geoms = [], []
    stats = dict(static_tris=0, body_tris=0, colliders=len(scene.colliders), bodies=len(scene.bodies))

    for mat, buf in sorted(scene.meshes.items()):
        arr = buf.arrays()
        if arr is None:
            continue
        v, n, t, f = arr
        name = f"hs_{mat}"
        write_msh(mdir / f"{name}.msh", v, n, t, f)
        mesh_lines.append(f'    <mesh name="{name}" file="../house/meshes/{name}.msh" inertia="shell"/>')
        static_geoms.append(f'    <geom name="vis_{mat}" type="mesh" mesh="{name}" material="hm_{mat}" '
                            f'contype="0" conaffinity="0" group="1"/>')
        used.add(mat)
        stats["static_tris"] += len(f)

    # visual planes (water surfaces) are real MuJoCo planes so they get mirror reflections
    for p in scene.planes:
        used.add(p["mat"])
        static_geoms.append(
            f'    <geom name="{p["name"]}" type="plane" pos="{vec(p["pos"])}" size="{vec(p["half"])} 0.1"'
            f'{_quat_attr(p.get("R"))} material="hm_{p["mat"]}" contype="0" conaffinity="0" group="1"/>')

    col_lines = []
    for i, c in enumerate(scene.colliders):
        col_lines.append(_collider_xml(c, i, STATIC_CT, STATIC_CA, "c"))
    for h in scene.hfields:
        data = np.asarray(h["heights"], np.float32)              # (nrow, ncol), row 0 = -y edge
        lo, hi = float(data.min()), float(data.max())
        norm = (data - lo) / max(hi - lo, 1e-6)
        np.save(hdir / f"{h['name']}_heights.npy", data)        # for the USD export / other tools
        rx, ry = h["half"]
        elev = " ".join(f"{x:.4f}" for x in norm.ravel())        # inline: no file-path resolution issues
        mesh_lines.append(f'    <hfield name="{h["name"]}" nrow="{data.shape[0]}" ncol="{data.shape[1]}" '
                          f'size="{F(rx)} {F(ry)} {F(hi - lo)} 0.5" elevation="{elev}"/>')
        col_lines.append(f'    <geom name="{h["name"]}" type="hfield" hfield="{h["name"]}" '
                         f'pos="{vec((h["centre"][0], h["centre"][1], lo))}" friction="{vec(h["friction"])}" '
                         f'contype="{STATIC_CT}" conaffinity="{STATIC_CA}" group="{COLLISION_GROUP}" '
                         f'rgba="0.3 0.8 0.3 0.3"/>')

    body_lines = []
    for b in scene.bodies:
        body_lines += _body_xml(b, mdir, mesh_lines, used, stats)

    for mat in sorted(used):
        m = mats[mat]
        tex = ""
        if m.texture:
            asset.append(f'    <texture name="ht_{m.texture}" type="2d" file="house/textures/{m.texture}.png"/>'
                         if f'name="ht_{m.texture}"' not in "\n".join(asset) else "")
            tex = f' texture="ht_{m.texture}" texrepeat="1 1"'
        if m.texture and m.extra.get("water"):          # used on plane geoms: repeat per metre
            tex = f' texture="ht_{m.texture}" texrepeat="{F(1 / m.tile)} {F(1 / m.tile)}" texuniform="true"'
        refl = m.reflectance if reflections else 0.0
        asset.append(f'    <material name="hm_{mat}"{tex} rgba="{vec(m.rgba)}" specular="{F(m.specular)}" '
                     f'shininess="{F(m.shininess)}" reflectance="{F(refl)}" emission="{F(m.emission)}"/>')
    asset = [a for a in asset if a]
    asset += mesh_lines + ["  </asset>", "</mujoco>"]
    (hdir / "house_assets.xml").write_text("\n".join(asset) + "\n")

    world = ["<mujoco>", "  <worldbody>"]
    for L in scene.lights:
        world.append("    " + _light_xml(L))
    world += static_geoms + col_lines
    for s in scene.sites:
        world.append(f'    <site name="{s["name"]}" pos="{vec(s["pos"])}" size="{F(s.get("size", 0.05))}" '
                     f'rgba="1 0 0 0" group="5"/>')
    for cam in scene.cameras:
        world.append(f'    <camera name="{cam["name"]}" pos="{vec(cam["pos"])}" xyaxes="{vec(cam["xyaxes"])}" '
                     f'fovy="{F(cam.get("fovy", 55))}"/>')
    world += body_lines
    world += ["  </worldbody>"]
    if scene.actuators:
        world.append("  <actuator>")
        for a in scene.actuators:
            world.append("    " + a)
        world.append("  </actuator>")
    world.append("</mujoco>")
    (hdir / "house_world.xml").write_text("\n".join(world) + "\n")

    _write_scene(models_dir / scene_name, robot_xml, scene, keyframes, sun_dir)
    return stats


def _collider_xml(c, i, ct, ca, prefix):
    kind, size = c["kind"], c["size"]
    name = c.get("name") or f"{prefix}{i}"
    fr = c["friction"]
    sol = f' solref="{vec(c["solref"])}"' if c.get("solref") is not None else ""
    if kind == "box":
        sz = vec(size)
    elif kind == "cylinder" or kind == "capsule":
        sz = vec(size[:2])
    else:
        sz = vec(size[:1])
    return (f'    <geom name="{name}" type="{kind}" pos="{vec(c["pos"])}" size="{sz}"{_quat_attr(c["R"])} '
            f'friction="{vec(fr)}"{sol} contype="{ct}" conaffinity="{ca}" group="{COLLISION_GROUP}" '
            f'rgba="{"0.95 0.45 0.1 0.35" if ct == STATIC_CT else "0.1 0.6 0.95 0.45"}"/>')


def _body_xml(b, mdir, mesh_lines, used, stats, indent="    "):
    out = []
    q = _quat_attr(b.frame.R)
    gc = f' gravcomp="{F(b.gravcomp)}"' if b.gravcomp else ""
    out.append(f'{indent}<body name="{b.name}" pos="{vec(b.frame.t)}"{q}{gc}>')
    j = b.joint
    if j:
        attrs = [f'name="{b.name}_j"', f'type="{j.get("type", "hinge")}"', f'axis="{vec(j.get("axis", (0, 0, 1)))}"']
        # the robot's <compiler angle="degree"> applies to the whole scene: hinge angles are written in degrees
        hinge = j.get("type", "hinge") == "hinge"
        ang = (lambda x: np.degrees(x)) if hinge else (lambda x: x)
        if "pos" in j:
            attrs.append(f'pos="{vec(j["pos"])}"')
        if "range" in j:
            attrs.append(f'range="{vec(ang(np.asarray(j["range"], float)))}"')
        for k in ("damping", "stiffness", "frictionloss", "armature"):
            if k in j:
                attrs.append(f'{k}="{F(j[k])}"')
        for k in ("springref", "ref"):
            if k in j:
                attrs.append(f'{k}="{F(float(ang(j[k])))}"')
        out.append(f'{indent}  <joint {" ".join(attrs)}/>')
    if b.mass is not None:
        I = list(b.inertia if b.inertia is not None else (b.mass * 0.05,) * 3)
        for k in range(3):                     # enforce the triangle inequality A + B >= C
            I[k] = min(I[k], 0.995 * (I[(k + 1) % 3] + I[(k + 2) % 3]))
        out.append(f'{indent}  <inertial pos="{vec(b.com)}" mass="{F(b.mass)}" diaginertia="{vec(I)}"/>')
    for mat, buf in sorted(b.meshes.items()):
        arr = buf.arrays()
        if arr is None:
            continue
        v, n, t, f = arr
        name = f"hb_{b.name}_{mat}"
        write_msh(mdir / f"{name}.msh", v, n, t, f)
        mesh_lines.append(f'    <mesh name="{name}" file="../house/meshes/{name}.msh" inertia="shell"/>')
        out.append(f'{indent}  <geom type="mesh" mesh="{name}" material="hm_{mat}" contype="0" conaffinity="0" '
                   f'group="1"/>')
        used.add(mat)
        stats["body_tris"] += len(f)
    for i, c in enumerate(b.colliders):
        out.append(f"{indent}  " + _collider_xml(c, i, MOVING_CT, MOVING_CA, f"{b.name}_c").strip())
    for s in b.sites:
        out.append(f'{indent}  <site name="{s["name"]}" pos="{vec(s["pos"])}" size="{F(s.get("size", 0.03))}" '
                   f'rgba="0 1 0 0" group="5"/>')
    if b.extra_xml:
        out.append(b.extra_xml)
    for ch in b.children:
        out += _body_xml(ch, mdir, mesh_lines, used, stats, indent + "  ")
    out.append(f"{indent}</body>")
    return out


def _light_xml(L):
    attrs = []
    for k, v in L.items():
        if isinstance(v, (tuple, list, np.ndarray)):
            attrs.append(f'{k}="{vec(v)}"')
        elif isinstance(v, bool):
            attrs.append(f'{k}="{"true" if v else "false"}"')
        elif isinstance(v, (int, float)):
            attrs.append(f'{k}="{F(v)}"')
        else:
            attrs.append(f'{k}="{v}"')
    return f"<light {' '.join(attrs)}/>"


def _write_scene(path, robot_xml, scene, keyframes, sun_dir):
    lines = [
        '<mujoco model="robo_bipedal_casa_cma">',
        "  <!-- GENERATED by sim/house/build.py: the PRIMUS robot inside the CMA house (Jacobsen Arquitetura,",
        "       Angra dos Reis), rebuilt from the published plans, section and photographs. Frames: X east (hill),",
        "       Y north, Z up; z = 0 is the living-room floor, the sea is at z = -6.25. Do not edit by hand. -->",
        f'  <include file="{robot_xml}"/>',
        '  <include file="house/house_assets.xml"/>',
        '  <include file="house/house_world.xml"/>',
        "  <visual>",
        '    <global azimuth="-60" elevation="-18" offwidth="1920" offheight="1080" fovy="50"/>',
        '    <headlight ambient="0.40 0.40 0.42" diffuse="0.22 0.22 0.22" specular="0.04 0.04 0.04"/>',
        '    <quality shadowsize="8192" offsamples="8"/>',
        '    <map znear="0.01" zfar="40" haze="0.25" shadowclip="1.2" fogstart="3" fogend="10"/>',
        '    <rgba haze="0.78 0.85 0.92 1" fog="0.78 0.85 0.92 1"/>',
        "  </visual>",
        '  <statistic center="6 8 2" extent="36"/>',
        "  <keyframe>",
    ]
    for k in keyframes:
        act = f' act="{vec(k["act"])}"' if k.get("act") is not None else ""
        ctrl = f' ctrl="{vec(k["ctrl"])}"' if k.get("ctrl") is not None else ""
        lines.append(f'    <key name="{k["name"]}" qpos="{vec(k["qpos"])}"{act}{ctrl}/>')
    lines += ["  </keyframe>", "</mujoco>"]
    Path(path).write_text("\n".join(lines) + "\n")
