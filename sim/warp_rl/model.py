"""MuJoCo model for GPU training (MuJoCo Warp): the robot of models/robo_bipedal.xml with simplified
collision, on one large terrain shared by all robots.

Inertias, joints, closed loops, actuators and sensors are those of the CAD-generated MJCF. Only collision
changes:
  * tires: the same exact cylinders;
  * landing gear: one sphere at each tip (what the parked robot rests on);
  * (collision="full") chassis as a box and legs as capsules along the main axis of their hulls.
The ~70 convex hulls make ~20 contacts per robot and slow MuJoCo Warp down; against the heightfield every
part is tested with every cell below it, so training uses collision="minimal" (tires + gear tips): falls
are detected from the geometry (tilt/height) and end the episode.

Collision bits: tires and gear tips touch everything; chassis and legs ("full") touch the plane and the box
features (stairs, curbs, bars) but not the heightfield, whose cell-by-cell test overflows with large capsules.
"""
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
ROBOT_XML = SIM / "models" / "robo_bipedal.xml"
FLAT_XML = SIM / "models" / "cena_plana.xml"
COLLISION_GROUP = 3
TOUCH_HFIELD, TOUCH_BOXES = 1, 4                # contype bits of the heightfield and of the box features
TOUCH_ALL = TOUCH_HFIELD | TOUCH_BOXES


def _body_hull_vertices(model):
    """Collision hull vertices of each body, in the body frame."""
    out = {}
    for g in range(model.ngeom):
        if model.geom_group[g] != COLLISION_GROUP or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        out.setdefault(name, []).append(v @ R.reshape(3, 3).T + model.geom_pos[g])
    return {k: np.concatenate(v) for k, v in out.items()}


def _capsule(v, radius_percentile=85.0):
    """Capsule along the principal (PCA) axis of the vertices: (point0, point1, radius)."""
    c = v.mean(0)
    axis = np.linalg.svd(v - c, full_matrices=False)[2][0]
    t = (v - c) @ axis
    perp = np.linalg.norm((v - c) - np.outer(t, axis), axis=1)
    r = float(np.percentile(perp, radius_percentile))
    lo, hi = t.min() + r, t.max() - r
    if hi <= lo:
        lo = hi = 0.5 * (t.min() + t.max())
    return c + lo * axis, c + hi * axis, max(r, 0.01)


# Hip joint range [deg] from the CAD pose: the closed 5-bar must not fold through itself — the front arm (Yp) never
# rotates back past pointing straight down, the rear arm (Ym) never forward past it (swept with the CAD collision
# hulls, self-collision on). Beyond this the mechanism flips to its other branch.
HIP_RANGE_DEG = {"hip_Yp": (-49.0, 118.0), "hip_Ym": (-124.0, 47.0)}


def apply_hip_limits(spec):
    import math
    for j in spec.joints:
        for prefix, (lo, hi) in HIP_RANGE_DEG.items():
            if j.name.startswith(prefix + "_"):
                j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                j.range = [lo, hi] if spec.compiler.degree else [math.radians(lo), math.radians(hi)]   # MJCF units


def robot_spec(collision="minimal"):
    """Robot MjSpec with simplified collision ("minimal": tires + gear tips; "full": + chassis and legs)."""
    source = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    verts = _body_hull_vertices(source)
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    apply_hip_limits(spec)
    for geom in list(spec.geoms):
        if geom.group == COLLISION_GROUP and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
        elif geom.name.endswith("_tire"):
            geom.conaffinity = TOUCH_ALL
    for body in spec.bodies:
        if body.name not in verts:
            continue
        v = verts[body.name]
        common = dict(group=COLLISION_GROUP, contype=2, conaffinity=TOUCH_ALL, density=0.0,
                      friction=[0.8, 0.02, 0.002], rgba=[0.8, 0.3, 0.3, 0.4])
        if body.name.startswith("gear"):
            p0, p1, r = _capsule(v)
            tip = p0 if np.linalg.norm(p0) > np.linalg.norm(p1) else p1      # tip = far from the hinge
            body.add_geom(name=f"{body.name}_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=tip,
                          size=[r, 0, 0], **common)
        elif collision != "full":
            continue
        elif body.name == "base":
            common["conaffinity"] = TOUCH_BOXES
            lo, hi = v.min(0), v.max(0)
            body.add_geom(name="base_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(lo + hi) / 2,
                          size=(hi - lo) / 2, **common)
        else:
            common["conaffinity"] = TOUCH_BOXES
            p0, p1, r = _capsule(v)
            body.add_geom(name=f"{body.name}_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=np.concatenate([p0, p1]), size=[r, 0, 0], **common)
    return spec


def parked_qpos():
    """qpos of the scenes' "parked" keyframe (start pose: legs at 0.18 m, gear down)."""
    flat = mujoco.MjModel.from_xml_path(str(FLAT_XML))
    k = [mujoco.mj_id2name(flat, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(flat.nkey)].index("parked")
    return flat.key_qpos[k].copy()


def build(terrain=None, timestep=0.001, collision="minimal"):
    """Robot + ground. `terrain` = None (plane) or a terrain.make_terrain() dict (shared hfield)."""
    spec = robot_spec(collision)
    spec.option.timestep = timestep
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.37]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.quality.shadowsize = 8192
    world = spec.worldbody
    world.add_light(name="sun", pos=[0, 0, 30], dir=[-0.35, 0.45, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.75, 0.73, 0.7], castshadow=True)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[0.62, 0.64, 0.60], rgb2=[0.50, 0.53, 0.50])
    mat = spec.add_material(name="ground", texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=256,
                     rgb1=[0.45, 0.62, 0.82], rgb2=[0.86, 0.90, 0.95])
    ground = dict(contype=TOUCH_ALL, conaffinity=0, friction=[1.2, 0.02, 0.002], material="ground")
    if terrain is None:
        world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], **ground)
    else:
        h = terrain["heights"]
        zmax = max(float(h.max() - h.min()), 1e-3)
        spec.add_hfield(name="terrain", nrow=h.shape[0], ncol=h.shape[1],
                        size=[terrain["size_x"] / 2, terrain["size_y"] / 2, zmax, 0.1],
                        userdata=((h - h.min()) / zmax).ravel().tolist())
        world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                       pos=[0, 0, float(h.min())], **{**ground, "contype": TOUCH_HFIELD})
        for i, (c, sz) in enumerate(terrain.get("boxes", [])):          # sharp features (stairs, curbs, islands)
            world.add_geom(name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=sz,
                           **{**ground, "contype": TOUCH_BOXES})
        for i, (c, sz) in enumerate(terrain.get("bars", [])):           # overhead bars
            world.add_geom(name=f"bar{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=sz,
                           **{**ground, "contype": TOUCH_BOXES, "rgba": [0.8, 0.5, 0.2, 1]})
    return spec.compile(), spec
