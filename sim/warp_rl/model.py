"""Modelo MuJoCo para treino em GPU (MuJoCo Warp): o mesmo robo de models/robo_bipedal.xml, com
colisao simplificada e um terreno grande compartilhado por todos os robos.

Inercias, juntas, lacos fechados, atuadores e sensores sao os do MJCF gerado do CAD. Muda so a colisao:
  * pneus: os mesmos cilindros exatos;
  * trens de pouso: uma esfera na ponta de cada um (apoio do robo estacionado);
  * (collision="full") chassi como caixa e pernas como capsulas no eixo principal das cascas.
As ~70 cascas convexas geram ~20 contatos por robo e deixam o MuJoCo Warp lento. Contra o heightfield,
cada peca e testada com todas as celulas embaixo dela, entao o treino usa collision="minimal" (pneus +
pontas do trem): a queda e detectada pela geometria (inclinacao/altura) e o episodio termina ali.
"""
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
ROBOT_XML = SIM / "models" / "robo_bipedal.xml"
FLAT_XML = SIM / "models" / "cena_plana.xml"
COLLISION_GROUP = 3


def _body_hull_vertices(model):
    """Vertices das cascas de colisao de cada corpo, no frame do corpo."""
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
    """Capsula no eixo principal (PCA) dos vertices: (ponto0, ponto1, raio)."""
    c = v.mean(0)
    axis = np.linalg.svd(v - c, full_matrices=False)[2][0]
    t = (v - c) @ axis
    perp = np.linalg.norm((v - c) - np.outer(t, axis), axis=1)
    r = float(np.percentile(perp, radius_percentile))
    lo, hi = t.min() + r, t.max() - r
    if hi <= lo:
        lo = hi = 0.5 * (t.min() + t.max())
    return c + lo * axis, c + hi * axis, max(r, 0.01)


def robot_spec(collision="minimal"):
    """MjSpec do robo com a colisao simplificada ("minimal": pneus + pontas do trem; "full": + chassi e pernas)."""
    source = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    verts = _body_hull_vertices(source)
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for geom in list(spec.geoms):
        if geom.group == COLLISION_GROUP and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
    for body in spec.bodies:
        if body.name not in verts:
            continue
        v = verts[body.name]
        common = dict(group=COLLISION_GROUP, contype=2, conaffinity=1, density=0.0,
                      friction=[0.8, 0.02, 0.002], rgba=[0.8, 0.3, 0.3, 0.4])
        if body.name.startswith("gear"):
            p0, p1, r = _capsule(v)
            tip = p0 if np.linalg.norm(p0) > np.linalg.norm(p1) else p1      # ponta = longe da dobradica
            body.add_geom(name=f"{body.name}_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=tip,
                          size=[r, 0, 0], **common)
        elif collision != "full":
            continue
        elif body.name == "base":
            lo, hi = v.min(0), v.max(0)
            body.add_geom(name="base_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(lo + hi) / 2,
                          size=(hi - lo) / 2, **common)
        else:
            p0, p1, r = _capsule(v)
            body.add_geom(name=f"{body.name}_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=np.concatenate([p0, p1]), size=[r, 0, 0], **common)
    return spec


def parked_qpos():
    """qpos da keyframe "parked" das cenas (pose inicial: perna em 0.18 m, trem abaixado)."""
    flat = mujoco.MjModel.from_xml_path(str(FLAT_XML))
    k = [mujoco.mj_id2name(flat, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(flat.nkey)].index("parked")
    return flat.key_qpos[k].copy()


def build(terrain=None, timestep=0.001, collision="minimal"):
    """Robo + chao. `terrain` = None (plano) ou dict de terrain.make_terrain() (hfield compartilhado)."""
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
    ground = dict(contype=1, conaffinity=1, friction=[1.2, 0.02, 0.002], material="ground")
    if terrain is None:
        world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], **ground)
    else:
        h = terrain["heights"]
        zmax = max(float(h.max() - h.min()), 1e-3)
        spec.add_hfield(name="terrain", nrow=h.shape[0], ncol=h.shape[1],
                        size=[terrain["size_x"] / 2, terrain["size_y"] / 2, zmax, 0.1],
                        userdata=((h - h.min()) / zmax).ravel().tolist())
        world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                       pos=[0, 0, float(h.min())], **ground)
    return spec.compile(), spec
