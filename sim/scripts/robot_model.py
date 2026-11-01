"""Utilitarios comuns: caminhos dos modelos, carga com a keyframe "parked" e cinematica da perna 5 barras.

Usado pelos scripts de simulacao e pelo lqr_design.py (que tira as massas/inercias direto do MJCF).
"""
import math
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (control/ no sys.path)
import five_bar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cena_arena.xml"        # arena (terreno + obstaculos)
FLAT_MODEL_PATH = PROJECT_ROOT / "models" / "cena_plana.xml"
ROBOT_PATH = PROJECT_ROOT / "models" / "robo_bipedal.xml"

SIDES = ("xp", "xm")                 # xp = direita (X+), xm = esquerda (X-)
SIDE_NAME = {"xp": "direita", "xm": "esquerda"}
LEG_BODIES = ("armYp", "armYm", "shin", "linkYm")


def require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError:
        print("MuJoCo nao esta instalado neste Python. Use o env conda:")
        print("  conda run -n bipedal python3 -m pip install -r requirements.txt")
        raise SystemExit(1)
    return mujoco


def load(path=None, flat=False):
    """Carrega a cena e coloca o robo na keyframe 0, "parked" (estacionado: perna em 0.18 m, trem abaixado)."""
    mujoco = require_mujoco()
    path = Path(path) if path else (FLAT_MODEL_PATH if flat else MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    reset(mujoco, model, data)
    return mujoco, model, data, path


def reset(mujoco, model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------------------
# Cinematica da perna (com a base parada, frame da base): pose para um comprimento L0
# --------------------------------------------------------------------------------------
def _jadr(mujoco, model, name):
    return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def solve_knees(mujoco, model, data, side, iters=30):
    """Ajusta os 2 joelhos passivos para fechar o laco (pe do link Y- == centro da roda na canela)."""
    kp, km = _jadr(mujoco, model, f"knee_Yp_{side}"), _jadr(mujoco, model, f"knee_Ym_{side}")
    s_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{side}")
    s_wc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{side}")

    def res():
        mujoco.mj_kinematics(model, data)
        return (data.site_xpos[s_foot] - data.site_xpos[s_wc])[1:]

    for _ in range(iters):
        r = res()
        if np.abs(r).max() < 1e-9:
            break
        J = np.zeros((2, 2))
        for c, adr in enumerate((kp, km)):
            data.qpos[adr] += 1e-6
            J[:, c] = (res() - r) / 1e-6
            data.qpos[adr] -= 1e-6
        dq = np.linalg.lstsq(J, -r, rcond=None)[0]
        data.qpos[kp] += dq[0]
        data.qpos[km] += dq[1]
    return float(np.abs(res()).max())


def set_symmetric_hips(mujoco, model, data, a):
    """hip_Yp = +a, hip_Ym = -a nas duas pernas (encolhe a perna simetricamente) e fecha os lacos.

    Os joelhos saem da solucao analitica (ramo com a roda para baixo); o Newton so refina."""
    for s in SIDES:
        data.qpos[_jadr(mujoco, model, f"hip_Yp_{s}")] = a
        data.qpos[_jadr(mujoco, model, f"hip_Ym_{s}")] = -a
        c = five_bar.closure(s, a, -a)
        if c is not None:
            data.qpos[_jadr(mujoco, model, f"knee_Yp_{s}")] = c[0]
            data.qpos[_jadr(mujoco, model, f"knee_Ym_{s}")] = c[1]
        solve_knees(mujoco, model, data, s)
    mujoco.mj_kinematics(model, data)


def leg_vector_base(mujoco, model, data, side):
    """Vetor quadril-virtual -> centro da roda no frame da base, como (frente, cima)."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = data.xmat[b].reshape(3, 3)
    h = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"hip_mid_{side}")]
    w = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{side}")]
    v = R.T @ (w - h)
    return float(v[1]), float(v[2])


def pose_for_leg_length(mujoco, model, data, L0):
    """Coloca as duas pernas no comprimento L0 (angulo simetrico dos quadris, analitico). Retorna o angulo."""
    a = five_bar.symmetric_angle_for_length("xp", L0)
    set_symmetric_hips(mujoco, model, data, a)
    return a


def leg_lqr_params(mujoco, model, L0):
    """Parametros do modelo planar (meio robo = 1 perna + 1 roda + meio corpo) para o LQR, no comprimento L0.

    Retorna L (roda -> CoM da perna), Lm (CoM da perna -> quadril), mp, Ip (perna, em torno do proprio CoM),
    l (CoM do corpo acima do quadril virtual), M, Im (meio corpo), mw, Iw (1 roda, inclui armature).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)          # base sem rotacao: frame da base == mundo
    pose_for_leg_length(mujoco, model, data, L0)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    side = "xp"
    hip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"hip_mid_{side}")].copy()
    wc = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{side}")].copy()

    def lump(names):
        """massa, CoM (mundo) e inercia em torno de X pelo CoM do conjunto (so plano YZ importa)."""
        ms, cs, Ixx = 0.0, np.zeros(3), 0.0
        parts = []
        for n in names:
            b = bid(n)
            mb = float(model.body_mass[b])
            com = data.xipos[b].copy()
            R = data.ximat[b].reshape(3, 3)
            Ib = R @ np.diag(model.body_inertia[b]) @ R.T
            parts.append((mb, com, Ib[0, 0]))
            ms += mb
            cs += mb * com
        cs /= ms
        for mb, com, ixx in parts:
            d = com - cs
            Ixx += ixx + mb * (d[1] ** 2 + d[2] ** 2)
        return ms, cs, Ixx

    mp, cp, Ip = lump([f"{n}_{side}" for n in LEG_BODIES])
    # distancia ao longo do eixo da perna (roda -> quadril)
    axis = (hip - wc)[1:]
    Lreal = float(np.linalg.norm(axis))
    u = axis / Lreal
    L = float(np.dot((cp - wc)[1:], u))
    L = min(max(L, 0.02), Lreal - 0.02)
    # corpo = base + trens (meio robo)
    Mb, cb, Ib = lump(["base", "gear_front", "gear_rear"])
    l = float(cb[2] - hip[2])
    wb = bid(f"wheel_{side}")
    jw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_{side}")
    Iw = float(model.body_inertia[wb][0]) + float(model.dof_armature[model.jnt_dofadr[jw]])
    return {
        "L": L, "Lm": Lreal - L, "mp": mp, "Ip": Ip,
        "M": Mb / 2.0, "Im": Ib / 2.0, "l": l,
        "mw": float(model.body_mass[wb]), "Iw": Iw,
        "R": float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{side}_tire")][0]),
        "body_com_forward_of_hip": float(cb[1] - hip[1]),
    }


def stand_up(mujoco, model, data, controller, timeout=8.0):
    """O robo nasce estacionado (keyframe "parked"). Roda o controlador parado ate ele levantar e
    recolher o trem (modo "operating"), para os scripts medirem a partir daqui. Devolve o tempo [s]."""
    from repo_control_policy import Mode, RobotCommand
    t = 0.0
    while controller.mode != Mode.OPERATING:
        if t > timeout:
            raise RuntimeError(f"o robo nao saiu do pouso em {timeout:.0f} s (modo {controller.mode})")
        controller.apply(data, RobotCommand())
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    return t
