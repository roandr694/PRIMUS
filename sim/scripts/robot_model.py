"""Shared helpers: model paths, loading at the "parked" keyframe, 5-bar leg pose, LQR model parameters.

Used by the simulation scripts and by control/lqr_design.py (which takes masses/inertias from the MJCF).
"""
import math
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (control/ on sys.path)
import five_bar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cena_arena.xml"        # arena (terrain + obstacles)
FLAT_MODEL_PATH = PROJECT_ROOT / "models" / "cena_plana.xml"
ROBOT_PATH = PROJECT_ROOT / "models" / "robo_bipedal.xml"

SIDES = ("xp", "xm")                 # xp = right (X+), xm = left (X-)
LEG_BODIES = ("armYp", "armYm", "shin", "linkYm")


def load(path=None, flat=False):
    """Loads the scene with the robot at keyframe 0, "parked" (legs at 0.18 m, landing gear down)."""
    import mujoco
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
# Leg kinematics (base at rest, base frame): pose for a leg length L0
# --------------------------------------------------------------------------------------
def _jadr(mujoco, model, name):
    return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def set_symmetric_hips(mujoco, model, data, a):
    """hip_Yp = +a, hip_Ym = -a on both legs; knees from the analytic closed 5-bar."""
    for s in SIDES:
        knee_p, knee_m, _ = five_bar.closure(s, a, -a)
        for joint, value in (("hip_Yp", a), ("hip_Ym", -a), ("knee_Yp", knee_p), ("knee_Ym", knee_m)):
            data.qpos[_jadr(mujoco, model, f"{joint}_{s}")] = value
    mujoco.mj_kinematics(model, data)


def leg_vector_base(mujoco, model, data, side):
    """Virtual hip -> wheel centre in the base frame, as (forward, up)."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = data.xmat[b].reshape(3, 3)
    h = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"hip_mid_{side}")]
    w = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{side}")]
    v = R.T @ (w - h)
    return float(v[1]), float(v[2])


def pose_for_leg_length(mujoco, model, data, L0):
    """Puts both legs at length L0 (symmetric hip angle, analytic). Returns the angle."""
    a = five_bar.symmetric_angle_for_length("xp", L0)
    set_symmetric_hips(mujoco, model, data, a)
    return a


def leg_lqr_params(mujoco, model, L0):
    """Planar model parameters (half robot = 1 leg + 1 wheel + half body) for the LQR at leg length L0.

    Returns L (wheel -> leg CoM), Lm (leg CoM -> hip), mp, Ip (leg, about its own CoM), l (body CoM above the
    virtual hip), M, Im (half body), mw, Iw (one wheel, including the rotor armature).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)          # unrotated base: base frame == world
    pose_for_leg_length(mujoco, model, data, L0)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    side = "xp"
    hip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"hip_mid_{side}")].copy()
    wc = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{side}")].copy()

    def lump(names):
        """mass, CoM (world) and inertia about X through the group CoM (only the YZ plane matters)."""
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
    # distance along the leg axis (wheel -> hip)
    axis = (hip - wc)[1:]
    Lreal = float(np.linalg.norm(axis))
    u = axis / Lreal
    L = float(np.dot((cp - wc)[1:], u))
    L = min(max(L, 0.02), Lreal - 0.02)
    # body = base + landing gear (half robot)
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
    """The robot starts parked: run the controller with a zero command until it has stood up and retracted the
    landing gear (mode "operating"), so scripts measure from there. Returns the elapsed time [s]."""
    from repo_control_policy import Mode, RobotCommand
    t = 0.0
    while controller.mode != Mode.OPERATING:
        if t > timeout:
            raise RuntimeError(f"the robot did not stand up within {timeout:.0f} s (mode {controller.mode})")
        controller.apply(data, RobotCommand())
        mujoco.mj_step(model, data)
        t += model.opt.timestep
    return t
