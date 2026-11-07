"""LQR + VMC + PIDs + automatic landing gear in one Warp kernel (one thread per robot).

The same controller as control/repo_control_policy.py (state_source="robot"), ported line by line so it runs
inside the MuJoCo Warp CUDA graph at every physics step (1 kHz) for thousands of robots. Differences:
  * 5-bar leg kinematics and Jacobian are analytic (control/five_bar.py) instead of mj_jac;
  * attitude and rates come from the base qpos/qvel (the IMU site has the base orientation, which is what the
    simulated IMU "sensor" mode returns), without the one-step sensordata delay;
  * the RL residual enters at defined points (see RESIDUAL); with a zero residual it is the original controller.
Physical constants and gains are read from a RepoControlPolicy instance, so the numpy controller is the single
source of truth; tests/test_controller_parity.py compares both on the same state.

RESIDUAL (per robot, physical units, only in OPERATING mode):
  [dL_xp, dL_xm]   leg length offsets [m]: the mean adds to the height target; the difference becomes a body
                   roll reference (the roll PID is what makes the legs differ)
  [dth_xp, dth_xm] leg angle offsets [rad]: the mean shifts the LQR equilibrium (leans); the difference is the
                   reference of the leg synchronisation PID (one leg ahead of the other)
  [dT_xp, dT_xm]   torque added to each wheel [N.m at the joint]
  [dF_xp, dF_xm]   force added along each leg [N] (jumps, tucking the legs); in the air the length offset also
                   moves the retracted length
  [jump]           > 0.5 starts the jump primitive (push the legs out, then tuck them; RepoControlPolicy)

DRIVE (per robot, `drv`): joint torque limit [N.m] and joint no-load speed [rad/s]. With a no-load speed > 0 the
available torque drops linearly with the joint speed in the direction of motion (DC motor; a model of the plant,
not part of the controller). The default (joint limit of the model, no droop) is the numpy controller.
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

CONTROL = Path(__file__).resolve().parents[2] / "control"
sys.path.insert(0, str(CONTROL))
import five_bar  # noqa: E402
import repo_control_policy as rcp  # noqa: E402
from lqr_gains import LQR_COEFFS, L0_VALID_RANGE  # noqa: E402

SIDES = ("xp", "xm")

# ---- modes (same as repo_control_policy.Mode) ----
PARKED, STANDING_UP, OPERATING, LANDING = wp.constant(0), wp.constant(1), wp.constant(2), wp.constant(3)
MODE_NAMES = ("parked", "standing_up", "operating", "landing")

# ---- per-robot state: columns of `cs` ----
_STATE = ["mode", "gear_down", "gear_timer", "stable_time", "length_ref",
          "x", "dx", "target_position", "target_speed", "target_yaw", "theta_zero",
          "yaw_i", "len_i",                                    # yaw and leg-length PI integrals
          "L_xp", "L_xm", "ba_xp", "ba_xm",                    # leg length and angle (previous step)
          "tf_xp", "tf_xm", "air_xp", "air_xm",                # estimated normal force, time in the air
          "z1_xp", "z1_xm", "z2_xp", "z2_xm", "ddz_xp", "ddz_xm",
          "a_lat",                                             # filtered lateral (centripetal) acceleration
          "jump_phase", "jump_t", "jump_cool"]                 # jump primitive (0 idle, 1 push, 2 tuck)
S = {n: i for i, n in enumerate(_STATE)}
NSTATE = len(_STATE)
# ---- per-robot outputs: columns of `out` (for observations and rewards) ----
_OUT = ["theta", "d_theta", "pitch", "roll", "yaw", "pitch_rate", "roll_rate", "yaw_rate", "x", "dx",
        "L_xp", "L_xm", "th_xp", "th_xm", "dL_xp", "dL_xm", "dth_xp", "dth_xm",
        "grounded", "tf_xp", "tf_xm", "protect",
        "tau_hYp_xp", "tau_hYm_xp", "tau_hYp_xm", "tau_hYm_xm", "tau_w_xp", "tau_w_xm", "yaw_limit", "jump_phase"]
O = {n: i for i, n in enumerate(_OUT)}
NOUT = len(_OUT)
# command: speed [m/s], yaw rate [rad/s], leg length [m], roll [rad], park (0/1), max accel [m/s^2]
NCMD = 6
NRES = 9

for _n, _i in list(S.items()):
    globals()[f"S_{_n.upper()}"] = wp.constant(_i)
for _n, _i in list(O.items()):
    globals()[f"O_{_n.upper()}"] = wp.constant(_i)


@wp.struct
class Params:
    dt: float
    # leg geometry in the base (forward, up) plane [m]; the same for both legs
    hip_p: wp.vec2
    hip_m: wp.vec2
    arm_p: wp.vec2      # knee Y+ - hip Y+ in the CAD pose
    arm_m: wp.vec2
    r_p: float          # lower bar Y+ length (knee -> axle)
    r_m: float
    hip_mid: wp.vec2
    hip_spacing: float  # lateral distance between the virtual hips of the two legs [m]
    # addresses
    qp_hipYp: wp.vec2i  # qpos of hip_Yp (xp, xm)
    qp_hipYm: wp.vec2i
    qv_hipYp: wp.vec2i
    qv_hipYm: wp.vec2i
    qv_kneeYp: wp.vec2i
    qv_wheel: wp.vec2i
    qp_gear: wp.vec2i
    sd_acc: int         # accelerometer in sensordata (3)
    u_hipYp: wp.vec2i   # ctrl indices
    u_hipYm: wp.vec2i
    u_wheel: wp.vec2i
    u_servo: wp.vec2i   # (front, rear)
    # physics
    wheel_radius: float
    below_hip_mass: float
    force_bias: float
    gear_ratio: float
    joint_limit: float
    gear_down_angle: wp.vec2
    gear_up_angle: wp.vec2
    # limits and mode
    min_len: float
    max_len: float
    park_len: float
    gear_safe_len: float
    l0_lo: float
    l0_hi: float
    gear_travel_s: float
    stand_rate: float
    land_rate: float
    park_wheel_damping: float
    park_leg_kp: float
    park_leg_kd: float
    # odometry / targets
    accel_step: float
    nominal_len: float
    pos_lead: float
    speed_lead: float
    max_yaw_lead: float
    theta_zero_ki: float
    theta_zero_max: float
    air_retract_length: float
    contact_min_force: float
    contact_lpf: float
    air_grace_s: float
    # turning: load-transfer feed-forward and turn-rate envelope
    robot_mass: float
    wheel_track_half: float
    com_a: float
    com_b: float
    load_transfer_gain: float
    inner_load_min: float
    tip_height_factor: float
    a_lat_lpf: float
    # jump primitive
    jump_force: float
    jump_ext_len: float
    jump_push_max_s: float
    jump_tuck_force: float
    jump_tuck_len: float
    jump_tuck_max_s: float
    jump_cooldown_s: float


def make_params(model, dt=None):
    """Params from the MjModel, with the constants of the numpy controller (single source of truth)."""
    ref = rcp.RepoControlPolicy(mujoco, model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    qp = lambda n: int(model.jnt_qposadr[jid(n)])
    qv = lambda n: int(model.jnt_dofadr[jid(n)])
    pair = lambda f, a, b=None: wp.vec2i(f(a.format("xp")), f(a.format("xm"))) if b is None else wp.vec2i(f(a), f(b))
    P = five_bar.leg_points("xp")
    sub = lambda a, b: wp.vec2(P[a][0] - P[b][0], P[a][1] - P[b][1])
    L = five_bar.LEGS
    p = Params()
    p.dt = float(dt if dt is not None else model.opt.timestep)
    p.hip_p, p.hip_m, p.hip_mid = wp.vec2(*P["hipYp"]), wp.vec2(*P["hipYm"]), wp.vec2(*five_bar.hip_mid("xp"))
    p.arm_p, p.arm_m = sub("kneeYp", "hipYp"), sub("kneeYm", "hipYm")
    p.r_p, p.r_m = wp.length(sub("foot", "kneeYp")), wp.length(sub("foot", "kneeYm"))
    p.qp_hipYp, p.qp_hipYm = pair(qp, "hip_Yp_{}"), pair(qp, "hip_Ym_{}")
    p.qv_hipYp, p.qv_hipYm, p.qv_kneeYp = pair(qv, "hip_Yp_{}"), pair(qv, "hip_Ym_{}"), pair(qv, "knee_Yp_{}")
    p.qv_wheel, p.qp_gear = pair(qv, "wheel_{}"), pair(qp, "gear_front", "gear_rear")
    p.sd_acc = ref.imu.adr_a
    p.u_hipYp = wp.vec2i(ref.act["hip_Yp_xp"], ref.act["hip_Yp_xm"])
    p.u_hipYm = wp.vec2i(ref.act["hip_Ym_xp"], ref.act["hip_Ym_xm"])
    p.u_wheel = wp.vec2i(ref.act["wheel_xp"], ref.act["wheel_xm"])
    p.u_servo = wp.vec2i(ref.servo["gear_front"], ref.servo["gear_rear"])
    p.wheel_radius, p.below_hip_mass, p.force_bias = ref.wheel_radius, ref.below_hip_mass, ref.force_bias_n
    p.gear_ratio, p.joint_limit = ref.gear["hip_Yp_xp"], ref.joint_limit["hip_Yp_xp"]
    p.gear_down_angle = wp.vec2(ref.gear_down_angle["gear_front"], ref.gear_down_angle["gear_rear"])
    p.gear_up_angle = wp.vec2(ref.gear_up_angle["gear_front"], ref.gear_up_angle["gear_rear"])
    p.min_len, p.max_len, p.park_len, p.gear_safe_len = rcp.MIN_LEG_LENGTH, rcp.MAX_LEG_LENGTH, rcp.PARK_LEG_LENGTH, rcp.GEAR_SAFE_LENGTH
    p.l0_lo, p.l0_hi = L0_VALID_RANGE
    p.nominal_len = rcp.NOMINAL_LEG_LENGTH
    for name in ("gear_travel_s", "stand_rate", "land_rate", "park_wheel_damping", "park_leg_kp", "park_leg_kd",
                 "accel_step", "pos_lead", "speed_lead", "max_yaw_lead", "theta_zero_ki", "theta_zero_max",
                 "air_retract_length", "contact_lpf", "air_grace_s"):
        setattr(p, name, float(getattr(ref, name)))
    p.contact_min_force = ref.contact_min_force_n
    p.robot_mass, p.wheel_track_half = ref.robot_mass, ref.wheel_track_half
    p.com_a, p.com_b = ref.com_height_fit
    p.hip_spacing = ref.hip_spacing
    for name in ("load_transfer_gain", "inner_load_min", "tip_height_factor", "a_lat_lpf"):
        setattr(p, name, float(getattr(ref, name)))
    p.jump_force, p.jump_ext_len, p.jump_push_max_s = ref.jump_force, ref.jump_ext_length, ref.jump_push_max_s
    p.jump_tuck_force, p.jump_tuck_len, p.jump_tuck_max_s = ref.jump_tuck_force, ref.jump_tuck_length, ref.jump_tuck_max_s
    p.jump_cooldown_s = ref.jump_cooldown_s
    return p


def lqr_coeffs(device):
    return wp.array(np.array(LQR_COEFFS, dtype=np.float32).reshape(12, 4), dtype=wp.vec4, device=device)


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
@wp.func
def rot2(v: wp.vec2, a: float):
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def perp(v: wp.vec2):
    """d rot2(v, a) / da: v rotated by +90 deg"""
    return wp.vec2(-v[1], v[0])


@wp.func
def wrap(a: float):
    return wp.atan2(wp.sin(a), wp.cos(a))


@wp.func
def knee_p(prm: Params, hp: float):
    return prm.hip_p + rot2(prm.arm_p, hp)


@wp.func
def knee_m(prm: Params, hm: float):
    return prm.hip_m + rot2(prm.arm_m, hm)


@wp.func
def foot(prm: Params, kp: wp.vec2, km: wp.vec2):
    """wheel axle: intersection of the circles (radii r_p, r_m) around the knees, lower branch"""
    dlt = km - kp
    d = wp.max(wp.length(dlt), 1.0e-6)
    along = (prm.r_p * prm.r_p - prm.r_m * prm.r_m + d * d) / (2.0 * d)
    h = wp.sqrt(wp.max(prm.r_p * prm.r_p - along * along, 0.0))
    c = kp + dlt * (along / d)
    n = wp.vec2(-dlt[1], dlt[0]) / d
    f1 = c + n * h
    f2 = c - n * h
    if f1[1] < f2[1]:
        return f1
    return f2


@wp.func
def leg_jacobian(prm: Params, hp: float, hm: float, kp: wp.vec2, km: wp.vec2, f: wp.vec2):
    """d(wheel axle)/d(hip_Yp, hip_Ym) in the base plane, knees following the closed loop"""
    a = f - kp
    b = f - km
    A = wp.mat22(a[0], a[1], b[0], b[1])
    Ai = wp.inverse(A)
    cp = wp.dot(a, perp(kp - prm.hip_p))
    cm = wp.dot(b, perp(km - prm.hip_m))
    jp = Ai * wp.vec2(cp, 0.0)
    jm = Ai * wp.vec2(0.0, cm)
    return wp.mat22(jp[0], jm[0], jp[1], jm[1])


@wp.func
def lqr_gain(coeffs: wp.array(dtype=wp.vec4), k: int, l0: float):
    c = coeffs[k]
    return ((c[0] * l0 + c[1]) * l0 + c[2]) * l0 + c[3]


@wp.func
def quat_rot(q: wp.quat):
    return wp.quat_to_matrix(q)


@wp.func
def drive_clamp(tau: float, speed: float, lim: float, w0: float):
    """torque limit with a linear torque-speed droop in the direction of motion (w0 <= 0: no droop)"""
    l = lim
    if w0 > 0.0 and tau * speed > 0.0:
        l = lim * wp.max(0.0, 1.0 - wp.abs(speed) / w0)
    return wp.clamp(tau, -l, l)


@wp.func
def pid_step(err: float, kp: float, ki: float, imax: float, omax: float, integ: float):
    """reference-firmware PI: returns (output, new integral)"""
    integ2 = wp.clamp(integ + err * ki, -imax, imax)
    return wp.clamp(err * kp + integ2, -omax, omax), integ2


# ------------------------------------------------------------------------------------------
# kernels
# ------------------------------------------------------------------------------------------
@wp.kernel
def reset_kernel(prm: Params, mask: wp.array(dtype=int), qpos: wp.array2d(dtype=float),
                 cs: wp.array2d(dtype=float)):
    """Initial controller state from the pose (as RepoControlPolicy.reset)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    for i in range(cs.shape[1]):
        cs[w, i] = 0.0
    q = wp.quat(qpos[w, 4], qpos[w, 5], qpos[w, 6], qpos[w, 3])
    R = quat_rot(q)
    pitch = wp.asin(wp.clamp(R[2, 1], -1.0, 1.0))
    yaw = wp.atan2(-R[0, 1], R[1, 1])
    avg = float(0.0)
    for s in range(2):
        hp = qpos[w, prm.qp_hipYp[s]]
        hm = qpos[w, prm.qp_hipYm[s]]
        kp = knee_p(prm, hp)
        km = knee_m(prm, hm)
        v = foot(prm, kp, km) - prm.hip_mid
        L = wp.length(v)
        cs[w, S_L_XP + s] = L
        cs[w, S_BA_XP + s] = wp.atan2(-v[0], -v[1])
        cs[w, S_TF_XP + s] = prm.force_bias
        zrel = R[2, 1] * v[0] + R[2, 2] * v[1]
        cs[w, S_Z1_XP + s] = zrel
        cs[w, S_Z2_XP + s] = zrel
        avg += 0.5 * L
    gear_down = int(1)
    for g in range(2):
        if wp.abs(qpos[w, prm.qp_gear[g]] - prm.gear_down_angle[g]) >= 0.3:
            gear_down = 0
    mode = OPERATING
    if gear_down == 1 and avg <= prm.park_len + 0.02:
        mode = PARKED
    elif gear_down == 1:
        mode = STANDING_UP
    cs[w, S_MODE] = float(mode)
    cs[w, S_GEAR_DOWN] = float(gear_down)
    cs[w, S_GEAR_TIMER] = prm.gear_travel_s
    cs[w, S_LENGTH_REF] = wp.clamp(avg, prm.min_len, prm.max_len)
    cs[w, S_TARGET_YAW] = yaw


@wp.kernel
def control_kernel(prm: Params, coeffs: wp.array(dtype=wp.vec4),
                   qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                   sensordata: wp.array2d(dtype=float), qfrc_actuator: wp.array2d(dtype=float),
                   cmd: wp.array2d(dtype=float), res: wp.array2d(dtype=float), drv: wp.array2d(dtype=float),
                   cs: wp.array2d(dtype=float), out: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    w = wp.tid()
    dt = prm.dt
    # ---------------- state: IMU (base orientation) + encoders ----------------
    q = wp.quat(qpos[w, 4], qpos[w, 5], qpos[w, 6], qpos[w, 3])
    R = quat_rot(q)
    pitch = wp.asin(wp.clamp(R[2, 1], -1.0, 1.0))
    roll = wp.atan2(-R[2, 0], R[2, 2])
    yaw = wp.atan2(-R[0, 1], R[1, 1])
    pitch_rate = qvel[w, 3]
    roll_rate = qvel[w, 4]
    yaw_rate = qvel[w, 5]

    Ls = wp.vec2(0.0, 0.0)
    dLs = wp.vec2(0.0, 0.0)
    ths = wp.vec2(0.0, 0.0)
    dths = wp.vec2(0.0, 0.0)
    bas = wp.vec2(0.0, 0.0)
    dbas = wp.vec2(0.0, 0.0)
    fwds = wp.vec2(0.0, 0.0)
    ups = wp.vec2(0.0, 0.0)
    J0 = wp.mat22(0.0, 0.0, 0.0, 0.0)
    J1 = wp.mat22(0.0, 0.0, 0.0, 0.0)
    for s in range(2):
        hp = qpos[w, prm.qp_hipYp[s]]
        hm = qpos[w, prm.qp_hipYm[s]]
        kp = knee_p(prm, hp)
        km = knee_m(prm, hm)
        f = foot(prm, kp, km)
        v = f - prm.hip_mid
        L = wp.max(wp.length(v), 1.0e-6)
        ba = wp.atan2(-v[0], -v[1])
        dL = (L - cs[w, S_L_XP + s]) / dt
        dba = wrap(ba - cs[w, S_BA_XP + s]) / dt
        cs[w, S_L_XP + s] = L
        cs[w, S_BA_XP + s] = ba
        Ls[s] = L
        dLs[s] = dL
        bas[s] = ba
        dbas[s] = dba
        ths[s] = ba - pitch
        dths[s] = dba - pitch_rate
        fwds[s] = v[0]
        ups[s] = v[1]
        J = leg_jacobian(prm, hp, hm, kp, km, f)
        if s == 0:
            J0 = J
        else:
            J1 = J

    # odometry: wheel encoder (relative to the shin) + shin rotation = IMU pitch rate + hip + knee
    vsum = float(0.0)
    for s in range(2):
        w_shin = pitch_rate + qvel[w, prm.qv_hipYp[s]] + qvel[w, prm.qv_kneeYp[s]]
        vsum += -prm.wheel_radius * (qvel[w, prm.qv_wheel[s]] + w_shin)
    dx = vsum / 2.0
    x = cs[w, S_X] + dx * dt
    cs[w, S_DX] = dx
    cs[w, S_X] = x

    # contact: normal force estimated from the hip torques (as in the reference firmware)
    acc_z = R[2, 0] * sensordata[w, prm.sd_acc] + R[2, 1] * sensordata[w, prm.sd_acc + 1] \
        + R[2, 2] * sensordata[w, prm.sd_acc + 2] - 9.81
    grounded = int(1)
    for s in range(2):
        J = J0
        if s == 1:
            J = J1
        tau = wp.vec2(qfrc_actuator[w, prm.qv_hipYp[s]], qfrc_actuator[w, prm.qv_hipYm[s]])
        ff = wp.inverse(wp.transpose(J)) * tau            # leg force on the wheel (forward, up)
        fz = R[2, 1] * ff[0] + R[2, 2] * ff[1]
        zrel = R[2, 1] * fwds[s] + R[2, 2] * ups[s]
        ddz = (zrel - 2.0 * cs[w, S_Z1_XP + s] + cs[w, S_Z2_XP + s]) / (dt * dt)
        cs[w, S_Z2_XP + s] = cs[w, S_Z1_XP + s]
        cs[w, S_Z1_XP + s] = zrel
        ddz_f = cs[w, S_DDZ_XP + s] + 0.05 * (ddz - cs[w, S_DDZ_XP + s])
        cs[w, S_DDZ_XP + s] = ddz_f
        n = -fz + prm.below_hip_mass * (9.81 + acc_z + ddz_f)
        tf = cs[w, S_TF_XP + s] + prm.contact_lpf * (n - cs[w, S_TF_XP + s])
        cs[w, S_TF_XP + s] = tf
        air = cs[w, S_AIR_XP + s] + dt
        if tf > prm.contact_min_force:
            air = 0.0
        cs[w, S_AIR_XP + s] = air
        if air > prm.air_grace_s:
            grounded = 0

    avg_len = 0.5 * (Ls[0] + Ls[1])

    # ---------------- automatic mode (landing / gear) ----------------
    park = cmd[w, 4] > 0.5
    mode = int(cs[w, S_MODE])
    gear_down = cs[w, S_GEAR_DOWN] > 0.5
    stable = wp.abs(pitch) < 0.25 and wp.abs(pitch_rate) < 0.3 and wp.abs(cs[w, S_DX]) < 0.1
    stable_time = float(0.0)
    if stable:
        stable_time = cs[w, S_STABLE_TIME] + dt
    gear_timer = cs[w, S_GEAR_TIMER] + dt
    gear_settled = gear_timer >= prm.gear_travel_s
    safe = prm.gear_safe_len + 0.01
    wanted = wp.clamp(cmd[w, 2], prm.min_len, prm.max_len)
    length_ref = cs[w, S_LENGTH_REF]

    if mode == PARKED and not park:
        mode = STANDING_UP
    elif (mode == STANDING_UP or mode == OPERATING) and park:
        mode = LANDING
    elif mode == LANDING and not park:
        mode = STANDING_UP

    if mode == STANDING_UP:
        if gear_down and avg_len >= prm.gear_safe_len and stable_time >= 0.5:
            gear_down = False
            gear_timer = 0.0
        elif (not gear_down) and gear_settled:
            mode = OPERATING
    elif mode == LANDING:
        if (not gear_down) and avg_len >= prm.gear_safe_len - 0.005 and stable_time >= 0.3:
            gear_down = True
            gear_timer = 0.0
        elif gear_down and gear_settled and avg_len <= prm.park_len + 0.004:
            mode = PARKED

    operating = mode == OPERATING
    r_dl = wp.vec2(0.0, 0.0)
    r_dth = wp.vec2(0.0, 0.0)
    r_dT = wp.vec2(0.0, 0.0)
    r_dF = wp.vec2(0.0, 0.0)
    trigger = False
    if operating:
        trigger = res[w, 8] > 0.5
        r_dl = wp.vec2(res[w, 0], res[w, 1])
        r_dth = wp.vec2(res[w, 2], res[w, 3])
        r_dT = wp.vec2(res[w, 4], res[w, 5])
        r_dF = wp.vec2(res[w, 6], res[w, 7])

    if mode == PARKED:
        length_ref = prm.park_len
    elif operating:
        length_ref = wp.clamp(wanted + 0.5 * (r_dl[0] + r_dl[1]), prm.min_len, prm.max_len)
    else:
        target = float(0.0)
        rate = prm.stand_rate
        if mode == LANDING and gear_down and gear_settled:
            target = prm.park_len
            if length_ref <= safe:
                rate = prm.land_rate
        elif mode == STANDING_UP and not gear_down:
            target = wanted
        elif mode == STANDING_UP:
            target = wp.max(wanted, safe)
        else:
            target = wp.max(length_ref, safe)
        step = rate * dt
        length_ref += wp.clamp(target - length_ref, -step, step)
    cs[w, S_MODE] = float(mode)
    cs[w, S_GEAR_DOWN] = wp.where(gear_down, 1.0, 0.0)
    cs[w, S_GEAR_TIMER] = gear_timer
    cs[w, S_STABLE_TIME] = stable_time
    cs[w, S_LENGTH_REF] = length_ref

    speed_cmd = float(0.0)
    yaw_cmd = float(0.0)
    if operating:
        speed_cmd = cmd[w, 0]
        yaw_cmd = cmd[w, 1]
    balancing = mode != PARKED
    # turn-rate envelope: |v * yaw_rate| <= a_max(leg length) (RepoControlPolicy.apply)
    h_com = prm.com_a * avg_len + prm.com_b
    v_ref = wp.max(wp.max(wp.abs(dx), wp.abs(speed_cmd)), 0.3)
    a_max = (0.5 - prm.inner_load_min) * 9.81 * (2.0 * prm.wheel_track_half) / (prm.tip_height_factor * h_com)
    yaw_limit = a_max / v_ref
    yaw_cmd = wp.clamp(yaw_cmd, -yaw_limit, yaw_limit)

    # ---------------- targets (RepoControlPolicy._update_target) ----------------
    target_position = cs[w, S_TARGET_POSITION]
    target_speed = cs[w, S_TARGET_SPEED]
    target_yaw = cs[w, S_TARGET_YAW]
    theta_zero = cs[w, S_THETA_ZERO]
    if balancing:
        step_ratio = dt / 0.004
        scale = cmd[w, 5] / 1.5
        slope = wp.max(0.003 * scale * step_ratio,
                       (-(avg_len - prm.nominal_len) * 0.02 + prm.accel_step) * scale * step_ratio)
        err = speed_cmd - target_speed
        if wp.abs(err) < slope:
            target_speed = speed_cmd
        else:
            target_speed = target_speed + wp.sign(err) * slope
        target_position = target_position + target_speed * dt
        target_position = wp.clamp(target_position, x - prm.pos_lead, x + prm.pos_lead)
        lead = prm.speed_lead * wp.max(1.0, 0.75 * scale)
        target_speed = wp.clamp(target_speed, dx - lead, dx + lead)
        if wp.abs(yaw_cmd) > 1.0e-3:
            target_yaw = yaw
        else:
            target_yaw = yaw + wp.clamp(wrap(target_yaw - yaw), -prm.max_yaw_lead, prm.max_yaw_lead)
    else:
        target_position = x
        target_speed = 0.0
        target_yaw = yaw
        theta_zero = 0.0
    if balancing and grounded == 1:
        theta_zero = wp.clamp(theta_zero + prm.theta_zero_ki * (x - target_position) * dt,
                              -prm.theta_zero_max, prm.theta_zero_max)
    cs[w, S_TARGET_POSITION] = target_position
    cs[w, S_TARGET_SPEED] = target_speed
    cs[w, S_TARGET_YAW] = target_yaw
    cs[w, S_THETA_ZERO] = theta_zero

    # ---------------- LQR ----------------
    theta = 0.5 * (ths[0] + ths[1]) - theta_zero - 0.5 * (r_dth[0] + r_dth[1])
    d_theta = 0.5 * (dths[0] + dths[1])
    st0 = theta
    st1 = d_theta
    st2 = x - target_position
    st3 = dx - target_speed
    st4 = pitch
    st5 = pitch_rate
    l0 = wp.clamp(avg_len, prm.l0_lo, prm.l0_hi)
    T = -(lqr_gain(coeffs, 0, l0) * st0 + lqr_gain(coeffs, 1, l0) * st1 + lqr_gain(coeffs, 2, l0) * st2
          + lqr_gain(coeffs, 3, l0) * st3 + lqr_gain(coeffs, 4, l0) * st4 + lqr_gain(coeffs, 5, l0) * st5)
    Tp = -(lqr_gain(coeffs, 6, l0) * st0 + lqr_gain(coeffs, 7, l0) * st1 + lqr_gain(coeffs, 8, l0) * st2
           + lqr_gain(coeffs, 9, l0) * st3 + lqr_gain(coeffs, 10, l0) * st4 + lqr_gain(coeffs, 11, l0) * st5)

    # yaw (wheel differential): P on heading drift (outer), PI on yaw rate (inner)
    yaw_sp = yaw_cmd + wp.clamp(1.8 * wrap(target_yaw - yaw), -2.0, 2.0)
    yaw_out, yaw_i = pid_step(yaw_sp - yaw_rate, 3.0, 0.004, 2.0, 6.0, cs[w, S_YAW_I])
    cs[w, S_YAW_I] = yaw_i
    T_side = wp.vec2(0.0, 0.0)
    if not balancing:
        T_side = wp.vec2(-prm.park_wheel_damping * dx, -prm.park_wheel_damping * dx)
    elif grounded == 1:
        T_side = wp.vec2(T + yaw_out, T - yaw_out) + r_dT

    # force along the leg: length + weight + roll
    length_target = prm.air_retract_length + 0.5 * (r_dl[0] + r_dl[1])
    if grounded == 1 or not balancing:
        length_target = wp.clamp(length_ref, prm.min_len, prm.max_len)
    # jump primitive: idle -> push (legs out at full force) -> tuck (legs in) -> idle (after a cooldown)
    jphase = int(cs[w, S_JUMP_PHASE])
    jt = cs[w, S_JUMP_T]
    jcool = cs[w, S_JUMP_COOL]
    if not operating:
        jphase = 0
    elif jphase == 0:
        jcool += dt
        if trigger and grounded == 1 and jcool >= prm.jump_cooldown_s and avg_len < prm.jump_ext_len - 0.05:
            jphase = 1
            jt = 0.0
    elif jphase == 1:
        jt += dt
        if avg_len >= prm.jump_ext_len or jt >= prm.jump_push_max_s:
            jphase = 2
            jt = 0.0
    else:
        jt += dt
        if (avg_len <= prm.jump_tuck_len + 0.02 and jt > 0.05) or jt >= prm.jump_tuck_max_s:
            jphase = 0
            jcool = 0.0
    cs[w, S_JUMP_PHASE] = float(jphase)
    cs[w, S_JUMP_T] = jt
    cs[w, S_JUMP_COOL] = jcool
    jump_f = float(0.0)
    if jphase == 1:
        length_target = prm.max_len
        jump_f = prm.jump_force
    elif jphase == 2:
        length_target = prm.jump_tuck_len
        jump_f = -prm.jump_tuck_force
    half = 0.5 * wp.abs(Ls[0] - Ls[1])     # keep both legs inside [min, max] on uneven ground
    length_target = wp.clamp(length_target, prm.min_len + half, prm.max_len - half)
    avg_dlen = 0.5 * (dLs[0] + dLs[1])
    len_sp = wp.clamp(12.0 * (length_target - avg_len), -1.0, 1.0)
    f_len, len_i = pid_step(len_sp - avg_dlen, 250.0, 0.2, 30.0, 140.0, cs[w, S_LEN_I])
    cs[w, S_LEN_I] = len_i
    roll_ref = cmd[w, 3] + wp.atan2(r_dl[1] - r_dl[0], prm.hip_spacing)
    roll_sp = wp.clamp(12.0 * (roll_ref - roll), -3.0, 3.0)
    roll_out = wp.clamp(25.0 * (roll_sp - roll_rate), -50.0, 50.0)
    bias = float(0.0)
    if grounded == 1 or not balancing:
        bias = prm.force_bias
    a_raw = float(0.0)
    if (grounded == 1 or not balancing) and balancing:
        a_raw = dx * yaw_rate
    a_lat = cs[w, S_A_LAT] + prm.a_lat_lpf * (a_raw - cs[w, S_A_LAT])
    cs[w, S_A_LAT] = a_lat
    f_lt = prm.load_transfer_gain * prm.robot_mass * a_lat * h_com / prm.hip_spacing    # centripetal load transfer
    F = wp.vec2(f_len + bias - roll_out + f_lt + jump_f, f_len + bias + roll_out - f_lt + jump_f) + r_dF
    for s in range(2):
        if Ls[s] > prm.max_len:
            F[s] = F[s] - (Ls[s] - prm.max_len) * 2000.0

    # virtual leg torque: LQR + leg synchronisation
    la_ref = r_dth[1] - r_dth[0]
    la_sp = wp.clamp(40.0 * (la_ref - (ths[1] - ths[0])), -8.0, 8.0)
    k_sync = (avg_len / prm.nominal_len) * (avg_len / prm.nominal_len)     # ~ leg inertia about the hip
    la = wp.clamp(4.0 * k_sync * (la_sp - (dths[1] - dths[0])), -25.0 * k_sync, 25.0 * k_sync)
    Tp_side = wp.vec2(Tp - la, Tp + la)          # (xp, xm)
    if not balancing:
        Tp_side = wp.vec2(-prm.park_leg_kp * bas[0] - prm.park_leg_kd * dbas[0],
                          -prm.park_leg_kp * bas[1] - prm.park_leg_kd * dbas[1])

    protect = wp.abs(theta) > 1.0471975512 or wp.abs(pitch) > 1.0471975512
    lim = drv[w, 0]
    w0 = drv[w, 1]
    for s in range(2):
        J = J0
        if s == 1:
            J = J1
        L = Ls[s]
        e = wp.vec2(fwds[s], ups[s]) / L
        pp = wp.vec2(e[1], -e[0])
        fvec = e * F[s] + pp * (Tp_side[s] / L)
        th = wp.transpose(J) * fvec
        tw = -T_side[s]
        if protect:
            th = wp.vec2(0.0, 0.0)
            tw = 0.0
        t_hp = drive_clamp(th[0], qvel[w, prm.qv_hipYp[s]], lim, w0)
        t_hm = drive_clamp(th[1], qvel[w, prm.qv_hipYm[s]], lim, w0)
        tw = drive_clamp(tw, qvel[w, prm.qv_wheel[s]], lim, w0)
        ctrl[w, prm.u_hipYp[s]] = t_hp / prm.gear_ratio
        ctrl[w, prm.u_hipYm[s]] = t_hm / prm.gear_ratio
        ctrl[w, prm.u_wheel[s]] = tw / prm.gear_ratio
        out[w, O_TAU_HYP_XP + 2 * s] = t_hp
        out[w, O_TAU_HYM_XP + 2 * s] = t_hm
        out[w, O_TAU_W_XP + s] = tw
    for g in range(2):
        if gear_down:
            ctrl[w, prm.u_servo[g]] = prm.gear_down_angle[g]
        else:
            ctrl[w, prm.u_servo[g]] = prm.gear_up_angle[g]

    out[w, O_THETA] = theta
    out[w, O_D_THETA] = d_theta
    out[w, O_PITCH] = pitch
    out[w, O_ROLL] = roll
    out[w, O_YAW] = yaw
    out[w, O_PITCH_RATE] = pitch_rate
    out[w, O_ROLL_RATE] = roll_rate
    out[w, O_YAW_RATE] = yaw_rate
    out[w, O_X] = x
    out[w, O_DX] = dx
    for s in range(2):
        out[w, O_L_XP + s] = Ls[s]
        out[w, O_TH_XP + s] = ths[s]
        out[w, O_DL_XP + s] = dLs[s]
        out[w, O_DTH_XP + s] = dths[s]
        out[w, O_TF_XP + s] = cs[w, S_TF_XP + s]
    out[w, O_GROUNDED] = float(grounded)
    out[w, O_YAW_LIMIT] = yaw_limit
    out[w, O_JUMP_PHASE] = float(jphase)
    out[w, O_PROTECT] = wp.where(protect, 1.0, 0.0)


class BatchedController:
    """Wraps the kernels; GPU arrays shared with MuJoCo Warp (mjw.Data)."""

    def __init__(self, mjm, nworld, device):
        self.device = device
        self.nworld = nworld
        self.prm = make_params(mjm)
        self.coeffs = lqr_coeffs(device)
        self.cs = wp.zeros((nworld, NSTATE), dtype=float, device=device)
        self.out = wp.zeros((nworld, NOUT), dtype=float, device=device)
        self.cmd = wp.zeros((nworld, NCMD), dtype=float, device=device)
        self.res = wp.zeros((nworld, NRES), dtype=float, device=device)
        self.mask = wp.zeros(nworld, dtype=int, device=device)
        # per-robot drive: (joint torque limit [N.m], no-load joint speed [rad/s], 0 = no droop)
        self.drv = wp.array(np.tile([[self.prm.joint_limit, 0.0]], (nworld, 1)).astype(np.float32), device=device)

    def reset(self, d, mask=None):
        """mask: int array (nworld), 1 on the robots to reset; None = all."""
        m = self.mask if mask is None else mask
        if mask is None:
            m.fill_(1)
        wp.launch(reset_kernel, dim=self.nworld, inputs=[self.prm, m, d.qpos, self.cs], device=self.device)

    def step(self, d):
        wp.launch(control_kernel, dim=self.nworld,
                  inputs=[self.prm, self.coeffs, d.qpos, d.qvel, d.sensordata, d.qfrc_actuator,
                          self.cmd, self.res, self.drv, self.cs, self.out, d.ctrl], device=self.device)
