"""Balance and motion controller: LQR + VMC + cascaded PIDs + automatic landing gear.

Structure from Skythinker616/foc-wheel-legged-robot, re-derived for this robot (CAD model, closed 5-bar legs,
9:1 actuators). Shared by the simulation and the real robot.

State (state_source="robot", default) uses only what the real robot measures:
  * attitude and rates from the IMU (simulated WitMotion; optional noise/bias, or Mahony from gyro + accel);
  * forward position/speed from wheel encoders, corrected by the shin rotation (IMU + hip kinematics);
  * ground contact from the normal force estimated with the hip motor torques.
state_source="sim" uses the simulator's ground truth instead (reference, or privileged critic for RL).

Conventions (robot frame: X right, Y forward, Z up). The LQR/VMC compute JOINT torques; actuator ctrl is
MOTOR torque = joint / 9.
    x, dx   wheel forward position/speed [m, m/s]
    theta   leg angle (wheel -> virtual hip) from vertical; + = hip ahead of the wheel
    phi     body pitch; + = nose up (rotation about +X)
    T       torque of ONE wheel; + = rolls forward (negative joint torque)
    Tp      virtual torque of ONE leg at the hip; + = increases theta (pushes the body nose-up)
    roll    rotation about +Y; + = right side (xp) down;   yaw about +Z; + = turn left
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from lqr_gains import LQR_COEFFS, L0_VALID_RANGE

NOMINAL_LEG_LENGTH = 0.27      # [m] virtual hip -> wheel axle (CAD pose = 0.394 m)
# Geometric range (collision meshes): the wheel hits the chassis below ~0.17 m; fully extended (singular)
# at 0.565 m. The operating range keeps margin at both ends.
MIN_LEG_LENGTH = 0.18          # close to wheel/chassis contact; also the parking height
MAX_LEG_LENGTH = 0.52          # far from the singularity (angle between the lower bars ~47 deg)
PARK_LEG_LENGTH = MIN_LEG_LENGTH   # with the gear down, its tips touch flat ground at this height
GEAR_SAFE_LENGTH = 0.24        # above this, the lowered gear is >= ~5 cm above flat ground
SIDES = ("xp", "xm")           # xp = right, xm = left


class Mode:
    """Automatic mode. The user never moves the landing gear: they only ask to operate or to land
    (RobotCommand.park) and the controller lowers/retracts the gear in the transitions."""
    PARKED = "parked"              # legs at PARK_LEG_LENGTH, gear down resting on the ground, no balancing
    STANDING_UP = "standing_up"    # balancing, raising the legs; retracts the gear once still
    OPERATING = "operating"        # gear retracted, any height in [MIN, MAX]
    LANDING = "landing"            # stops, lowers the gear at a safe height, descends onto it


@dataclass
class RobotCommand:
    speed_cmd: float = 0.0          # [m/s] + = forward
    yaw_speed_cmd: float = 0.0      # [rad/s] + = turn left
    leg_length: float = NOMINAL_LEG_LENGTH
    roll_angle: float = 0.0         # [rad]
    brake: bool = False             # hard brake: zeroes the speed target immediately
    park: bool = False              # True: stop, lower the gear and park; False: stand up and operate
    max_accel: float = 1.5          # [m/s^2] speed-target ramp (responsiveness); 1.5 = the original tuning
    # learned policy action in physical units, only used while operating (sim/warp_rl/controller.py, RESIDUAL):
    # (dL_xp, dL_xm [m], dth_xp, dth_xm [rad], dT_xp, dT_xm [N.m], dF_xp, dF_xm [N], jump trigger (> 0.5));
    # None = model-based only
    residual: tuple = None


@dataclass
class LegState:
    length: float = NOMINAL_LEG_LENGTH
    body_angle: float = 0.0     # leg angle in the base frame (0 = vertical), + = hip ahead
    theta: float = 0.0          # body_angle - phi (from world vertical)
    d_length: float = 0.0
    d_body_angle: float = 0.0
    d_theta: float = 0.0
    fwd: float = 0.0            # hip -> wheel vector in the base frame (forward, up)
    up: float = -NOMINAL_LEG_LENGTH


class PI:
    """Discrete PI from the reference firmware: integral clamped, output clamped (gains tuned for dt = 1 ms)."""

    def __init__(self, kp, ki, max_integral, max_output):
        self.kp, self.ki, self.max_integral, self.max_output = kp, ki, max_integral, max_output
        self.integral = 0.0

    def __call__(self, error):
        self.integral = clamp(self.integral + error * self.ki, -self.max_integral, self.max_integral)
        return clamp(error * self.kp + self.integral, -self.max_output, self.max_output)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lqr_k(leg_length):
    """K(L0), 2x6 (rows T, Tp; columns [theta, dTheta, x, dx, phi, dPhi]); u = -K x."""
    l0 = clamp(float(leg_length), *L0_VALID_RANGE)
    p = (l0 ** 3, l0 ** 2, l0, 1.0)
    return [[c[0] * p[0] + c[1] * p[1] + c[2] * p[2] + c[3] for c in row] for row in LQR_COEFFS]


def rot_to_rpy(R):
    """Base rotation (world <- base) -> (roll, pitch, yaw) in the robot convention (yaw Z, pitch X, roll Y)."""
    R = np.asarray(R).reshape(3, 3)
    return math.atan2(-R[2, 0], R[2, 2]), math.asin(clamp(R[2, 1], -1.0, 1.0)), math.atan2(-R[0, 1], R[1, 1])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def site_lateral(mujoco, model, site_id):
    """Lateral (X) position of a site in the base frame at qpos0 (the legs move only in their Y-Z planes)."""
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    R = d.xmat[1].reshape(3, 3)
    return float((R.T @ (d.site_xpos[site_id] - d.xpos[1]))[0])


def body_rpy(data, base_body=1):
    """TRUE (roll, pitch, yaw) of the base, for analysis scripts."""
    return rot_to_rpy(data.xmat[base_body])


# --------------------------------------------------------------------------------------------
# Simulated IMU
# --------------------------------------------------------------------------------------------
@dataclass
class ImuConfig:
    mode: str = "sensor"          # "sensor": fused orientation (as the WitMotion outputs); "mahony": gyro + accel
    gyro_noise: float = 0.0       # [rad/s] white noise std
    gyro_bias: float = 0.0        # [rad/s] std of a constant per-axis bias
    angle_noise: float = 0.0      # [rad] orientation noise ("sensor" mode)
    acc_noise: float = 0.0        # [m/s^2]
    mahony_kp: float = 0.5
    mahony_ki: float = 0.01
    mahony_acc_gate: float = 1.0  # [m/s^2] ignore the accelerometer when |a| is further than this from g
    seed: int = 0


class SimulatedIMU:
    """Reads the `imu` site sensors (same orientation as the base)."""

    def __init__(self, mujoco, model, config=None):
        self.cfg = config or ImuConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        adr = lambda n: int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)])
        self.adr_q, self.adr_g, self.adr_a = adr("imu_quat"), adr("imu_gyro"), adr("imu_acc")
        self.bias = self.rng.normal(0.0, self.cfg.gyro_bias, 3) if self.cfg.gyro_bias > 0 else np.zeros(3)
        self.q_est, self.e_int = None, np.zeros(3)

    def reset(self, data):
        self.q_est = np.array(data.sensordata[self.adr_q:self.adr_q + 4], float)
        self.e_int[:] = 0.0

    def read(self, data, dt):
        """-> (R world<-base, gyro [rad/s] and accelerometer [m/s^2] in the base frame)."""
        c, sd = self.cfg, data.sensordata
        gyro = np.array(sd[self.adr_g:self.adr_g + 3], float) + self.bias
        acc = np.array(sd[self.adr_a:self.adr_a + 3], float)
        if c.gyro_noise > 0:
            gyro = gyro + self.rng.normal(0.0, c.gyro_noise, 3)
        if c.acc_noise > 0:
            acc = acc + self.rng.normal(0.0, c.acc_noise, 3)
        if c.mode == "mahony":
            return self._mahony(gyro, acc, dt), gyro, acc
        R = quat_to_rot(np.array(sd[self.adr_q:self.adr_q + 4], float))
        if c.angle_noise > 0:
            R = R @ _small_rot(self.rng.normal(0.0, c.angle_noise, 3))
        return R, gyro, acc

    def _mahony(self, gyro, acc, dt):
        q = self.q_est if self.q_est is not None else np.array([1.0, 0, 0, 0])
        n, w = np.linalg.norm(acc), gyro.copy()
        if abs(n - 9.81) < self.cfg.mahony_acc_gate:     # correct only when the accelerometer ~ gravity
            e = np.cross(acc / n, quat_to_rot(q).T @ np.array([0.0, 0.0, 1.0]))
            self.e_int += self.cfg.mahony_ki * e * dt
            w = w + self.cfg.mahony_kp * e + self.e_int
        dq = 0.5 * np.array([-q[1] * w[0] - q[2] * w[1] - q[3] * w[2], q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
                             q[0] * w[1] - q[1] * w[2] + q[3] * w[0], q[0] * w[2] + q[1] * w[1] - q[2] * w[0]])
        q = q + dq * dt
        self.q_est = q / np.linalg.norm(q)
        return quat_to_rot(self.q_est)


def _small_rot(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# --------------------------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------------------------
class RepoControlPolicy:
    def __init__(self, mujoco, model, imu_config=None, state_source="robot"):
        """state_source: "robot" (default, real-robot sensors only) or "sim" (simulator ground truth)."""
        assert state_source in ("robot", "sim")
        self.state_source, self.mujoco, self.model = state_source, mujoco, model
        O = mujoco.mjtObj

        def mid(objtype, name):
            i = mujoco.mj_name2id(model, objtype, name)
            if i < 0:
                raise KeyError(f"'{name}' not found in the model")
            return i
        dof = lambda name: int(model.jnt_dofadr[mid(O.mjOBJ_JOINT, name)])
        mass = lambda name: float(model.body_mass[mid(O.mjOBJ_BODY, name)])

        self.base_body = mid(O.mjOBJ_BODY, "base")
        self.imu = SimulatedIMU(mujoco, model, imu_config)
        self.dt = float(model.opt.timestep)
        # actuators: ctrl = motor torque; joint torque = gear * ctrl
        self.act = {f"{j}_{s}": mid(O.mjOBJ_ACTUATOR, f"{j}_{s}") for s in SIDES for j in ("hip_Yp", "hip_Ym", "wheel")}
        self.gear = {k: float(model.actuator_gear[a][0]) for k, a in self.act.items()}
        self.joint_limit = {k: float(model.actuator_ctrlrange[a][1]) * self.gear[k] for k, a in self.act.items()}
        self.servo = {g: mid(O.mjOBJ_ACTUATOR, f"servo_{g}") for g in ("gear_front", "gear_rear")}
        self.gear_qpos = {g: int(model.jnt_qposadr[mid(O.mjOBJ_JOINT, g)]) for g in self.servo}
        self.site = {s: {k: mid(O.mjOBJ_SITE, f"{n}_{s}") for k, n in
                         (("hip", "hip_mid"), ("wheel", "wheel_center"), ("foot", "foot"))} for s in SIDES}
        self.hip_dofs = {s: [dof(f"hip_Yp_{s}"), dof(f"hip_Ym_{s}")] for s in SIDES}
        self.knee_dofs = {s: [dof(f"knee_Yp_{s}"), dof(f"knee_Ym_{s}")] for s in SIDES}
        self.shin_body = {s: mid(O.mjOBJ_BODY, f"shin_{s}") for s in SIDES}
        self.link_body = {s: mid(O.mjOBJ_BODY, f"linkYm_{s}") for s in SIDES}
        self.tire_geom = {s: mid(O.mjOBJ_GEOM, f"wheel_{s}_tire") for s in SIDES}
        # encoders: wheel (relative to the shin) and base -> arm Y+ -> shin (on the real robot the passive knee
        # angle comes from the 5-bar kinematics of the two hip encoders; here it is read from the joint)
        self.wheel_dof = {s: dof(f"wheel_{s}") for s in SIDES}
        self.shin_chain_dofs = {s: [dof(f"hip_Yp_{s}"), dof(f"knee_Yp_{s}")] for s in SIDES}
        self.wheel_radius = float(model.geom_size[self.tire_geom["xp"]][0])
        self.wheel_track_half = 0.5 * abs(site_lateral(mujoco, model, self.site["xp"]["wheel"]) - site_lateral(mujoco, model, self.site["xm"]["wheel"]))
        # mass below the hip motors that their torque does not "see": wheel + shin + link Y-
        self.below_hip_mass = mass("wheel_xp") + mass("shin_xp") + mass("linkYm_xp")
        # weight carried by the legs (everything but the wheels), per leg
        self.robot_mass = float(model.body_subtreemass[self.base_body])
        self.force_bias_n = (self.robot_mass - mass("wheel_xp") - mass("wheel_xm")) * 9.81 / 2.0

        self.yaw_pi = PI(3.0, 0.004, 2.0, 6.0)          # yaw rate -> wheel differential [N.m]
        self.leg_length_pi = PI(250.0, 0.2, 30.0, 140.0)  # leg length rate -> force along the leg [N]
        self.accel_step = 0.006       # [m/s per 4 ms] speed command ramp
        self.pos_lead = 0.15          # [m] how far the position target may lead the state
        self.speed_lead = 0.6         # [m/s]
        self.brake_speed_lead = 3.0
        self.max_yaw_lead = 1.0
        self.max_leg_length, self.min_leg_length = MAX_LEG_LENGTH, MIN_LEG_LENGTH
        self.air_retract_length = NOMINAL_LEG_LENGTH
        # jump primitive (triggered by the policy): push with the legs to jump_ext_length, then tuck them to
        # jump_tuck_length so the wheels clear an edge; the balance loops keep running
        self.jump_force, self.jump_ext_length, self.jump_push_max_s = 300.0, 0.50, 0.20
        self.jump_tuck_force, self.jump_tuck_length, self.jump_tuck_max_s = 200.0, 0.20, 0.25
        self.jump_cooldown_s = 0.5
        # active suspension from the simulated base height/vertical speed; 0 = off
        self.susp_height_gain, self.susp_vel_gain, self.susp_ref_alpha = 0.0, 0.0, 0.004
        # contact: tire "on the ground" when the normal force (true in "sim", estimated from hip torques in
        # "robot") exceeds the threshold; short hops are tolerated
        self.contact_min_force_n = 5.0 if state_source == "sim" else 15.0
        self.contact_lpf, self.air_grace_s = 0.1, 0.03
        self.theta_zero_ki, self.theta_zero_max = -0.1, 0.15   # [rad/(m.s)], [rad]
        # turning: the centripetal acceleration a = v * yaw_rate moves load to the outer wheel. Feed-forward the
        # extra outer-leg force so the body stays level, and limit the commanded turn rate so the inner wheel keeps
        # a minimum share of the weight (the robot tips over the outer wheel otherwise).
        hip = [site_lateral(mujoco, model, self.site[s]["hip"]) for s in SIDES]
        self.hip_spacing = abs(hip[0] - hip[1])           # lateral lever of the leg forces [m]
        self.com_height_fit = (0.585, 0.112)             # CoM height above ground = a * L0 + b [m] (measured)
        self.load_transfer_gain = 1.0                    # pure physics (m a h / lever); 1.0 keeps the body level
        self.inner_load_min = 0.2                        # minimum share of the weight on the inner wheel
        self.tip_height_factor = 1.25                    # effective tipping height / CoM height (measured in sim)
        self.a_lat_lpf = 0.05                            # ~20 ms low-pass of the measured lateral acceleration
        self.gear_down_angle = {"gear_front": 0.0, "gear_rear": 0.0}
        self.gear_up_angle = {"gear_front": math.radians(-100.0), "gear_rear": math.radians(100.0)}
        # automatic mode + landing gear (see Mode)
        self.gear_travel_s = 0.6        # [s] servo travel time (no position sensor)
        self.stand_rate = 0.12          # [m/s] leg speed up/down to the gear-safe height
        self.land_rate = 0.05           # [m/s] final descent onto the gear
        self.park_wheel_damping = 3.0   # [N.m/(m/s)] parked: hold the wheels
        self.park_leg_kp, self.park_leg_kd = 30.0, 1.5   # parked: legs perpendicular to the body
        self.mode, self.gear_down = Mode.OPERATING, False
        self.last_leg = {s: LegState() for s in SIDES}

    # ---------------------------------------------------------------------------------------
    def reset(self, data):
        self.yaw_pi.integral = self.leg_length_pi.integral = 0.0
        self.imu.reset(data)
        self._air_time = {s: 0.0 for s in SIDES}
        # start "on the ground": the torque-based estimate depends on the commanded force itself,
        # starting at 0 would lock the robot in "airborne"
        self.tire_force = {s: self.force_bias_n for s in SIDES}
        self._zrel_hist, self._ddz_f = {}, {}
        self.ref_z = float(data.qpos[2])
        self.x = self.dx = self.target_position = self.target_speed = self.theta_zero = self.a_lat = 0.0
        self.jump_phase, self.jump_t, self.jump_cool = 0, 0.0, 0.0
        R, gyro, _ = self.imu.read(data, self.dt)
        _, pitch, self.target_yaw = rot_to_rpy(R)
        self.last_leg = {s: self._leg_state(data, s, None, pitch, gyro[0]) for s in SIDES}
        # initial mode from the state: the real robot knows its last gear servo command and reads the hips
        avg_len = 0.5 * (self.last_leg["xp"].length + self.last_leg["xm"].length)
        self.gear_down = all(abs(float(data.qpos[a]) - self.gear_down_angle[g]) < 0.3 for g, a in self.gear_qpos.items())
        self.mode = (Mode.OPERATING if not self.gear_down else
                     Mode.PARKED if avg_len <= PARK_LEG_LENGTH + 0.02 else Mode.STANDING_UP)
        self.gear_timer, self.stable_time = self.gear_travel_s, 0.0
        self.length_ref = clamp(avg_len, self.min_leg_length, self.max_leg_length)

    def apply(self, data, command=None):
        command = command or RobotCommand()
        dt = self.dt
        # ---------- state: IMU + encoders ----------
        R, gyro, acc = self.imu.read(data, dt)
        roll, pitch, yaw = rot_to_rpy(R)
        pitch_rate, roll_rate, yaw_rate = float(gyro[0]), float(gyro[1]), float(gyro[2])
        legs = {s: self._leg_state(data, s, self.last_leg[s], pitch, pitch_rate) for s in SIDES}
        self.last_leg = legs
        self._forward_state(data, R, gyro)
        grounded = self._touching_ground(data, R, acc)
        avg_len = 0.5 * (legs["xp"].length + legs["xm"].length)
        avg_dlen = 0.5 * (legs["xp"].d_length + legs["xm"].d_length)

        command = self._update_mode(command, pitch, pitch_rate, avg_len, dt)
        r = command.residual if (command.residual is not None and self.mode == Mode.OPERATING) else (0.0,) * 9
        dl, dth, dT, dF = (r[0], r[1]), (r[2], r[3]), (r[4], r[5]), (r[6], r[7])      # (xp, xm)
        trigger = len(r) > 8 and r[8] > 0.5
        if self.mode == Mode.OPERATING:
            self.length_ref = clamp(self.length_ref + 0.5 * (dl[0] + dl[1]), self.min_leg_length, self.max_leg_length)
        # turn-rate envelope: |v * yaw_rate| <= a_max(leg length) (inner wheel keeps inner_load_min of the weight)
        h_com = self.com_height_fit[0] * avg_len + self.com_height_fit[1]
        v_ref = max(abs(self.dx), abs(command.speed_cmd), 0.3)
        a_max = (0.5 - self.inner_load_min) * 9.81 * (2 * abs(self.wheel_track_half)) / (self.tip_height_factor * h_com)
        yaw_limit = a_max / v_ref
        if abs(command.yaw_speed_cmd) > yaw_limit:
            command = replace(command, yaw_speed_cmd=math.copysign(yaw_limit, command.yaw_speed_cmd))
        balancing = self.mode != Mode.PARKED
        if balancing:
            self._update_target(command, yaw, avg_len, dt)
        else:   # parked: targets follow the state so the LQR starts without error when standing up
            self.target_position, self.target_speed, self.target_yaw, self.theta_zero = self.x, 0.0, yaw, 0.0
        # slow integral of the position error -> leg equilibrium offset. Compensates the off-axis CoM (the body
        # CoM is ~1 cm behind the hip) and slopes, without the robot slowly rolling away.
        if balancing and grounded and not command.brake:
            self.theta_zero = clamp(self.theta_zero + self.theta_zero_ki * (self.x - self.target_position) * dt,
                                    -self.theta_zero_max, self.theta_zero_max)

        # ---------- LQR ----------
        theta = 0.5 * (legs["xp"].theta + legs["xm"].theta) - self.theta_zero - 0.5 * (dth[0] + dth[1])
        d_theta = 0.5 * (legs["xp"].d_theta + legs["xm"].d_theta)
        state = [theta, d_theta, self.x - self.target_position, self.dx - self.target_speed, pitch, pitch_rate]
        K = lqr_k(avg_len)
        T = -sum(K[0][i] * state[i] for i in range(6))     # u = -K x
        Tp = -sum(K[1][i] * state[i] for i in range(6))

        # ---------- yaw (wheel differential): P on heading drift (outer) + PI on yaw rate (inner) ----------
        yaw_out = self.yaw_pi(command.yaw_speed_cmd + clamp(1.8 * wrap(self.target_yaw - yaw), -2.0, 2.0) - yaw_rate)
        if not balancing:
            T_side = {s: -self.park_wheel_damping * self.dx for s in SIDES}   # just hold the wheels
        elif grounded:
            T_side = {"xp": T + yaw_out + dT[0], "xm": T - yaw_out + dT[1]}      # right wheel ahead -> turns left
        else:
            T_side = {"xp": 0.0, "xm": 0.0}

        # ---------- force along the leg: length + weight + roll ----------
        body_z, body_vz = float(data.qpos[2]), float(data.qvel[2])
        self.ref_z += self.susp_ref_alpha * (body_z - self.ref_z)
        susp = clamp(-self.susp_height_gain * (body_z - self.ref_z) - self.susp_vel_gain * body_vz, -0.06, 0.06)
        on_ground = grounded or not balancing
        length_target = (clamp(self.length_ref + susp, self.min_leg_length, self.max_leg_length) if on_ground
                         else self.air_retract_length + 0.5 * (dl[0] + dl[1]))
        # jump primitive: idle -> push (legs out at full force) -> tuck (legs in) -> idle (after a cooldown)
        if self.mode != Mode.OPERATING:
            self.jump_phase = 0
        elif self.jump_phase == 0:
            self.jump_cool += self.dt
            if trigger and grounded and self.jump_cool >= self.jump_cooldown_s and avg_len < self.jump_ext_length - 0.05:
                self.jump_phase, self.jump_t = 1, 0.0
        elif self.jump_phase == 1:
            self.jump_t += self.dt
            if avg_len >= self.jump_ext_length or self.jump_t >= self.jump_push_max_s:
                self.jump_phase, self.jump_t = 2, 0.0
        else:
            self.jump_t += self.dt
            if (avg_len <= self.jump_tuck_length + 0.02 and self.jump_t > 0.05) or self.jump_t >= self.jump_tuck_max_s:
                self.jump_phase, self.jump_cool = 0, 0.0
        jump_f = 0.0
        if self.jump_phase == 1:
            length_target, jump_f = self.max_leg_length, self.jump_force
        elif self.jump_phase == 2:
            length_target, jump_f = self.jump_tuck_length, -self.jump_tuck_force
        # keep BOTH legs inside [min, max]: when the body is levelled on uneven ground (one leg shorter than the
        # other), shift the mean target instead of pushing one leg past its limit (e.g. one wheel on a ramp
        # with the legs high: the ramp leg retracts instead of the ground leg over-extending)
        half = 0.5 * abs(legs["xp"].length - legs["xm"].length)
        length_target = clamp(length_target, self.min_leg_length + half, self.max_leg_length - half)
        f_len = self.leg_length_pi(clamp(12.0 * (length_target - avg_len), -1.0, 1.0) - avg_dlen)
        roll_ref = command.roll_angle + math.atan2(dl[1] - dl[0], self.hip_spacing)   # leg length difference
        roll_out = clamp(25.0 * (clamp(12.0 * (roll_ref - roll), -3.0, 3.0) - roll_rate), -50.0, 50.0)
        bias = self.force_bias_n if on_ground else 0.0
        # centripetal load transfer (+ a_lat = towards the left -> the right leg is the outer one), feed-forward
        self.a_lat += self.a_lat_lpf * ((self.dx * yaw_rate if on_ground and balancing else 0.0) - self.a_lat)
        ff = self.load_transfer_gain * self.robot_mass * self.a_lat * h_com / self.hip_spacing
        F = {"xp": f_len + bias - roll_out + ff + dF[0] + jump_f,
             "xm": f_len + bias + roll_out - ff + dF[1] + jump_f}   # roll + (right down) -> push right
        for s in SIDES:
            if legs[s].length > self.max_leg_length:
                F[s] -= (legs[s].length - self.max_leg_length) * 2000.0

        # ---------- virtual leg torque: LQR + leg synchronisation (yaw differential splays the legs) ----------
        diff, d_diff = legs["xm"].theta - legs["xp"].theta, legs["xm"].d_theta - legs["xp"].d_theta
        # gains scale with the leg inertia about the hip (~L^2; tuned at 0.27 m): with fixed gains the tall legs
        # oscillated against the yaw loop when turning
        k = (avg_len / NOMINAL_LEG_LENGTH) ** 2
        la = clamp(4.0 * k * (clamp(40.0 * (dth[1] - dth[0] - diff), -8.0, 8.0) - d_diff), -25.0 * k, 25.0 * k)
        Tp_side = {"xm": Tp + la, "xp": Tp - la}
        if not balancing:   # parked: each leg perpendicular to the body (body_angle = 0)
            Tp_side = {s: -self.park_leg_kp * legs[s].body_angle - self.park_leg_kd * legs[s].d_body_angle
                       for s in SIDES}

        tau = {}
        for s in SIDES:
            tau[f"hip_Yp_{s}"], tau[f"hip_Ym_{s}"] = self._vmc(data, s, legs[s], F[s], Tp_side[s])
            tau[f"wheel_{s}"] = -T_side[s]          # rolling forward = negative joint torque (+X)
        if abs(theta) > math.pi / 3.0 or abs(pitch) > math.pi / 3.0:    # fell over: cut the torques
            tau = {k: 0.0 for k in tau}
        controls = {k: clamp(v, -self.joint_limit[k], self.joint_limit[k]) for k, v in tau.items()}
        for k, v in controls.items():
            data.ctrl[self.act[k]] = v / self.gear[k]
        for g, a in self.servo.items():
            data.ctrl[a] = (self.gear_down_angle if self.gear_down else self.gear_up_angle)[g]

        return {"theta": theta, "d_theta": d_theta, "pitch": pitch, "roll": roll, "yaw": yaw,
                "x": self.x, "dx": self.dx, "target_position": self.target_position, "target_speed": self.target_speed,
                "left_leg_length": legs["xm"].length, "right_leg_length": legs["xp"].length,
                "touching_ground": grounded, "tire_force": dict(self.tire_force),
                "mode": self.mode, "gear_down": self.gear_down, "leg_length_ref": self.length_ref,
                "a_lat": self.a_lat, "yaw_limit": yaw_limit, "jump_phase": self.jump_phase,
                "controls": controls}   # JOINT torques [N.m]

    # ---------------------------------------------------------------------------------------
    def _update_mode(self, command, pitch, pitch_rate, avg_len, dt):
        """Landing state machine. Updates mode, gear and self.length_ref; returns the effective command.

        parked --(park=False)--> standing_up: balance, raise the legs to >= GEAR_SAFE_LENGTH, retract the gear
            once still --> operating (gear up, any height)
        operating/standing_up --(park=True)--> landing: stop, go up to the safe height if needed, lower the gear,
            descend slowly to PARK_LEG_LENGTH --> parked."""
        # "still": not turning, not driving. The equilibrium pitch depends on the CoM (uncertain on the real
        # robot), so the pitch bound is loose; it only excludes a falling robot.
        stable = abs(pitch) < 0.25 and abs(pitch_rate) < 0.3 and abs(self.dx) < 0.1
        self.stable_time = self.stable_time + dt if stable else 0.0
        self.gear_timer += dt
        gear_settled = self.gear_timer >= self.gear_travel_s
        safe = GEAR_SAFE_LENGTH + 0.01
        wanted = clamp(command.leg_length, self.min_leg_length, self.max_leg_length)

        if self.mode == Mode.PARKED and not command.park:
            self.mode = Mode.STANDING_UP
        elif self.mode in (Mode.STANDING_UP, Mode.OPERATING) and command.park:
            self.mode = Mode.LANDING
        elif self.mode == Mode.LANDING and not command.park:
            self.mode = Mode.STANDING_UP

        if self.mode == Mode.STANDING_UP:
            if self.gear_down and avg_len >= GEAR_SAFE_LENGTH and self.stable_time >= 0.5:
                self._set_gear(False)
            elif not self.gear_down and gear_settled:
                self.mode = Mode.OPERATING
        elif self.mode == Mode.LANDING:
            if not self.gear_down and avg_len >= GEAR_SAFE_LENGTH - 0.005 and self.stable_time >= 0.3:
                self._set_gear(True)
            elif self.gear_down and gear_settled and avg_len <= PARK_LEG_LENGTH + 0.004:
                self.mode = Mode.PARKED

        if self.mode == Mode.PARKED:
            self.length_ref = PARK_LEG_LENGTH
        elif self.mode == Mode.OPERATING:
            self.length_ref = wanted
        else:
            rate = self.stand_rate
            if self.mode == Mode.LANDING and self.gear_down and gear_settled:
                target = PARK_LEG_LENGTH
                if self.length_ref <= safe:
                    rate = self.land_rate
            elif self.mode == Mode.STANDING_UP:
                target = wanted if not self.gear_down else max(wanted, safe)
            else:   # landing, gear still going down: stay at a safe height
                target = max(self.length_ref, safe)
            step = rate * dt
            self.length_ref += clamp(target - self.length_ref, -step, step)

        if self.mode == Mode.OPERATING:
            return command
        # standing up / landing / parked: stay in place, ignore the user's speed and turn rate
        return RobotCommand(leg_length=self.length_ref, roll_angle=command.roll_angle, park=command.park,
                            max_accel=command.max_accel)

    def _set_gear(self, down):
        if down != self.gear_down:
            self.gear_down, self.gear_timer = down, 0.0

    def _update_target(self, command, yaw, leg_length, dt):
        step_ratio = dt / 0.004
        scale = command.max_accel / 1.5          # 1.5 m/s^2 = accel_step per 4 ms at the nominal leg length
        slope = max(0.003 * scale * step_ratio,
                    (-(leg_length - NOMINAL_LEG_LENGTH) * 0.02 + self.accel_step) * scale * step_ratio)
        if command.brake:
            self.target_speed = 0.0
        else:
            err = command.speed_cmd - self.target_speed
            self.target_speed = command.speed_cmd if abs(err) < slope else self.target_speed + math.copysign(slope, err)
        lead = self.brake_speed_lead if command.brake else self.speed_lead * max(1.0, 0.75 * scale)
        self.target_position += self.target_speed * dt
        self.target_position = (self.x if command.brake else
                                clamp(self.target_position, self.x - self.pos_lead, self.x + self.pos_lead))
        self.target_speed = clamp(self.target_speed, self.dx - lead, self.dx + lead)
        if abs(command.yaw_speed_cmd) > 1e-3:
            self.target_yaw = yaw
        else:
            self.target_yaw = yaw + clamp(wrap(self.target_yaw - yaw), -self.max_yaw_lead, self.max_yaw_lead)

    def _forward_state(self, data, R, gyro):
        """Forward speed. "robot": wheel encoder (relative to the shin, where the stator is) + shin rotation in
        the world (IMU pitch rate + hip/knee kinematics); rolling without slip v = -R*omega (+X joint = backwards).
        Does not see slip or a wheel in the air, like the real robot. "sim": true wheel-centre speed on the heading."""
        if self.state_source == "robot":
            v = 0.0
            for s in SIDES:
                w_shin = float(gyro[0]) + sum(float(data.qvel[d]) for d in self.shin_chain_dofs[s])
                v += -self.wheel_radius * (float(data.qvel[self.wheel_dof[s]]) + w_shin)
            self.dx = v / len(SIDES)
        else:
            h = np.array(R)[:, 1].copy()
            h[2] = 0.0
            n = np.linalg.norm(h)
            h = h / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
            v, vel = np.zeros(6), np.zeros(3)
            for s in SIDES:
                self.mujoco.mj_objectVelocity(self.model, data, self.mujoco.mjtObj.mjOBJ_SITE, self.site[s]["wheel"], v, 0)
                vel += v[3:6]
            self.dx = float(np.dot(vel / len(SIDES), h))
        self.x += self.dx * self.dt

    def _leg_state(self, data, side, prev, pitch, pitch_rate):
        """Leg geometry from the kinematics (= hip encoders) in the base frame."""
        Rb = data.xmat[self.base_body].reshape(3, 3)
        v = Rb.T @ (data.site_xpos[self.site[side]["wheel"]] - data.site_xpos[self.site[side]["hip"]])
        fwd, up = float(v[1]), float(v[2])
        length = max(1e-6, math.hypot(fwd, up))
        body_angle = math.atan2(-fwd, -up)          # hip relative to the wheel; + = hip ahead
        st = LegState(length=length, body_angle=body_angle, theta=body_angle - pitch, fwd=fwd, up=up)
        if prev is not None:
            st.d_length = (length - prev.length) / self.dt
            st.d_body_angle = wrap(body_angle - prev.body_angle) / self.dt
            st.d_theta = st.d_body_angle - pitch_rate
        return st

    def _vmc(self, data, side, leg, force_n, tp):
        """VMC: force F along the leg (+ = extend, push the ground) and virtual torque Tp -> 2 hip torques,
        tau = J^T F_world with the closed-loop Jacobian."""
        L = max(leg.length, 1e-6)
        e = np.array([leg.fwd, leg.up]) / L                 # unit hip -> wheel
        perp = np.array([e[1], -e[0]])                      # force on the wheel that makes +Tp at the hip
        f_fu = force_n * e + (tp / L) * perp
        f_world = data.xmat[self.base_body].reshape(3, 3) @ np.array([0.0, f_fu[0], f_fu[1]])
        t = self._leg_jacobian(data, side).T @ f_world
        return float(t[0]), float(t[1])

    def _leg_jacobian(self, data, side):
        """3x2 world Jacobian of the wheel centre w.r.t. the 2 hips, passive knees following the closed loop."""
        m, mj, nv = self.model, self.mujoco, self.model.nv
        Js, J1, J2, scratch = (np.zeros((3, nv)) for _ in range(4))
        mj.mj_jacSite(m, data, Js, scratch, self.site[side]["wheel"])
        anchor = data.site_xpos[self.site[side]["foot"]]
        mj.mj_jac(m, data, J1, scratch, anchor, self.link_body[side])
        mj.mj_jac(m, data, J2, scratch, anchor, self.shin_body[side])
        C = J1 - J2
        hips, knees = self.hip_dofs[side], self.knee_dofs[side]
        return Js[:, hips] + Js[:, knees] @ (-np.linalg.lstsq(C[:, knees], C[:, hips], rcond=None)[0])

    def _touching_ground(self, data, R, acc):
        if self.state_source == "robot":
            force = self._estimated_normal_force(data, float((np.asarray(R) @ acc)[2]) - 9.81)
        else:   # simulator contacts: total normal force on each tire
            force, f6 = {s: 0.0 for s in SIDES}, np.zeros(6)
            for i in range(data.ncon):
                c = data.contact[i]
                for s in SIDES:
                    if self.tire_geom[s] in (c.geom1, c.geom2):
                        self.mujoco.mj_contactForce(self.model, data, i, f6)
                        force[s] += abs(float(f6[0]))
        self.tire_force = force
        for s in SIDES:
            self._air_time[s] = 0.0 if force[s] > self.contact_min_force_n else self._air_time[s] + self.dt
        return all(self._air_time[s] <= self.air_grace_s for s in SIDES)

    def _estimated_normal_force(self, data, acc_world_z):
        """As on the real robot (reference firmware): measured hip torques -> force of the leg on the wheel
        (inverting tau = J^T f in the leg plane) -> vertical part + m_below * (g + vertical wheel acceleration).
        Wheel acceleration = base acceleration (IMU) + 2nd derivative of the wheel height relative to the hip;
        without it, a wheel in the air would show the commanded force as support. Low-pass filtered."""
        Rb = data.xmat[self.base_body].reshape(3, 3)
        out = {}
        for s in SIDES:
            tau = np.array([float(data.qfrc_actuator[d]) for d in self.hip_dofs[s]])
            try:   # tau = A @ f_(forward, up) in the base frame
                f_fu = np.linalg.solve(self._leg_jacobian(data, s).T @ Rb[:, 1:3], tau)
            except np.linalg.LinAlgError:
                f_fu = np.zeros(2)
            fz = float((Rb[:, 1:3] @ f_fu)[2])
            leg = self.last_leg[s]
            zrel = float((Rb[:, 1:3] @ np.array([leg.fwd, leg.up]))[2])
            h = self._zrel_hist.setdefault(s, [zrel, zrel])
            ddz = (zrel - 2.0 * h[0] + h[1]) / (self.dt * self.dt)
            h[1], h[0] = h[0], zrel
            self._ddz_f[s] = self._ddz_f.get(s, 0.0) + 0.05 * (ddz - self._ddz_f.get(s, 0.0))
            n = -fz + self.below_hip_mass * (9.81 + acc_world_z + self._ddz_f[s])
            prev = self.tire_force.get(s, n)
            out[s] = prev + self.contact_lpf * (n - prev)
        return out
