"""The CMA house as the robot sees it: physics + controller + the Pi runtime in-process, and the sensor stream the
Jetson gets on the real robot (docs/home_autonomy_architecture.md §6).

    sim = HomeSim(spawn="home")
    while ...:
        ev = sim.step(command, patch)       # 20 ms of physics (20 x 1 kHz); command / patch: link messages or None
        ev.state                            # link.State (odometry pose, drifting like the real wheel-inertial one)
        ev.depth                            # DepthFrame at 15 Hz (decimated D435i depth, for the elevation map)
        ev.rgbd                             # RGBDFrame at rgbd_hz (colour 640x360 + aligned depth, for semantics)
        ev.scan                             # LidarScan at 5.5 Hz (ranges + the pose to project them)
        sim.truth()                         # true pose, for scoring only

Everything the brain gets is in the odom frame: the sim keeps a drifting odometry (wheel scale error, gyro bias,
slip noise) and re-expresses sensor poses in it, exactly what a perfect perception stack would see on the robot.
The odom frame starts equal to the world (plan) frame at the spawn pose; sim.odom_to_world() gives the drift.
"""
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMUS = HERE.parents[1]
for p in (PRIMUS, PRIMUS / "sim" / "scripts", PRIMUS / "sim" / "house", PRIMUS / "robot", PRIMUS / "control"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mujoco  # noqa: E402

import link  # noqa: E402
import skill_world as SW  # noqa: E402
from perception.d435i import D435i  # noqa: E402
from pi_runtime import PiRuntime  # noqa: E402
from repo_control_policy import RepoControlPolicy, RobotCommand  # noqa: E402

EXPERIMENTS = PRIMUS.parent / "experiments"
STUDENT_V1 = EXPERIMENTS / "2026-11-19_16-46-41-student-v1" / "exported"
GT_PATH = PRIMUS / "sim" / "models" / "house" / "house_gt.json"

DT_CTRL = 0.02


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Pose:
    """Base pose: position [3] and rotation base -> frame [3, 3]; yaw in the controller convention (0 = +Y)."""
    p: np.ndarray
    R: np.ndarray

    @property
    def yaw(self):
        f = self.R[:, 1]                                       # base forward axis (+Y)
        return math.atan2(-f[0], f[1])

    @property
    def xy(self):
        return self.p[:2].copy()


@dataclass
class DepthFrame:
    stamp: float
    depth: np.ndarray            # [H, W] metres, 0 = no return (noisy, decimated D435i stream)
    pose: Pose                   # base pose in odom at capture
    cam: D435i


@dataclass
class RGBDFrame:
    stamp: float
    rgb: np.ndarray              # [H, W, 3] uint8
    depth: np.ndarray            # [H, W] metres aligned to the colour image, 0 = no return
    pose: Pose
    cam: D435i                   # intrinsics of the colour camera (same mount, colour FOV)


@dataclass
class LidarScan:
    stamp: float
    ranges: np.ndarray           # [n_beams] metres, 0 = no return; angle 0 = forward, counter-clockwise
    pose: Pose


@dataclass
class Events:
    state: link.State
    depth: DepthFrame = None
    rgbd: RGBDFrame = None
    scan: LidarScan = None
    fallen: bool = False
    info: dict = field(default_factory=dict)


class Odometry:
    """Wheel-inertial odometry with the errors of the real one: wheel radius scale error, gyro bias (random walk),
    and slip noise that grows with acceleration (hops, stairs)."""

    def __init__(self, pose, seed=0, scale_err=0.012, gyro_bias=2e-4, bias_walk=2e-5, slip=0.02):
        self.rng = np.random.default_rng(seed)
        self.scale = 1.0 + self.rng.normal(0.0, scale_err)
        self.bias = self.rng.normal(0.0, gyro_bias)               # [rad/s]
        self.bias_walk, self.slip = bias_walk, slip
        self.reset(pose)

    def reset(self, pose):
        self.x, self.y = float(pose.p[0]), float(pose.p[1])
        self.yaw_err = 0.0
        self.last_true = pose
        self.speed = self.yaw_rate = 0.0
        self.v_prev = 0.0

    def update(self, true_pose, v_world, wz, dt):
        f = true_pose.R[:, 1]
        v = float(v_world[:2] @ f[:2] / max(np.linalg.norm(f[:2]), 1e-6))   # forward speed over the ground
        acc = abs(v - self.v_prev) / dt
        self.v_prev = v
        self.bias += self.rng.normal(0.0, self.bias_walk * math.sqrt(dt))
        self.yaw_err += self.bias * dt
        yaw = true_pose.yaw + self.yaw_err
        vm = v * self.scale + self.rng.normal(0.0, self.slip * min(acc, 20.0) * math.sqrt(dt))
        th = yaw + math.pi / 2
        self.x += vm * math.cos(th) * dt
        self.y += vm * math.sin(th) * dt
        self.speed, self.yaw_rate = vm, wz + self.bias
        return self.pose(true_pose)

    def pose(self, true_pose):
        R = rot_z(self.yaw_err) @ true_pose.R
        return Pose(np.array([self.x, self.y, true_pose.p[2]]), R)


class HomeSim:
    def __init__(self, spawn="home", policy_dir=STUDENT_V1, seed=0, rgbd_hz=0.0, rgb_size=(640, 360),
                 torque=20.0, odom_noise=True, house_runtime=True):
        self.world = SW.SkillWorld(scene="house", torque=torque, seed=seed)
        self.m, self.d, self.sens = self.world.model, self.world.data, self.world.sens
        self.ctl = RepoControlPolicy(mujoco, self.m)
        self.pi = PiRuntime(policy_dir)
        self.seed, self.odom_noise, self.rgbd_hz = seed, odom_noise, rgbd_hz
        self.house = None
        if house_runtime:
            from house_runtime import HouseRuntime
            self.house = HouseRuntime(mujoco, self.m)
        # colour camera: same mount as the depth origin, colour FOV 69 x 42 deg (D435i RGB), depth aligned to it
        w, h = rgb_size
        self.color = D435i(width=w, height=h, vfov_deg=42.0, max_range=6.0)
        self.color.hfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(42.0) / 2) * w / h))  # square pixels
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, w)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, h)
        self._rgb_renderer = None
        self.reset(spawn)

    # ------------------------------------------------------------------------------------------------ setup
    def reset(self, spawn=None):
        if spawn is not None:
            self.spawn = spawn
        self.world.spawn = self.spawn
        self.world.place()
        self.world.tau[:] = 0.0
        self.ctl.reset(self.d)
        self.pi.reset()
        if self.house is not None:
            self.house.reset(self.d)
        mujoco.mj_forward(self.m, self.d)
        self.t, self.k = 0.0, 0
        self.out = self.ctl.apply(self.d, RobotCommand(park=True))
        self.odom = Odometry(self.truth(), seed=self.seed) if self.odom_noise else \
            Odometry(self.truth(), seed=self.seed, scale_err=0.0, gyro_bias=0.0, bias_walk=0.0, slip=0.0)
        self.fallen, self.fall_time = False, None
        self.rc = RobotCommand(park=True)
        self._next = dict(depth=0.0, rgbd=0.0, scan=0.0)

    def truth(self):
        p, R = self.sens.pose(self.d)
        return Pose(p, R)

    def odom_to_world(self):
        """(dx, dy, dyaw): world pose of the odom origin drift (for scoring)."""
        tp, op = self.truth(), self.odom.pose(self.truth())
        dyaw = tp.yaw - op.yaw
        Rz = rot_z(dyaw)
        t = tp.p - Rz @ op.p
        return t[0], t[1], dyaw

    # ------------------------------------------------------------------------------------------------ sensors
    def _odom_pose(self):
        return self.odom.pose(self.truth())

    def _depth(self):
        self.sens.renderer.update_scene(self.d, camera="d435i")
        depth = self.sens.cam.add_noise(self.sens.renderer.render(), self.sens.rng)
        return DepthFrame(self.t, depth, self._odom_pose(), self.sens.cam)

    def _rgbd(self):
        if self._rgb_renderer is None:
            c = self.color
            self._rgb_renderer = mujoco.Renderer(self.m, c.height, c.width)
            self._rgbd_renderer = mujoco.Renderer(self.m, c.height, c.width)
            self._rgbd_renderer.enable_depth_rendering()
            # the colour camera: a second camera at the same mount with the colour vertical FOV
            self._cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "d435i")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self._cam_id
        fovy = self.m.cam_fovy[self._cam_id]
        self.m.cam_fovy[self._cam_id] = self.color.vfov_deg
        try:
            self._rgb_renderer.update_scene(self.d, camera=cam)
            rgb = self._rgb_renderer.render().copy()
            self._rgbd_renderer.update_scene(self.d, camera=cam)
            depth = self._rgbd_renderer.render().copy()
        finally:
            self.m.cam_fovy[self._cam_id] = fovy
        depth = self.color.add_noise(depth, self.sens.rng)
        return RGBDFrame(self.t, rgb, depth, self._odom_pose(), self.color)

    def _scan(self):
        d, sens = self.d, self.sens
        p, R = sens.pose(d)
        lid = sens.lidar
        origin = p + R @ np.asarray(lid.pos_base)
        vec = (lid.directions_base() @ R.T).ravel()
        mujoco.mj_multiRay(self.m, d, origin, vec, sens.groups, 1, -1, sens.geomid, sens.dist, None, lid.n_beams,
                           lid.max_range)
        r = np.where(sens.geomid >= 0, sens.dist, 0.0)
        return LidarScan(self.t, lid.add_noise(r, sens.rng), self._odom_pose())

    # ------------------------------------------------------------------------------------------------ measurements
    def _meas(self, op):
        d, w = self.d, self.world
        return dict(ctl=self.ctl, R=op.R, gyro=d.qvel[3:6].copy(), wheel_speeds=d.qvel[w.wheel_dof].copy(),
                    base_z=float(d.qpos[2]), xy=op.xy, heading=op.yaw, contact_z=w.contact_z(self.ctl.wheel_radius))

    def _check_fall(self):
        R = self.d.xmat[self.sens.base].reshape(3, 3)
        tilted = R[2, 2] < math.cos(math.radians(50.0))
        if (tilted or abs(self.out["pitch"]) > 1.0) and not self.fallen:
            self.fallen, self.fall_time = True, self.t
        return self.fallen

    # ------------------------------------------------------------------------------------------------ 50 Hz step
    def step(self, command=None, patch=None, pad=None):
        """Advance one control period (20 ms). command / patch: the Jetson's latest messages (or None)."""
        if command is not None:
            self.pi.on_command(command)
        if patch is not None:
            self.pi.on_patch(patch)
        op = self._odom_pose()
        if self.out is not None:
            self.rc = self.pi.tick(self.t, self.out, self._meas(op), pad)
        n = int(round(DT_CTRL / self.m.opt.timestep))
        for _ in range(n):
            self.out = self.ctl.apply(self.d, self.rc)
            self.world.drive_limits()
            if self.house is not None:
                self.house.step(self.d)
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            self.k += 1
        tp = self.truth()
        op = self.odom.update(tp, self.d.qvel[0:3], float(self.d.qvel[5]), DT_CTRL)
        o = dict(x=op.p[0], y=op.p[1], z=op.p[2], yaw=op.yaw, speed=self.odom.speed, yaw_rate=self.odom.yaw_rate)
        ev = Events(self.pi.state(self.t, self.out, o, pad=pad), fallen=self._check_fall())
        if self.t >= self._next["depth"]:
            ev.depth, self._next["depth"] = self._depth(), self.t + 1 / 15
        if self.t >= self._next["scan"]:
            ev.scan, self._next["scan"] = self._scan(), self.t + 1 / 5.5
        if self.rgbd_hz > 0 and self.t >= self._next["rgbd"]:
            ev.rgbd, self._next["rgbd"] = self._rgbd(), self.t + 1 / self.rgbd_hz
        ev.info = dict(mode=str(self.out["mode"]), source=self.pi.source, lift=self.house.status() if self.house else "")
        return ev

    def close(self):
        for r in (self.sens.renderer, self._rgb_renderer, getattr(self, "_rgbd_renderer", None)):
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
