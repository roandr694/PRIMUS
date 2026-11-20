"""A CPU MuJoCo world with the robot's sensors, for the autonomy stack (perception/):

    SkillWorld   scene "skills": skill_terrain.py tiles (stairs, curbs, gaps, overhangs, ...) + the robot
                 (collision="full"), a start pose per tile; or "arena" / "flat" (the project scenes); with a D435i
                 camera and an RPLIDAR A1 on the base and the motor torque-speed droop
    SimSensors   rendered depth (with the D435i noise model) -> world points; lidar by ray casting
    TruthTerrain ground / ceiling of the terrain (what the teacher policy saw in training)
"""
import math
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (control/ on sys.path)
HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1], HERE.parent / "warp_rl"):          # PRIMUS/ (perception package), sim/warp_rl
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import mujoco                                                  # noqa: E402
import model as M                                              # noqa: E402
import skill_terrain as ST                                     # noqa: E402
from perception.d435i import D435i                             # noqa: E402
from perception.lidar import RPLidarA1                         # noqa: E402

W0 = 24 * 130 * 2 * math.pi / 60 / 9                           # joint no-load speed [rad/s] (skill_env.W0)


class SimSensors:
    """D435i (offscreen depth render of a camera on the base) and RPLIDAR A1 (mj_multiRay) in CPU MuJoCo."""

    def __init__(self, spec, cam=None, lidar=None, seed=0):
        self.cam, self.lidar = cam or D435i(), lidar or RPLidarA1()
        t = math.radians(self.cam.tilt_deg)
        spec.body("base").add_camera(name="d435i", pos=list(self.cam.pos_base), fovy=self.cam.vfov_deg,
                                     xyaxes=[1, 0, 0, 0, math.sin(t), math.cos(t)])
        self.model = spec.compile()
        # depth precision: MuJoCo sets the clip planes from the model extent (huge with the terrain)
        ext = self.model.stat.extent
        self.model.vis.map.znear, self.model.vis.map.zfar = 0.05 / ext, 2.0 * self.cam.max_range / ext
        # MuJoCo uses square pixels: the horizontal FOV follows from the vertical one and the aspect ratio
        self.cam.hfov_deg = math.degrees(2 * math.atan(math.tan(math.radians(self.cam.vfov_deg) / 2)
                                                       * self.cam.width / self.cam.height))
        self.renderer = mujoco.Renderer(self.model, self.cam.height, self.cam.width)
        self.renderer.enable_depth_rendering()
        self.rng = np.random.default_rng(seed)
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        n = self.lidar.n_beams
        self.geomid, self.dist = np.zeros(n, np.int32), np.zeros(n, np.float64)
        self.groups = np.array([1, 0, 0, 0, 1, 0], np.uint8)      # world geoms (0) and house colliders (4); robot is 2/3

    def pose(self, data):
        return data.xpos[self.base].copy(), data.xmat[self.base].reshape(3, 3).copy()

    def depth_points(self, data):
        self.renderer.update_scene(data, camera="d435i")
        depth = self.cam.add_noise(self.renderer.render(), self.rng)
        pts, sig = self.cam.points(depth, stride=2)
        p, R = self.pose(data)
        world, _ = self.cam.to_world(pts, p, R)
        return world, sig

    def lidar_points(self, data):
        p, R = self.pose(data)
        origin = p + R @ np.asarray(self.lidar.pos_base)
        vec = (self.lidar.directions_base() @ R.T).ravel()
        mujoco.mj_multiRay(self.model, data, origin, vec, self.groups, 1, -1, self.geomid, self.dist, None,
                           self.lidar.n_beams, self.lidar.max_range)
        r = np.where(self.geomid >= 0, self.dist, 0.0)
        return self.lidar.points(self.lidar.add_noise(r, self.rng), p, R)[0]


class TruthTerrain:
    """Ground (bilinear) and ceiling (nearest) of the fine perception grids of skill_terrain."""

    def __init__(self, ter):
        self.G, self.C, self.res = ter["ground"], ter["ceiling"], ter["fine_res"]
        self.F = ter.get("floor", ter["ground"])          # walkable surface (no walls): base height
        self.sx, self.sy = ter["size_x"], ter["size_y"]

    def floor(self, xy):
        G, self.G = self.G, self.F
        try:
            return self.ground(xy)
        finally:
            self.G = G

    def ground(self, xy):
        ny, nx = self.G.shape
        u = np.clip((xy[:, 0] + self.sx / 2) / self.res, 0, nx - 1.001)
        v = np.clip((xy[:, 1] + self.sy / 2) / self.res, 0, ny - 1.001)
        j, i = u.astype(int), v.astype(int)
        fu, fv = u - j, v - i
        G = self.G
        return (G[i, j] * (1 - fu) + G[i, j + 1] * fu) * (1 - fv) + (G[i + 1, j] * (1 - fu) + G[i + 1, j + 1] * fu) * fv

    def ceiling(self, xy):
        ny, nx = self.C.shape
        j = np.clip(np.round((xy[:, 0] + self.sx / 2) / self.res).astype(int), 0, nx - 1)
        i = np.clip(np.round((xy[:, 1] + self.sy / 2) / self.res).astype(int), 0, ny - 1)
        return self.C[i, j]


def start_pose(ter, tile, level, start):
    """foot: outside the feature facing the tile centre; top: on the stairs top / island / centre platform."""
    c = ter["centers"][level, ST.TYPES.index(tile)]
    if tile == "stairs_narrow":                                    # middle flight: foot facing up, or the top facing down
        x = ST.NARROW_FLIGHTS[1][0]
        return (c + np.array([x, 2.6]), math.pi) if start == "top" else (c + np.array([x, -3.2]), 0.0)
    if start == "foot":
        return c + np.array([ST.STAIR_HALF + 0.9, 0.0]), math.pi / 2
    if start == "top":
        return c + np.array([0.6 if tile == "stairs" else 0.0, 0.0]), -math.pi / 2
    return c + np.array([1.3, 0.0]), -math.pi / 2


class SkillWorld:
    def __init__(self, scene="skills", torque=20.0, seed=0, extra_cameras=(), torque_lag=0.002, power_limit=700.0,
                 spec_hook=None):
        import robot_model as rm
        self.scene = scene
        if scene == "skills":
            self.ter = ST.make_terrain(seed=seed)
            _, spec = M.build(self.ter, collision="full")
        elif scene == "plane":                                     # training collision shapes on a plane
            self.ter = None
            _, spec = M.build(None, collision="full")
        else:
            self.ter = None
            path = {"flat": rm.FLAT_MODEL_PATH, "house": rm.HOUSE_MODEL_PATH}.get(scene, rm.MODEL_PATH)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: build it with `python3 house/build.py` (from sim/)")
            spec = mujoco.MjSpec.from_file(str(path))
            M.apply_hip_limits(spec)
        self.spawn = "parked"                                      # house scene: keyframe used by place()
        for name, pos, fovy, tilt in extra_cameras:                # e.g. the pilot's first-person camera
            t = math.radians(tilt)
            spec.body("base").add_camera(name=name, pos=list(pos), fovy=fovy, xyaxes=[1, 0, 0, 0, math.sin(t), math.cos(t)])
        if spec_hook is not None:                                  # e.g. sim/home_sim adds a walking person
            spec_hook(spec)
        self.sens = SimSensors(spec, seed=seed)
        self.model = m = self.sens.model
        self.data = mujoco.MjData(m)
        self.torque = torque or float(m.actuator_ctrlrange[0, 1] * 9.0)
        m.actuator_ctrlrange[:6] = [-self.torque / 9.0, self.torque / 9.0]
        self.act_dof = [(a, int(m.jnt_dofadr[m.actuator_trnid[a, 0]])) for a in range(6)]
        self.wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"wheel_center_{s}") for s in ("xp", "xm")]
        self.wheel_dof = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_{s}")])
                          for s in ("xp", "xm")]
        self.truth = TruthTerrain(self.ter) if self.ter is not None else None
        self.torque_lag, self.power_limit = torque_lag, power_limit
        self.tau = np.zeros(len(self.act_dof))                    # applied torques (lag state)

    def place(self, tile="flat", level=0, start="center"):
        d = self.data
        if self.ter is None:                                       # project scene: its "parked" keyframe
            import robot_model as rm
            if self.scene == "plane":
                mujoco.mj_resetData(self.model, d)
                d.qpos[:] = M.parked_qpos()
                mujoco.mj_forward(self.model, d)
            elif self.scene == "house":
                rm.reset_to_key(mujoco, self.model, d, self.spawn)
            else:
                rm.reset(mujoco, self.model, d)
            return
        self.tau[:] = 0.0
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = M.parked_qpos()
        xy, h = start_pose(self.ter, tile, level, start)
        d.qpos[0:2] = xy
        d.qpos[2] += float(self.truth.ground(xy[None])[0])
        d.qpos[3:7] = [math.cos(h / 2), 0, 0, math.sin(h / 2)]
        mujoco.mj_forward(self.model, d)

    def axle(self):
        """wheel axle midpoint (distances to features are measured from it)"""
        return self.data.site_xpos[self.wheels, :2].mean(0)

    def contact_z(self, wheel_radius):
        return float(self.data.site_xpos[self.wheels, 2].mean() - wheel_radius)

    def drive_limits(self):
        """Plant model of the hip/wheel drives, as in the training kernel: torque-speed droop, torque lag
        (current loop + CAN) and the battery power limit shared by the motors doing positive work."""
        d, dt = self.data, self.model.opt.timestep
        alpha = dt / (self.torque_lag + dt)
        p_pos = 0.0
        for i, (a, dof) in enumerate(self.act_dof):
            tau, w = d.ctrl[a] * 9.0, d.qvel[dof]
            lim = self.torque * max(0.0, 1.0 - abs(w) / W0) if tau * w > 0 else self.torque
            self.tau[i] += alpha * (max(-lim, min(lim, tau)) - self.tau[i])
            p_pos += max(0.0, self.tau[i] * w)
        k = min(1.0, self.power_limit / p_pos) if p_pos > 0 else 1.0
        for i, (a, dof) in enumerate(self.act_dof):
            t = self.tau[i] * (k if self.tau[i] * d.qvel[dof] > 0 else 1.0)
            d.ctrl[a] = t / 9.0

    def policy_inputs(self, ctl, out, autonomy, scan="truth"):
        """SkillPolicy.act arguments. scan: "truth" (as the teacher trained) or "map" (the elevation map)."""
        d = self.data
        R = d.xmat[self.sens.base].reshape(3, 3).copy()
        cz = self.contact_z(ctl.wheel_radius)
        base_h = float(d.qpos[2] - (self.truth.floor(d.qpos[None, 0:2])[0] if self.truth else cz))
        if scan == "truth" and self.truth is not None:
            gfn, cfn = self.truth.ground, self.truth.ceiling
        else:
            gfn, cfn = (lambda xy: autonomy.emap.lookup(xy)[0]), autonomy.map_ceiling       # nan = unknown
        priv = dict(v_body=R.T @ d.qvel[0:3], base_height=base_h, torque_limit=self.torque, friction=1.0,
                    mass_scale=1.0, grounded=out["touching_ground"])
        return dict(ctl=ctl, R=R, gyro=d.qvel[3:6].copy(), wheel_speeds=d.qvel[self.wheel_dof].copy(),
                    base_z=float(d.qpos[2]), ground_fn=gfn, ceiling_fn=cfn, xy=d.qpos[0:2].copy(),
                    heading=out["yaw"], priv=priv, contact_z=cz)


class FallRecovery:
    """Stand-up policy (control/recovery_policy.py) in CPU MuJoCo: start() when the robot is down, then step() at
    every physics step instead of the balance controller until it is parked (upright, legs 0.18 m, gear down)."""

    def __init__(self, world, ctl, exported_dir, timeout=12.0):
        from recovery_policy import RecoveryPolicy
        self.world, self.ctl, self.pol, self.timeout = world, ctl, RecoveryPolicy(exported_dir), timeout
        m = world.model
        jq = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        jv = lambda n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
        names = ("hip_Yp_xp", "hip_Ym_xp", "hip_Yp_xm", "hip_Ym_xm")
        self.q_hips, self.v_hips = [jq(n) for n in names], [jv(n) for n in names]
        self.q_gear_rear = jq("gear_rear")
        self.q_gear = {g: jq(g) for g in ("gear_front", "gear_rear")}
        self.u_hips = [ctl.act[n] for n in names]
        self.u_wheels = [ctl.act["wheel_xp"], ctl.act["wheel_xm"]]
        self.active, self.t, self.hold, self.k = False, 0.0, 0.0, 0

    @staticmethod
    def tilted(data, base, limit_deg=50.0):
        R = data.xmat[base].reshape(3, 3)
        return R[2, 2] < math.cos(math.radians(limit_deg))

    def start(self):
        self.pol.reset()
        self.active, self.t, self.hold, self.k = True, 0.0, 0.0, 0

    def step(self):
        """one physics step of control; -> "running", "done" or "timeout" """
        w, d, ctl = self.world, self.world.data, self.ctl
        if self.k % 20 == 0:                                       # policy 50 Hz
            R = d.xmat[w.sens.base].reshape(3, 3)
            legs = [ctl._leg_state(d, s, None, 0.0, 0.0) for s in ("xp", "xm")]
            self.pol.act(R, d.qvel[3:6], [legs[0].length, legs[1].length, legs[0].body_angle, legs[1].body_angle],
                         d.qvel[self.v_hips], d.qvel[w.wheel_dof], d.qpos[self.q_gear_rear] > 1.0)
            upright = R[2, 2] > math.cos(math.radians(15.0))
            parked = max(abs(legs[0].length - 0.18), abs(legs[1].length - 0.18)) < 0.03
            still = np.linalg.norm(d.qvel[0:3]) < 0.2 and np.linalg.norm(d.qvel[3:6]) < 0.5
            gear_down = all(abs(d.qpos[q] - ctl.gear_down_angle[g]) < 0.3 for g, q in self.q_gear.items())  # servo there
            self.hold = self.hold + 0.02 if (upright and parked and still and gear_down) else 0.0
        self.k += 1
        self.t += w.model.opt.timestep
        tg = self.pol.targets
        tau = self.pol.pd(tg, d.qpos[self.q_hips], d.qvel[self.v_hips])
        for u, t in zip(self.u_hips, tau):
            d.ctrl[u] = t / 9.0
        for u, t in zip(self.u_wheels, tg["wheels"]):
            d.ctrl[u] = t / 9.0
        for g, a in ctl.servo.items():
            d.ctrl[a] = (ctl.gear_down_angle if tg["gear_down"] else ctl.gear_up_angle)[g]
        w.drive_limits()
        if self.hold >= 0.5:
            self.active = False
            ctl.reset(d)                                           # gear down + legs short -> PARKED
            return "done"
        if self.t > self.timeout:
            self.active = False
            return "timeout"
        return "running"
