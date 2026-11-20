"""Autonomy stack in the CPU simulator, on the skill terrain (skill_world.py):

    D435i depth (rendered, with noise) + RPLIDAR A1 (ray cast) -> elevation map -> features
    -> skill planner / edge guard -> controller (+ skill policy)

A scripted pilot drives straight toward a feature and the script reports what the stack saw and did:

    MUJOCO_GL=egl python3 scripts/autonomy_sim.py --tile stairs --level 4 --start foot      # edge guard stop
    MUJOCO_GL=egl python3 scripts/autonomy_sim.py --tile stairs --level 3 --start top \\
        --policy <run>/exported --press-a                                                  # the skill crosses it
    ... --scan map     the policy sees the elevation map instead of the true terrain (the teacher saw the truth)
    ... --png map.png  elevation map + robot path at the end

Robot pose and heading come from the simulator (on the robot: odometry + IMU).
"""
import argparse

import numpy as np

import skill_world as SW
from perception.autonomy import Autonomy
from repo_control_policy import RepoControlPolicy, RobotCommand
import mujoco


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="stairs", choices=SW.ST.TYPES)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--start", default="foot", choices=("foot", "top", "center"))
    ap.add_argument("--speed", type=float, default=0.8, help="pilot speed command [m/s]")
    ap.add_argument("--leg", type=float, default=0.30)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--press-a", action="store_true", help="press the skill button when a skill is offered")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--policy", default=None, help="exported skill policy directory (export_policy.py)")
    ap.add_argument("--scan", choices=("truth", "map"), default="truth", help="policy terrain input")
    ap.add_argument("--policy-mode", choices=("skill", "always"), default="skill")
    ap.add_argument("--torque", type=float, default=20.0, help="joint torque limit [N.m] (with speed droop)")
    ap.add_argument("--trace", action="store_true", help="print the planner state every 0.2 s")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    world = SW.SkillWorld(torque=args.torque)
    world.place(args.tile, args.level, args.start)
    m, d, sens = world.model, world.data, world.sens
    ctl = RepoControlPolicy(mujoco, m)
    ctl.reset(d)
    auto = Autonomy(args.policy, args.policy_mode)
    auto.plan.guard_on = not args.no_guard
    dt = m.opt.timestep
    cmd, out, res = RobotCommand(leg_length=args.leg), None, None
    path, last_msg, pressed = [], None, False
    for k in range(int(args.seconds / dt)):
        t = k * dt
        if out is not None and k % 66 == 0:                                   # camera 15 Hz
            L = 0.5 * (out["left_leg_length"] + out["right_leg_length"])
            auto.on_depth(*sens.depth_points(d), world.axle(), out["yaw"], L)
        if out is not None and k % 182 == 0:                                  # lidar 5.5 Hz
            auto.on_lidar(sens.lidar_points(d), world.contact_z(ctl.wheel_radius), world.axle(), out["yaw"])
        if out is not None and k % 20 == 0:                                   # planner + policy 50 Hz
            operating = out["mode"] == "operating"
            speed = args.speed if operating and t > 1.0 else 0.0
            # press A whenever a skill is offered (a rising edge: alternate pressed / released)
            press = args.press_a and operating and not pressed and res is not None and res.state == "PILOT" \
                and res.offer is not None
            pressed = press
            pilot = RobotCommand(speed_cmd=speed, leg_length=args.leg)
            cmd, res = auto.step(0.02, pilot, {"skill": press}, out,
                                 lambda: world.policy_inputs(ctl, out, auto, args.scan))
            if res.message != last_msg:
                print(f"t={t:5.2f}  [{res.state:7s}] {res.message}")
                last_msg = res.message
            path.append(d.qpos[0:3].copy())
            if args.trace and k % 200 == 0:
                g = next((f for f in auto.ahead if f.guard), None)
                print(f"  t={t:5.2f} {res.state:7s} cmd {res.speed:5.2f} v {out['dx']:5.2f} brake {int(res.brake)} "
                      f"guard-feature {g.text() if g else '-'}")
        out = ctl.apply(d, cmd)
        world.drive_limits()
        mujoco.mj_step(m, d)
        if abs(out["pitch"]) > 1.0:
            print(f"t={t:5.2f}  FELL")
            break
    c = world.ter["centers"][args.level, SW.ST.TYPES.index(args.tile)]
    a = world.axle()
    print(f"final axle position rel. tile centre: ({a[0] - c[0]:+.2f}, {a[1] - c[1]:+.2f}) z {d.qpos[2]:.2f}; "
          f"features ahead: {[f.text() for f in auto.ahead[:3]]}")
    if args.png:
        save_png(args.png, auto.emap, np.array(path))


def save_png(path, emap, p):
    from PIL import Image
    n, res = emap.n, emap.res
    xs = (emap.center[0] + np.arange(-n // 2, n - n // 2)) * res
    ys = (emap.center[1] + np.arange(-n // 2, n - n // 2)) * res
    X, Y = np.meshgrid(xs + res / 2, ys + res / 2, indexing="xy")
    h, v, c = emap.lookup(np.stack([X, Y], -1))
    known = np.isfinite(h)
    lo, hi = (np.nanmin(h), np.nanmax(h)) if known.any() else (0.0, 1.0)
    val = np.clip((h - lo) / max(hi - lo, 1e-3), 0, 1)
    img = np.zeros((n, n, 3), np.uint8)
    img[..., 0] = np.where(known, 60 + 195 * val, 30)
    img[..., 1] = np.where(known, 60 + 120 * (1 - val), 30)
    img[..., 2] = np.where(np.isfinite(c), 255, np.where(known, 80, 30))
    for q in p:
        i, j = int((q[1] - ys[0]) / res), int((q[0] - xs[0]) / res)
        if 0 <= i < n and 0 <= j < n:
            img[i, j] = (255, 255, 255)
    Image.fromarray(img[::-1]).resize((n * 4, n * 4), Image.NEAREST).save(path)
    print(f"map: {path}")


if __name__ == "__main__":
    main()
