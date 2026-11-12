"""Video of the skill policy on the skill terrain (CPU MuJoCo): from the foot of a staircase, up, across the top
and down the other side, with a camera following from the side.

    MUJOCO_GL=egl python3 scripts/record_skill_video.py --policy <run>/exported --level 3 out.mp4

The policy drives the whole time (as during a skill) with a constant forward speed and heading; it sees the true
terrain (--scan truth, as the teacher trained) or the elevation map of the simulated D435i (--scan map).
"""
import argparse
import math
import shutil
import subprocess

import numpy as np

import skill_world as SW
from perception.autonomy import Autonomy
from repo_control_policy import RepoControlPolicy, RobotCommand
import mujoco


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tile", default="stairs", choices=SW.ST.TYPES)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--start", default="foot", choices=("foot", "top", "center"), help="gaps: top = centre island")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--leg", type=float, default=0.30)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--scan", choices=("truth", "map"), default="truth")
    ap.add_argument("--torque", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    world = SW.SkillWorld(torque=args.torque)
    world.place(args.tile, args.level, args.start)
    m, d, sens = world.model, world.data, world.sens
    m.vis.map.znear, m.vis.map.zfar = 0.02 / m.stat.extent, 60.0 / m.stat.extent    # whole scene in the video
    ctl = RepoControlPolicy(mujoco, m)
    ctl.reset(d)
    auto = Autonomy(args.policy, policy_mode="always")
    renderer = mujoco.Renderer(m, args.height, args.width)
    cam = mujoco.MjvCamera()
    base = sens.base
    ff = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out], stdin=subprocess.PIPE)
    dt = m.opt.timestep
    every = int(round(1.0 / (args.fps * dt)))
    cmd, out, heading0, max_z = RobotCommand(leg_length=args.leg), None, None, 0.0
    cam_xy = d.qpos[0:2].copy()
    for k in range(int(args.seconds / dt)):
        t = k * dt
        if out is not None and k % 66 == 0 and args.scan == "map":
            L = 0.5 * (out["left_leg_length"] + out["right_leg_length"])
            auto.on_depth(*sens.depth_points(d), world.axle(), out["yaw"], L)
        if out is not None and k % 20 == 0:
            operating = out["mode"] == "operating"
            if operating and heading0 is None:
                heading0 = out["yaw"]
            yaw = 0.0 if heading0 is None else max(-1.5, min(1.5, 1.5 * math.atan2(
                math.sin(heading0 - out["yaw"]), math.cos(heading0 - out["yaw"]))))
            pilot = RobotCommand(speed_cmd=args.speed if operating and t > 1.5 else 0.0, yaw_speed_cmd=yaw,
                                 leg_length=args.leg)
            if operating:
                inputs = world.policy_inputs(ctl, out, auto, args.scan)
                cmd = RobotCommand(speed_cmd=pilot.speed_cmd, yaw_speed_cmd=pilot.yaw_speed_cmd,
                                   leg_length=args.leg, residual=auto.policy.act(cmd=pilot, **inputs))
            else:
                cmd = pilot
        out = ctl.apply(d, cmd)
        world.drive_limits()
        mujoco.mj_step(m, d)
        max_z = max(max_z, float(d.qpos[2]))
        if k % every == 0:
            cam_xy += 0.08 * (d.qpos[0:2] - cam_xy)                   # smooth follow
            cam.lookat[:] = [cam_xy[0], cam_xy[1], d.qpos[2] - 0.05]
            fwd = math.degrees(math.atan2(math.cos(heading0 or math.pi / 2), -math.sin(heading0 or math.pi / 2)))
            cam.distance, cam.azimuth, cam.elevation = 2.6, fwd - 70.0, -14.0
            renderer.update_scene(d, cam)
            ff.stdin.write(renderer.render().tobytes())
        if abs(out["pitch"]) > 1.2:
            print(f"fell at t={t:.1f} s")
            for _ in range(args.fps):                                 # one more second, then stop
                renderer.update_scene(d, cam)
                ff.stdin.write(renderer.render().tobytes())
            break
    ff.stdin.close()
    ff.wait()
    c = world.ter["centers"][args.level, SW.ST.TYPES.index(args.tile)]
    print(f"{args.out}: highest base z {max_z:.2f} m (top of the stairs: {5 * SW.ST.STAIR_RISE[args.level] + 0.41:.2f}), "
          f"end at x - centre = {d.qpos[0] - c[0]:+.2f} m, z {d.qpos[2]:.2f}")


if __name__ == "__main__":
    main()
