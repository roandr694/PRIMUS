"""Records a headless mp4 of the robot in the arena with the controller: stands up, changes height, drives,
turns, goes up and down the ramp. Needs ffmpeg on PATH.

    python3 scripts/record_video.py docs/demo_arena.mp4        # Linux without a display: MUJOCO_GL=egl
"""
import argparse
import math
import subprocess

import numpy as np

import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, body_rpy

# (start time, command). The robot starts parked facing world +X, towards the ramp.
SCRIPT = [
    (0.0, dict()),
    (2.0, dict(leg_length=0.20)),
    (4.0, dict(leg_length=0.34)),
    (6.0, dict()),
    (7.0, dict(speed_cmd=0.7)),                       # ramp: up, platform, down
    (17.0, dict(speed_cmd=0.0)),
    (18.5, dict(yaw_speed_cmd=1.5)),
    (20.6, dict(speed_cmd=0.8, yaw_speed_cmd=0.6)),
    (24.0, dict(speed_cmd=0.0, brake=True)),
    (25.0, dict()),
    (27.0, None),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="demo_arena.mp4")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()

    mujoco, model, data, _ = rm.load(flat=args.flat)
    ctrl = RepoControlPolicy(mujoco, model)
    ctrl.reset(data)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 2.0, -15.0
    dt = model.opt.timestep
    T = SCRIPT[-1][0]
    every = int(round(1.0 / (args.fps * dt)))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", args.out],
                          stdin=subprocess.PIPE)
    look = data.qpos[:3].copy()
    fell = None
    for i in range(int(T / dt)):
        t = i * dt
        cmd = RobotCommand()
        for t0, kw in SCRIPT:
            if kw is not None and t >= t0:
                cmd = RobotCommand(**kw)
        ctrl.apply(data, cmd)
        mujoco.mj_step(model, data)
        _, pitch, _ = body_rpy(data)
        if fell is None and abs(pitch) > 0.8:
            fell = t
        if i % every == 0:
            look += 0.08 * (data.qpos[:3] - look)
            cam.lookat[:] = [look[0], look[1], look[2] - 0.08]
            cam.azimuth = 200.0 + 50.0 * math.sin(2 * math.pi * t / T)
            renderer.update_scene(data, cam)
            ff.stdin.write(renderer.render().tobytes())
    renderer.close()
    ff.stdin.close()
    ff.wait()
    print(f"video: {args.out}  ({T:.0f} s)  {'fell at t=%.1f s' % fell if fell else 'no fall'}")


if __name__ == "__main__":
    main()
