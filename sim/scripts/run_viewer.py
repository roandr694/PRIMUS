"""MuJoCo viewer with the controller and a zero command (the robot stands up from parked and holds still).

    python3 scripts/run_viewer.py [--flat]          # macOS: mjpython scripts/run_viewer.py
"""
import argparse
import time

import mujoco.viewer

import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    mujoco, model, data, path = rm.load(flat=ap.parse_args().flat)
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    print(f"opening {path.name}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            for _ in range(4):
                controller.apply(data, RobotCommand())
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, 4 * model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
