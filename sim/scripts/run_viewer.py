"""MuJoCo viewer with the controller and a zero command (the robot stands up from parked and holds still).

    python3 scripts/run_viewer.py [--flat]          # macOS: mjpython scripts/run_viewer.py
    python3 scripts/run_viewer.py --house [--spawn upper_lounge]   # the CMA house (house/README.md)
"""
import argparse
import sys
import time

import mujoco.viewer

import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    ap.add_argument("--house", action="store_true", help="the CMA house scene (build it first: house/build.py)")
    ap.add_argument("--spawn", default="parked", help="--house: start keyframe")
    args = ap.parse_args()
    mujoco, model, data, path = rm.load(rm.HOUSE_MODEL_PATH if args.house else None, flat=args.flat)
    house_rt = None
    if args.house:
        sys.path.insert(0, str(rm.PROJECT_ROOT / "house"))
        from house_runtime import HouseRuntime
        rm.reset_to_key(mujoco, model, data, args.spawn)
        house_rt = HouseRuntime(mujoco, model)
        house_rt.reset(data)
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    print(f"opening {path.name}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            for _ in range(4):
                controller.apply(data, RobotCommand())
                if house_rt is not None:
                    house_rt.step(data)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, 4 * model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
