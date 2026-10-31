"""Abre o viewer do MuJoCo com o controlador parado (velocidade e yaw zero).

    conda run -n bipedal mjpython scripts/run_viewer.py            # macOS: mjpython
    conda run -n bipedal mjpython scripts/run_viewer.py --flat
"""
import argparse
import time

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()
    mujoco, model, data, path = rm.load(flat=args.flat)
    import mujoco.viewer
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    print(f"Abrindo {path} (controlador com comando zero). Backspace no viewer reseta.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            for _ in range(4):
                controller.apply(data, RobotCommand())
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = 4 * model.opt.timestep - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
