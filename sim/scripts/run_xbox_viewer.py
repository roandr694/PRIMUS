import argparse
import os
import time

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy
from xbox_controller import XboxController, XboxMapping


def _require_runtime():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError:
        print("MuJoCo is not installed in the active Python environment.")
        print("Use the existing conda env:")
        print("  conda run -n bipedal python3 -m pip install -r requirements.txt")
        raise SystemExit(1)
    try:
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        import pygame
    except ModuleNotFoundError:
        print("pygame is not installed in the active Python environment.")
        print("Use the existing conda env:")
        print("  conda run -n bipedal python3 -m pip install pygame")
        raise SystemExit(1)
    return mujoco, pygame


def main():
    parser = argparse.ArgumentParser(description="Run the MuJoCo viewer controlled by a wireless Xbox controller.")
    parser.add_argument("--speed-axis", type=int, default=1, help="Left stick Y (vertical).")
    parser.add_argument("--yaw-axis", type=int, default=2, help="Right stick X (horizontal). Use inspect_gamepad.py if your mapping differs.")
    parser.add_argument("--left-trigger-axis", type=int, default=4)
    parser.add_argument("--right-trigger-axis", type=int, default=5)
    parser.add_argument("--stop-button", type=int, default=0, help="Button that zeros speed/yaw while held (soft stop, coasts down). Xbox A is usually 0.")
    parser.add_argument("--brake-button", type=int, default=1, help="Button for a HARD brake (snap speed target to 0, max decel). Xbox B is usually 1.")
    parser.add_argument("--reset-button", type=int, default=7, help="Button that resets policy targets. Xbox Start/Menu is often 7.")
    parser.add_argument("--gear-button", type=int, default=2, help="Botao que alterna o trem de pouso. Xbox X costuma ser 2.")
    parser.add_argument("--flat", action="store_true", help="Chao plano em vez da arena.")
    parser.add_argument("--no-invert-speed", action="store_true")
    parser.add_argument("--no-invert-yaw", action="store_true", help="Yaw is inverted by default (stick right turns right); pass this to undo.")
    parser.add_argument("--max-speed", type=float, default=1.5, help="Velocidade maxima comandada (m/s).")
    parser.add_argument("--max-yaw-rate", type=float, default=1.8)
    parser.add_argument("--max-leg-adjust", type=float, default=0.09, help="Faixa de altura +-(m) em torno da perna nominal 0.27 m -> perna 0.18-0.36 m.")
    args = parser.parse_args()

    mujoco, pygame = _require_runtime()
    try:
        gamepad = XboxController(
            pygame,
            mapping=XboxMapping(
                speed_axis=args.speed_axis,
                yaw_axis=args.yaw_axis,
                left_trigger_axis=args.left_trigger_axis,
                right_trigger_axis=args.right_trigger_axis,
                stop_button=args.stop_button,
                brake_button=args.brake_button,
                reset_button=args.reset_button,
                gear_button=args.gear_button,
                invert_speed=not args.no_invert_speed,
                invert_yaw=not args.no_invert_yaw,
            ),
            max_speed=args.max_speed,
            max_yaw_rate=args.max_yaw_rate,
            max_leg_adjust=args.max_leg_adjust,
        )
    except RuntimeError as exc:
        print(exc)
        print("Use scripts/inspect_gamepad.py to verify pygame can see the controller.")
        raise SystemExit(1)

    _, model, data, path = rm.load(flat=args.flat)
    policy = RepoControlPolicy(mujoco, model)
    policy.reset(data)

    print(f"Opening MuJoCo viewer for {path}")
    print(f"Using controller: {gamepad.name}")
    print("Left stick Y: speed | right stick X: yaw | ride height: D-pad up/down OR triggers (latching)")
    print("A: soft stop (coast) | B: HARD brake | X: trem de pouso | Start/Menu: reset targets + height.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            command = gamepad.read()
            if gamepad.consume_reset_requested():
                policy.reset(data)
            policy.apply(data, command)
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
