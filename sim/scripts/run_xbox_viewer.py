"""Drive the robot in the MuJoCo viewer with an Xbox (or any SDL) gamepad.

    python3 scripts/run_xbox_viewer.py [--flat]        # macOS: mjpython scripts/run_xbox_viewer.py

Left stick Y: speed | right stick X: turn | triggers or D-pad: leg height (latches, 0.18-0.52 m)
A: soft stop | B: hard brake | X: land <-> stand up (the landing gear is automatic) | Start: restart
LB / RB: responsiveness smooth <-> normal <-> sport.
If your gamepad maps differently, check the indices with scripts/inspect_gamepad.py.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer

import robot_model as rm
from pilot import Pilot
from repo_control_policy import RepoControlPolicy


class Gamepad:
    """pygame joystick -> logical pilot inputs (see pilot.py)."""

    def __init__(self, args):
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        try:
            import pygame
        except ModuleNotFoundError:
            raise SystemExit("pygame is missing: python3 -m pip install pygame")
        self.pygame, self.a = pygame, args
        pygame.init()
        if pygame.joystick.get_count() < 1:
            raise SystemExit("No gamepad found by pygame. Pair the controller and try scripts/inspect_gamepad.py.")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"gamepad: {self.js.get_name()}")

    def _axis(self, i, default=0.0):
        return float(self.js.get_axis(i)) if 0 <= i < self.js.get_numaxes() else default

    def _button(self, i):
        return 0 <= i < self.js.get_numbuttons() and bool(self.js.get_button(i))

    def read(self):
        self.pygame.event.pump()
        a = self.a
        trigger = lambda i: max(0.0, min(1.0, (self._axis(i, -1.0) + 1.0) / 2))   # SDL triggers: -1 released, +1 pressed
        dpad = float(self.js.get_hat(0)[1]) if self.js.get_numhats() else 0.0
        return dict(speed=self._axis(a.speed_axis) * (1 if a.no_invert_speed else -1),
                    yaw=self._axis(a.yaw_axis) * (-1 if a.no_invert_yaw else 1),
                    up=trigger(a.right_trigger_axis) + max(dpad, 0.0), down=trigger(a.left_trigger_axis) + max(-dpad, 0.0),
                    stop=self._button(a.stop_button), brake=self._button(a.brake_button),
                    park=self._button(a.gear_button), reset=self._button(a.reset_button),
                    slower=self._button(a.slower_button), faster=self._button(a.faster_button))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    ap.add_argument("--max-speed", type=float, default=1.5, help="[m/s]")
    ap.add_argument("--max-yaw-rate", type=float, default=1.8, help="[rad/s]")
    for name, default in (("speed-axis", 1), ("yaw-axis", 2), ("left-trigger-axis", 4), ("right-trigger-axis", 5),
                          ("stop-button", 0), ("brake-button", 1), ("gear-button", 2), ("reset-button", 7),
                          ("slower-button", 4), ("faster-button", 5)):
        ap.add_argument(f"--{name}", type=int, default=default)
    ap.add_argument("--no-invert-speed", action="store_true")
    ap.add_argument("--no-invert-yaw", action="store_true", help="by default stick right = turn right")
    args = ap.parse_args()

    pad, pilot = Gamepad(args), Pilot(args.max_speed, args.max_yaw_rate)
    _, model, data, path = rm.load(flat=args.flat)
    policy = RepoControlPolicy(mujoco, model)
    policy.reset(data)
    print(f"{path.name}: the robot starts parked and stands up by itself.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            level = pilot.level
            command, reset = pilot.command(pad.read(), model.opt.timestep)
            if pilot.level != level:
                print(f"responsiveness: {pilot.level_name}")
            if reset:
                rm.reset(mujoco, model, data)
                policy.reset(data)
            policy.apply(data, command)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
