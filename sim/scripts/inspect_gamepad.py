"""Prints live pygame joystick axes/buttons, to find the indices for run_xbox_viewer.py."""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description="Print live pygame joystick axis/button values.")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f"joystick_count={count}")
    if count < 1:
        print("No joystick detected. Pair the Xbox controller over Bluetooth and run again.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"name={joystick.get_name()}")
    print(f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")

    interval = 1.0 / max(args.rate, 0.1)
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        pygame.event.pump()
        axes = [round(float(joystick.get_axis(i)), 3) for i in range(joystick.get_numaxes())]
        buttons = [int(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
        print(f"axes={axes} buttons={buttons} hats={hats}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
