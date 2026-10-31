"""Demo headless curta: anda para frente com uma pequena variacao de altura.

    conda run -n bipedal python3 scripts/drive_demo.py --seconds 5 --target-speed 0.5
"""
import argparse
import math

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, NOMINAL_LEG_LENGTH, body_rpy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--target-speed", type=float, default=0.5, help="velocidade para frente (m/s)")
    ap.add_argument("--yaw", type=float, default=0.0, help="taxa de yaw (rad/s)")
    ap.add_argument("--height-amplitude", type=float, default=0.03, help="amplitude da variacao de perna (m)")
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()

    mujoco, model, data, _ = rm.load(flat=args.flat)
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    x0 = data.qpos[:3].copy()
    for step in range(int(args.seconds / model.opt.timestep)):
        t = step * model.opt.timestep
        leg = NOMINAL_LEG_LENGTH + args.height_amplitude * math.sin(2.0 * math.pi * 0.5 * t)
        info = controller.apply(data, RobotCommand(speed_cmd=args.target_speed if t > 1.0 else 0.0,
                                                   yaw_speed_cmd=args.yaw if t > 1.0 else 0.0, leg_length=leg))
        mujoco.mj_step(model, data)
    roll, pitch, yaw = body_rpy(data)
    print(f"Simulado {args.seconds:.2f} s")
    print(f"Deslocamento da base (mundo): {data.qpos[0] - x0[0]:+.3f} {data.qpos[1] - x0[1]:+.3f} m;  z = {data.qpos[2]:.3f} m")
    print(f"Distancia para frente (odometria do controlador): {info['x']:+.3f} m;  velocidade final {info['dx']:+.2f} m/s")
    print(f"pitch {math.degrees(pitch):+.2f} graus  roll {math.degrees(roll):+.2f}  yaw {math.degrees(yaw):+.1f}")


if __name__ == "__main__":
    main()
