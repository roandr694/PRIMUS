"""Log de trajetoria headless para ajustar o controlador (como falha: cai, deriva, oscila).

    conda run -n bipedal python3 scripts/diagnose.py --seconds 6 --speed 0.5 --yaw 0.5
"""
import argparse
import math

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, ImuConfig, body_rpy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--leg", type=float, default=None, help="comprimento da perna (m)")
    ap.add_argument("--interval", type=float, default=0.25, help="intervalo do log (s)")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--imu", choices=["sensor", "mahony"], default="sensor")
    ap.add_argument("--state", choices=["robot", "sim"], default="robot",
                        help="robot = so sensores do robo real (encoders+IMU+torques); sim = estado perfeito do simulador")
    args = ap.parse_args()

    mujoco, model, data, _ = rm.load(flat=args.flat)
    controller = RepoControlPolicy(mujoco, model, ImuConfig(mode=args.imu), state_source=args.state)
    controller.reset(data)
    leg = args.leg if args.leg is not None else controller.target.leg_length
    steps = int(args.seconds / model.opt.timestep)
    every = max(1, int(args.interval / model.opt.timestep))

    print(f"{'t':>5} {'x':>7} {'dx':>6} {'z':>6} {'pitch':>6} {'roll':>6} {'L':>6} {'Troda':>6} {'Tquad':>6} {'Fpneu':>6} chao")
    for step in range(steps):
        go = step * model.opt.timestep >= 1.5   # assenta na pose inicial antes do comando
        info = controller.apply(data, RobotCommand(speed_cmd=args.speed if go else 0.0,
                                                   yaw_speed_cmd=args.yaw if go else 0.0, leg_length=leg))
        mujoco.mj_step(model, data)
        if step % every == 0:
            roll, pitch, _ = body_rpy(data)
            c = info["controls"]
            print(f"{step * model.opt.timestep:5.2f} {info['x']:7.3f} {info['dx']:6.2f} {data.qpos[2]:6.3f} "
                  f"{math.degrees(pitch):6.2f} {math.degrees(roll):6.2f} {info['right_leg_length']:6.3f} "
                  f"{c['wheel_xp']:6.2f} {c['hip_Yp_xp']:6.2f} {info['tire_force']['xp']:6.1f} {int(info['touching_ground'])}")
    roll, pitch, _ = body_rpy(data)
    print(f"\nResultado: {'CAIU/INSTAVEL' if abs(pitch) > 0.8 else 'em pe'} (z={data.qpos[2]:.3f}, pitch={math.degrees(pitch):.1f} graus)")


if __name__ == "__main__":
    main()
