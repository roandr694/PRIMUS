"""Headless balance analysis: how much the robot swings, drifts and how hard the motors work.

The robot starts parked, stands up, holds still for 1.5 s, then follows the command. Statistics use the
"settled" window (last 60% of the run) so the start-up transient does not pollute them.

    python3 scripts/analyze_balance.py --flat                               # standing still
    python3 scripts/analyze_balance.py --flat --speed 0.6 --yaw 1.5          # driving and turning
    python3 scripts/analyze_balance.py --speed 0.5 --yaw 0.3 --height-amplitude 0.03   # arena, leg going up/down
    python3 scripts/analyze_balance.py --flat --imu mahony --imu-noise       # realistic IMU
    python3 scripts/analyze_balance.py --flat --log 0.25                     # trajectory log for tuning
    python3 scripts/analyze_balance.py --csv /tmp/run.csv                    # per-step dump
"""
import argparse
import math

import numpy as np

import robot_model as rm
from repo_control_policy import NOMINAL_LEG_LENGTH, ImuConfig, RepoControlPolicy, RobotCommand, body_rpy

FALL_PITCH = 0.8   # [rad]
SETTLE_S = 1.5     # [s] zero command at the start


def rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=0.0, help="[m/s]")
    ap.add_argument("--yaw", type=float, default=0.0, help="[rad/s]")
    ap.add_argument("--leg", type=float, default=NOMINAL_LEG_LENGTH, help="leg length command [m]")
    ap.add_argument("--height-amplitude", type=float, default=0.0, help="sinusoidal leg length variation (0.5 Hz) [m]")
    ap.add_argument("--settle-frac", type=float, default=0.4, help="fraction of the run skipped by the statistics")
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    ap.add_argument("--model", default=None, help="model path override")
    ap.add_argument("--imu", choices=["sensor", "mahony"], default="sensor",
                    help="sensor = fused IMU orientation; mahony = estimated from gyro + accelerometer")
    ap.add_argument("--imu-noise", action="store_true", help="typical IMU noise and bias")
    ap.add_argument("--state", choices=["robot", "sim"], default="robot",
                    help="robot = real-robot sensors only; sim = simulator ground truth")
    ap.add_argument("--log", type=float, default=0.0, help="print a trajectory line every LOG seconds")
    ap.add_argument("--csv", default=None, help="per-step CSV dump")
    args = ap.parse_args()

    mujoco, model, data, _ = rm.load(args.model, flat=args.flat)
    imu = ImuConfig(mode=args.imu)
    if args.imu_noise:
        imu.gyro_noise, imu.gyro_bias, imu.angle_noise, imu.acc_noise = 0.005, 0.002, 0.002, 0.05
    controller = RepoControlPolicy(mujoco, model, imu, state_source=args.state)
    controller.reset(data)
    rm.stand_up(mujoco, model, data, controller)
    dt = model.opt.timestep
    steps = int(args.seconds / dt)
    every = max(1, int(args.log / dt)) if args.log else 0
    if every:
        print(f"{'t':>5} {'x':>7} {'dx':>6} {'z':>6} {'pitch':>6} {'roll':>6} {'L':>6} {'Twheel':>6} {'Thip':>6} {'Ftire':>6} ground")

    rows, fell_at = [], None
    for step in range(steps):
        t = step * dt
        go = t >= SETTLE_S
        leg = args.leg + args.height_amplitude * math.sin(math.pi * t)
        info = controller.apply(data, RobotCommand(speed_cmd=args.speed if go else 0.0,
                                                   yaw_speed_cmd=args.yaw if go else 0.0, leg_length=leg))
        mujoco.mj_step(model, data)
        roll, pitch, yaw = body_rpy(data)
        c = info["controls"]
        rows.append(dict(t=t, x=float(data.qpos[0]), y=float(data.qpos[1]), z=float(data.qpos[2]), roll=roll,
                         pitch=pitch, theta=info["theta"], dx=info["dx"],
                         len=0.5 * (info["left_leg_length"] + info["right_leg_length"]),
                         wheelL=c["wheel_xm"], wheelR=c["wheel_xp"], hipLf=c["hip_Yp_xm"], hipLr=c["hip_Ym_xm"],
                         grnd=float(info["touching_ground"])))
        if fell_at is None and abs(pitch) > FALL_PITCH:
            fell_at = t
        if every and step % every == 0:
            print(f"{t:5.2f} {info['x']:7.3f} {info['dx']:6.2f} {data.qpos[2]:6.3f} {math.degrees(pitch):6.2f} "
                  f"{math.degrees(roll):6.2f} {info['right_leg_length']:6.3f} {c['wheel_xp']:6.2f} {c['hip_Yp_xp']:6.2f} "
                  f"{info['tire_force']['xp']:6.1f} {int(info['touching_ground'])}")

    s = rows[int(steps * args.settle_frac):]
    col = lambda k: np.array([r[k] for r in s])
    pitch, theta, z, wheel, hip = col("pitch"), col("theta"), col("z"), col("wheelL"), col("hipLf")
    window = args.seconds * (1.0 - args.settle_frac)
    sign = np.sign(pitch - pitch.mean())
    osc_hz = np.count_nonzero(sign[1:] != sign[:-1]) / (2.0 * window) if window > 0 else 0.0
    deg = math.degrees
    print(f"=== balance analysis: speed={args.speed} yaw={args.yaw} leg={args.leg:.3f} "
          f"({args.seconds:.0f}s, stats over last {window:.0f}s) ===")
    if fell_at is not None:
        print(f"  !! FELL at t={fell_at:.2f}s (|pitch| > {deg(FALL_PITCH):.0f} deg)")
    print(f"  pitch:   RMS {deg(rms(pitch)):6.2f} deg   peak-to-peak {deg(np.ptp(pitch)):6.2f} deg   ~{osc_hz:.1f} Hz")
    print(f"  theta:   RMS {deg(rms(theta)):6.2f} deg   peak-to-peak {deg(np.ptp(theta)):6.2f} deg")
    print(f"  body z:  mean {z.mean():6.3f} m    peak-to-peak {np.ptp(z) * 1000:6.1f} mm")
    print(f"  airborne: {(1.0 - col('grnd').mean()) * 100:5.1f}% of settled window (wheels off ground -> balance torque OFF)")
    print(f"  leg len: mean {col('len').mean():6.3f} m  (commanded {args.leg:.3f} m)")
    print(f"  wheel T: RMS {rms(wheel):6.2f} Nm   peak {np.abs(wheel).max():6.2f} Nm  (at the joint; limit {controller.joint_limit['wheel_xm']:.0f})")
    print(f"  hip  T:  RMS {rms(hip):6.2f} Nm   peak {np.abs(hip).max():6.2f} Nm  (at the joint; limit {controller.joint_limit['hip_Yp_xm']:.0f})")
    print(f"  drift:   x {rows[-1]['x'] - rows[0]['x']:+.3f} m   y {rows[-1]['y'] - rows[0]['y']:+.3f} m   "
          f"(odometry {info['x']:+.3f} m, final speed {info['dx']:+.2f} m/s)")

    if args.csv:
        keys = list(rows[0])
        with open(args.csv, "w") as fh:
            fh.write(",".join(keys) + "\n" + "".join(",".join(f"{r[k]:.6f}" for k in keys) + "\n" for r in rows))
        print(f"  wrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
