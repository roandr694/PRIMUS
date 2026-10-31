"""Quantitative balance analysis: measures *how much* the robot swings.

Runs the controller headless and reports summary statistics that capture the
"swinging" the operator sees: pitch oscillation (RMS + peak-to-peak), vertical
bounce, ground-contact dropout (wheels going airborne kills the balance torque),
torque chatter, and drift. Use it to compare the controller before/after a tuning
change with one number instead of eyeballing the viewer.

    conda run -n bipedal python3 scripts/analyze_balance.py
    conda run -n bipedal python3 scripts/analyze_balance.py --speed 0.4 --yaw 0.0
    conda run -n bipedal python3 scripts/analyze_balance.py --csv /tmp/run.csv
    conda run -n bipedal python3 scripts/analyze_balance.py --flat --imu mahony --imu-noise

The "settled" window (default: last 60% of the run) is what the statistics are
computed over, so the start-up transient does not pollute the swing metrics.
"""

import argparse
import math
import numpy as np

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, ImuConfig, body_rpy

FALL_PITCH = 0.8   # [rad] |pitch| acima disso conta como queda
SETTLE_S = 1.5     # [s] comando zero no comeco


def rms(values):
    a = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def zero_crossings(values):
    """Count sign changes of (values - mean): a proxy for oscillation count."""
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return 0
    a = a - a.mean()
    s = np.sign(a)
    s[s == 0] = 1
    return int(np.sum(s[1:] != s[:-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--leg", type=float, default=None, help="Leg-length command (m)")
    parser.add_argument("--settle-frac", type=float, default=0.4,
                        help="Fraction of the run to skip before collecting stats")
    parser.add_argument("--csv", type=str, default=None, help="Optional per-step CSV dump")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path override (use the flat model for clean measurement)")
    parser.add_argument("--flat", action="store_true",
                        help="Shortcut for --model models/cena_plana.xml")
    parser.add_argument("--imu", choices=["sensor", "mahony"], default="sensor",
                        help="sensor = orientacao fundida do IMU; mahony = estima so com giro+acelerometro")
    parser.add_argument("--imu-noise", action="store_true", help="liga ruido/bias tipicos no IMU")
    parser.add_argument("--state", choices=["robot", "sim"], default="robot",
                        help="robot = so sensores do robo real (encoders+IMU+torques); sim = estado perfeito do simulador")
    args = parser.parse_args()

    mujoco, model, data, _ = rm.load(args.model, flat=args.flat)
    imu = ImuConfig(mode=args.imu)
    if args.imu_noise:
        imu.gyro_noise, imu.gyro_bias, imu.angle_noise, imu.acc_noise = 0.005, 0.002, 0.002, 0.05
    controller = RepoControlPolicy(mujoco, model, imu, state_source=args.state)
    controller.reset(data)

    leg = args.leg if args.leg is not None else controller.target.leg_length
    dt = model.opt.timestep
    steps = int(args.seconds / dt)
    settle_step = int(steps * args.settle_frac)

    rows = []
    fell_at = None
    for step in range(steps):
        go = step * dt >= SETTLE_S   # o robo nasce na pose do CAD e assenta antes de receber comando
        info = controller.apply(
            data, RobotCommand(speed_cmd=args.speed if go else 0.0, yaw_speed_cmd=args.yaw if go else 0.0, leg_length=leg))
        mujoco.mj_step(model, data)
        roll, pitch, yaw = body_rpy(data)
        ctrl = info["controls"]
        rows.append({
            "t": step * dt,
            "x": float(data.qpos[0]),
            "y": float(data.qpos[1]),
            "z": float(data.qpos[2]),
            "roll": roll,
            "pitch": pitch,
            "theta": info["theta"],
            "dx": info["dx"],
            "len": 0.5 * (info["left_leg_length"] + info["right_leg_length"]),
            "wheelL": ctrl["wheel_xm"],
            "wheelR": ctrl["wheel_xp"],
            "hipLf": ctrl["hip_Yp_xm"],
            "hipLr": ctrl["hip_Ym_xm"],
            "grnd": 1.0 if info["touching_ground"] else 0.0,
        })
        if fell_at is None and abs(pitch) > FALL_PITCH:
            fell_at = step * dt

    settled = rows[settle_step:]
    pitch = [r["pitch"] for r in settled]
    theta = [r["theta"] for r in settled]
    z = [r["z"] for r in settled]
    grnd = [r["grnd"] for r in settled]
    wheel = [r["wheelL"] for r in settled]
    hip = [r["hipLf"] for r in settled]

    pitch_pp = (max(pitch) - min(pitch)) if pitch else 0.0
    z_pp = (max(z) - min(z)) if z else 0.0
    air_frac = 1.0 - (sum(grnd) / len(grnd)) if grnd else 1.0
    window_s = args.seconds * (1.0 - args.settle_frac)
    osc_hz = zero_crossings(pitch) / (2.0 * window_s) if window_s > 0 else 0.0

    print(f"=== balance analysis: speed={args.speed} yaw={args.yaw} leg={leg:.3f} "
          f"({args.seconds:.0f}s, stats over last {window_s:.0f}s) ===")
    if fell_at is not None:
        print(f"  !! FELL at t={fell_at:.2f}s (|pitch| > {math.degrees(FALL_PITCH):.0f} deg)")
    print(f"  pitch:   RMS {math.degrees(rms(pitch)):6.2f} deg   "
          f"peak-to-peak {math.degrees(pitch_pp):6.2f} deg   ~{osc_hz:.1f} Hz")
    print(f"  theta:   RMS {math.degrees(rms(theta)):6.2f} deg   "
          f"peak-to-peak {math.degrees(max(theta)-min(theta)):6.2f} deg")
    print(f"  body z:  mean {np.mean(z):6.3f} m    peak-to-peak {z_pp*1000:6.1f} mm")
    print(f"  airborne: {air_frac*100:5.1f}% of settled window (wheels off ground -> balance torque OFF)")
    print(f"  leg len: mean {np.mean([r['len'] for r in settled]):6.3f} m  (commanded {leg:.3f} m)")
    print(f"  wheel T: RMS {rms(wheel):6.2f} Nm   peak {max(abs(v) for v in wheel):6.2f} Nm  (at the joint; limit {controller.joint_limit['wheel_xm']:.0f})")
    print(f"  hip  T:  RMS {rms(hip):6.2f} Nm   peak {max(abs(v) for v in hip):6.2f} Nm  (at the joint; limit {controller.joint_limit['hip_Yp_xm']:.0f})")
    print(f"  drift:   x {rows[-1]['x']-rows[0]['x']:+.3f} m   y {rows[-1]['y']-rows[0]['y']:+.3f} m")

    if args.csv:
        keys = list(rows[0].keys())
        with open(args.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(f"{r[k]:.6f}" for k in keys) + "\n")
        print(f"  wrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
