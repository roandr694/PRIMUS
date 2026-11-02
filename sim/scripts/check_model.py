"""Loads the model, lists joints/actuators, stands up and balances for 5 s, and measures the 5-bar loop gap.

    python3 scripts/check_model.py [--flat]
"""
import argparse
import math

import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, body_rpy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    mujoco, model, data, path = rm.load(flat=args.flat)
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    rm.stand_up(mujoco, model, data, controller)

    name = lambda t, i: mujoco.mj_id2name(model, t, i) or f"<unnamed {i}>"
    print(f"model: {path}\nnq={model.nq} nv={model.nv} nu={model.nu} bodies={model.nbody} joints={model.njnt} geoms={model.ngeom}")
    print(f"total mass = {model.body_subtreemass[1]:.2f} kg   equality constraints (loops) = {model.neq}\n\njoints:")
    print("\n".join(f"  {name(mujoco.mjtObj.mjOBJ_JOINT, i)}" for i in range(model.njnt)))
    print("\nactuators (ctrl = motor torque; joint = gear x ctrl):")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {name(mujoco.mjtObj.mjOBJ_ACTUATOR, i):18s} gear={model.actuator_gear[i][0]:g}  ctrl [{lo:g}, {hi:g}]")

    worst_loop = 0.0
    for _ in range(int(args.seconds / model.opt.timestep)):
        controller.apply(data, RobotCommand())
        mujoco.mj_step(model, data)
        for s in rm.SIDES:
            worst_loop = max(worst_loop, float(((data.site(f"foot_{s}").xpos - data.site(f"wheel_center_{s}").xpos) ** 2).sum() ** 0.5))
    roll, pitch, _ = body_rpy(data)
    print(f"\nafter {args.seconds:.1f} s balancing: base z={data.qpos[2]:.3f} m, pitch={math.degrees(pitch):+.2f} deg, "
          f"roll={math.degrees(roll):+.2f} deg")
    print(f"largest 5-bar loop gap (link Y- anchor vs wheel axle): {worst_loop * 1000:.3f} mm")
    print("OK" if abs(pitch) < 0.3 else "!! fell / unstable")


if __name__ == "__main__":
    main()
