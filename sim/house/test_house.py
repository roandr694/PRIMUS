"""Physics checks of the PRIMUS robot inside the CMA house (headless, model-based controller only).

    cd sim && python3 house/test_house.py            # all scenarios
    python3 house/test_house.py --only lift          # one scenario

Scenarios: stand up and drive at several spawn points (living room, varanda, upper lounge, garden pavers, pool
deck, sea deck, parking), drive into furniture (contact must be stable), push a pivot door open, and ride the
inclined lift from the terrace to the parking. Prints pass/fail per scenario and the simulation speed.
"""
import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import robot_model as rm  # noqa: E402
from house_runtime import HouseRuntime  # noqa: E402
from repo_control_policy import Mode, RobotCommand, body_rpy  # noqa: E402

SCENE = HERE.parent / "models" / "cena_casa_cma.xml"


def setup(key):
    mujoco, model, data, _ = rm.load(SCENE)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    assert kid >= 0, f"no keyframe {key}"
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    from repo_control_policy import RepoControlPolicy
    ctl = RepoControlPolicy(mujoco, model)
    ctl.reset(data)
    rt = HouseRuntime(mujoco, model)
    rt.reset(data)
    return mujoco, model, data, ctl, rt


def run(mujoco, model, data, ctl, rt, seconds, cmd_fn, log=None):
    n = int(seconds / model.opt.timestep)
    worst_tilt = 0.0
    for i in range(n):
        t = i * model.opt.timestep
        ctl.apply(data, cmd_fn(t))
        rt.step(data)
        mujoco.mj_step(model, data)
        r, p, _ = body_rpy(data)
        worst_tilt = max(worst_tilt, abs(r), abs(p))
        if log is not None and i % 500 == 0:
            log.append((t, data.qpos[0], data.qpos[1], data.qpos[2], rt.state))
    return worst_tilt


def fell(data):
    r, p, _ = body_rpy(data)
    return abs(r) > 0.8 or abs(p) > 0.8


def scenario_drive(key, speed=0.5, seconds=4.0, yaw=0.0):
    mujoco, model, data, ctl, rt = setup(key)
    rm.stand_up(mujoco, model, data, ctl)
    x0 = data.qpos[0:3].copy()
    t0 = time.time()
    tilt = run(mujoco, model, data, ctl, rt, seconds, lambda t: RobotCommand(speed_cmd=speed if t < seconds - 1 else 0.0,
                                                                       yaw_speed_cmd=yaw))
    wall = time.time() - t0
    d = float(math.hypot(*(data.qpos[0:2] - x0[0:2])))
    ok = not fell(data) and ctl.mode == Mode.OPERATING
    return ok, f"moved {d:.2f} m, z {x0[2]:.2f}->{data.qpos[2]:.2f}, max tilt {math.degrees(tilt):.1f} deg, " \
               f"realtime x{seconds / wall:.2f}"


def scenario_bump():
    """Drive into the sofa at 0.6 m/s from the living-room spawn: the robot must stay up or fall gracefully
    without the simulation blowing up."""
    mujoco, model, data, ctl, rt = setup("parked")
    rm.stand_up(mujoco, model, data, ctl)
    tilt = run(mujoco, model, data, ctl, rt, 5.0, lambda t: RobotCommand(speed_cmd=0.6 if t < 4 else 0.0,
                                                                     yaw_speed_cmd=0.0 if t < 1.2 else 0.6))
    finite = bool(abs(data.qpos).max() < 1e3)
    return finite, f"pos ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}), fell={fell(data)}, finite={finite}"


def scenario_door():
    """Push one of the varanda pivot doors with the robot (closed first) and check the door turns."""
    mujoco, model, data, ctl, rt = setup("parked")
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "living_pivot5_j")
    adr = model.jnt_qposadr[j]
    data.qpos[adr] = 0.0                           # close door 5 (y 10.3 .. 11.5), push it from the living room
    data.qpos[0:2] = [7.75, 11.0]
    h = math.pi                                     # heading west (-X); the base faces its local +Y
    data.qpos[3:7] = [math.cos((h - math.pi / 2) / 2), 0, 0, math.sin((h - math.pi / 2) / 2)]
    mujoco.mj_forward(model, data)
    ctl.reset(data)                                 # heading target from the new pose
    rm.stand_up(mujoco, model, data, ctl)
    a0 = float(data.qpos[adr])
    run(mujoco, model, data, ctl, rt, 6.0, lambda t: RobotCommand(speed_cmd=0.35 if t < 5 else 0.0))
    a1 = float(data.qpos[adr])
    return abs(a1 - a0) > 0.15 and not fell(data), f"door angle {math.degrees(a0):.0f} -> {math.degrees(a1):.0f} deg, " \
                                                   f"robot x {data.qpos[0]:.2f}, fell={fell(data)}"


def scenario_lift():
    """Terrace -> drive into the cabin -> stop -> automatic ride -> exit onto the parking."""
    mujoco, model, data, ctl, rt = setup("lift_entry")
    rm.stand_up(mujoco, model, data, ctl)
    log = []
    target_x = 27.3

    def go_in(t):
        dx = target_x - data.qpos[0]
        v = max(-0.3, min(0.4, 0.8 * dx))
        return RobotCommand(speed_cmd=v if abs(dx) > 0.05 else 0.0)
    run(mujoco, model, data, ctl, rt, 12.0, go_in, log)
    z_in = float(data.qpos[2])
    run(mujoco, model, data, ctl, rt, 40.0, lambda t: RobotCommand(), log)
    z_top = float(data.qpos[2])
    state_top = rt.state
    run(mujoco, model, data, ctl, rt, 8.0, lambda t: RobotCommand(speed_cmd=0.4 if t < 6 else 0.0), log)
    ok = state_top == "top" and z_top > 13.0 and data.qpos[0] > 40.2 and not fell(data)
    trace = " ".join(f"[{t:.0f}s x{x:.1f} z{z:.1f} {s}]" for t, x, y, z, s in log[::8])
    return ok, f"z inside {z_in:.2f}, at top {z_top:.2f} ({state_top}), exit x {data.qpos[0]:.2f} " \
               f"z {data.qpos[2]:.2f}, fell={fell(data)}\n      {trace}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    t0 = time.time()
    mujoco, model, data, _, _ = setup("parked")
    print(f"scene: {SCENE.name}  nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} "
          f"nmesh={model.nmesh}  load {time.time() - t0:.1f} s")
    tests = {
        "living": lambda: scenario_drive("parked", 0.5, 4.0),
        "varanda": lambda: scenario_drive("varanda", 0.6, 5.0),
        "upper_lounge": lambda: scenario_drive("upper_lounge", 0.4, 3.0),
        "garden": lambda: scenario_drive("garden", 0.5, 3.0),
        "pool_deck": lambda: scenario_drive("pool_deck", 0.5, 3.0),
        "sea_deck": lambda: scenario_drive("sea_deck", 0.5, 3.0),
        "lawn_turn": lambda: scenario_drive("lawn", 0.6, 5.0, yaw=0.5),
        "parking": lambda: scenario_drive("parking", 0.5, 4.0),
        "bump_sofa": scenario_bump,
        "push_door": scenario_door,
        "lift": scenario_lift,
    }
    fails = 0
    for name, fn in tests.items():
        if a.only and a.only != name:
            continue
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:                               # noqa: BLE001
            ok, msg = False, f"exception: {e}"
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:13s} {msg}   ({time.time() - t1:.1f} s)")
    print("all passed" if not fails else f"{fails} failed")
    return fails


if __name__ == "__main__":
    sys.exit(main())
