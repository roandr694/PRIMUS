"""Carrega o modelo, lista juntas/atuadores e roda 5 s equilibrando com o controlador.

    conda run -n bipedal python3 scripts/check_model.py            # arena
    conda run -n bipedal python3 scripts/check_model.py --flat     # chao plano
"""
import argparse
import math

import _paths  # noqa: F401  (control/ no sys.path)
import robot_model as rm
from repo_control_policy import RepoControlPolicy, RobotCommand, body_rpy


def _name(mujoco, model, objtype, index):
    value = mujoco.mj_id2name(model, objtype, index)
    return value if value is not None else f"<sem nome:{index}>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    mujoco, model, data, path = rm.load(flat=args.flat)
    controller = RepoControlPolicy(mujoco, model)
    controller.reset(data)
    rm.stand_up(mujoco, model, data, controller)   # nasce estacionado; mede a partir do robo em pe

    print(f"Modelo: {path}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}  corpos={model.nbody} juntas={model.njnt} geoms={model.ngeom}")
    print(f"massa total = {model.body_subtreemass[1]:.2f} kg   restricoes de igualdade (lacos) = {model.neq}")
    print("\nJuntas:")
    for i in range(model.njnt):
        print(f"  {_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, i)}")
    print("\nAtuadores (ctrl = torque no motor; junta = gear x ctrl):")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  {_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):18s} gear={model.actuator_gear[i][0]:g}  ctrl [{lo:g}, {hi:g}]")

    steps = int(args.seconds / model.opt.timestep)
    worst_loop = 0.0
    for _ in range(steps):
        controller.apply(data, RobotCommand())
        mujoco.mj_step(model, data)
        for s in rm.SIDES:
            a = data.site(f"foot_{s}").xpos
            b = data.site(f"wheel_center_{s}").xpos
            worst_loop = max(worst_loop, float(((a - b) ** 2).sum() ** 0.5))

    roll, pitch, _ = body_rpy(data)
    print(f"\nDepois de {args.seconds:.1f} s equilibrando: base z={data.qpos[2]:.3f} m, pitch={math.degrees(pitch):+.2f} graus, "
          f"roll={math.degrees(roll):+.2f} graus")
    print(f"Maior folga do laco 5 barras (pe do link Y- vs eixo da roda): {worst_loop * 1000:.3f} mm")
    print("OK" if abs(pitch) < 0.3 else "!! caiu / instavel")


if __name__ == "__main__":
    main()
