"""Avaliacao: controlador model-based sozinho (residual zero) x model-based + politica residual.

Os dois rodam exatamente os mesmos cenarios: mesmos ladrilhos (cada robo fixo num tipo/nivel de terreno),
mesmas sementes de comandos, empurroes, atrito e massa. Sem curriculum. Mede, por tipo e nivel de
terreno: quedas por minuto de robo, erro de velocidade e de giro, e o tamanho medio do residual.

    ~/test/isaac-rl/bin/python sim/warp_rl/eval.py --checkpoint <run>/logs/model_1999.pt --out <run>
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def make_env(num_envs, seed, device):
    from env import EnvCfg, PrimusWarpEnv
    cfg = EnvCfg(num_envs=num_envs, curriculum=False, random_tile_type=False, p_parked_start=0.0,
                 spawn_offset=0.25)
    env = PrimusWarpEnv(cfg, device=device, seed=seed)
    n_types, n_levels = len(env.terrain["types"]), cfg.terrain_levels
    idx = torch.arange(num_envs, device=env.device)
    env.ttype[:] = idx % n_types
    env.level[:] = (idx // n_types) % n_levels
    env._reset(idx)
    env._compute_obs()
    return env


def load_policy(env, checkpoint, device):
    from rsl_rl.runners import OnPolicyRunner
    from train import PPO_CFG
    runner = OnPolicyRunner(env, copy.deepcopy(PPO_CFG), log_dir=None, device=device)
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=device)


def rollout(env, policy, steps):
    """Roda `steps` passos; devolve somatorios por robo."""
    N = env.num_envs
    z = lambda: torch.zeros(N, device=env.device)
    falls, speed_err, yaw_err, res, oper = z(), z(), z(), z(), z()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            a = policy(obs) if policy is not None else torch.zeros(N, env.num_actions, device=env.device)
            R = env.R
            heading = torch.stack([R[:, 0, 1], R[:, 1, 1]], -1)
            heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            obs, _, dones, ex = env.step(a)
            o = (env.cs[:, 0] == 2).float()        # modo OPERATING
            v_fwd = (env.qvel[:, 0:2] * heading).sum(-1)
            yaw_rate = (env.R @ env.qvel[:, 3:6, None])[:, 2, 0]
            speed_err += (v_fwd - env.cmd[:, 0]).abs() * o
            yaw_err += (yaw_rate - env.cmd[:, 1]).abs() * o
            res += env.actions.clamp(-1, 1).abs().mean(-1) * o
            oper += o
            falls += (dones & ~ex["time_outs"]).float()
    return dict(falls=falls, speed_err=speed_err, yaw_err=yaw_err, res=res, oper=oper)


def summarize(env, r, steps):
    types = env.terrain["types"]
    minutes = steps * env.dt / 60.0
    out = {}
    for t, name in enumerate(types):
        for lv_group, lvs in (("easy", (0, 1)), ("medium", (2, 3)), ("hard", (4, 5))):
            sel = (env.ttype == t) & torch.isin(env.level, torch.tensor(lvs, device=env.device))
            if not sel.any():
                continue
            k = sel.float().sum()
            op = r["oper"][sel].sum().clamp_min(1)
            out[f"{name}/{lv_group}"] = dict(
                robots=int(k), falls_per_robot_minute=float(r["falls"][sel].sum() / (k * minutes)),
                speed_error_mps=float(r["speed_err"][sel].sum() / op), yaw_error_rps=float(r["yaw_err"][sel].sum() / op),
                mean_abs_action=float(r["res"][sel].sum() / op))
    k = env.num_envs
    out["all"] = dict(robots=k, falls_per_robot_minute=float(r["falls"].sum() / (k * minutes)),
                      speed_error_mps=float(r["speed_err"].sum() / r["oper"].sum()),
                      yaw_error_rps=float(r["yaw_err"].sum() / r["oper"].sum()),
                      mean_abs_action=float(r["res"].sum() / r["oper"].sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="pasta do experimento")
    ap.add_argument("--num-envs", type=int, default=2160)    # 6 tipos x 6 niveis x 60 robos
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    results = {}
    for label in ("model_based_only", "model_based_plus_rl"):
        env = make_env(args.num_envs, args.seed, args.device)
        policy = load_policy(env, args.checkpoint, args.device) if label.endswith("rl") else None
        steps = int(args.seconds / env.dt)
        results[label] = summarize(env, rollout(env, policy, steps), steps)
        del env, policy
        torch.cuda.empty_cache()

    base, rl = results["model_based_only"], results["model_based_plus_rl"]
    lines = ["| terreno | quedas/min LQR | quedas/min LQR+RL | erro vel. LQR | erro vel. LQR+RL | |residual| |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in base:
        lines.append(f"| {k} | {base[k]['falls_per_robot_minute']:.3f} | {rl[k]['falls_per_robot_minute']:.3f} | "
                     f"{base[k]['speed_error_mps']:.3f} | {rl[k]['speed_error_mps']:.3f} | {rl[k]['mean_abs_action']:.2f} |")
    table = "\n".join(lines)
    print(table)
    out = Path(args.out)
    (out / "evaluation.json").write_text(json.dumps(dict(checkpoint=str(args.checkpoint), seconds=args.seconds,
                                                         seed=args.seed, num_envs=args.num_envs, results=results), indent=2))
    (out / "evaluation.md").write_text(
        f"# Avaliacao\n\nCheckpoint: `{args.checkpoint}`\n\n{args.num_envs} robos, {args.seconds:.0f} s cada, "
        f"empurroes e randomizacao ligados, sem curriculum. Mesmos cenarios nas duas colunas.\n"
        f"Erro de velocidade em m/s (medio, so com o robo operando). |residual| = media de |acao| (0 a 1).\n\n{table}\n")


if __name__ == "__main__":
    main()
