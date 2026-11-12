"""Exports a trained policy (RSL-RL checkpoint) to TorchScript and ONNX, with the observation normaliser inside,
for the CPU simulator and the robot (Jetson: ONNX -> TensorRT).

    ~/test/isaac-rl/bin/python sim/warp_rl/export_policy.py --task skills --checkpoint <run>/logs/model_3999.pt

Writes <run>/exported/policy.pt, policy.onnx and policy.json (task, observation size, action scale).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", choices=("residual", "skills", "recovery", "student"), default="skills")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--recurrent", action="store_true", help="student checkpoint with a GRU")
    ap.add_argument("--arch", choices=("mlp", "encoders"), default="mlp", help="skills teacher network")
    args = ap.parse_args()

    from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    from train import DISTILL_CFG, PPO_CFG, recurrent_student, skill_ppo
    if args.task == "skills":
        from skill_env import SkillCfg as Cfg, SkillEnv as Env
    elif args.task == "recovery":
        from recovery_env import RecoveryCfg as Cfg, RecoveryEnv as Env
    elif args.task == "student":
        from student_env import StudentCfg as Cfg, StudentEnv as Env
    else:
        from env import EnvCfg as Cfg, PrimusWarpEnv as Env
    cfg = Cfg(num_envs=16)
    env = Env(cfg, device=args.device)
    ppo = copy.deepcopy(DISTILL_CFG if args.task == "student" else PPO_CFG)
    if args.recurrent:
        ppo["student"] = recurrent_student()
    if args.task == "skills":
        ppo["actor"]["distribution_cfg"].update(init_std=0.3, std_range=[0.1, 0.5])
    elif args.task == "recovery":
        ppo["actor"]["distribution_cfg"].update(init_std=0.5, std_range=[0.1, 0.8])
    if args.task == "skills":
        ppo = skill_ppo(env, args.arch, ppo)
    runner = (DistillationRunner if args.task == "student" else OnPolicyRunner)(env, ppo, log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    out = Path(args.checkpoint).resolve().parents[1] / "exported"
    out.mkdir(exist_ok=True)
    runner.export_policy_to_jit(str(out), "policy.pt")
    runner.export_policy_to_onnx(str(out), "policy.onnx")
    student = args.task == "student"
    meta = dict(task=args.task, checkpoint=str(Path(args.checkpoint).resolve()), obs_version=3 if student else 2,
                obs_dim=int(env.obs["student" if student else "policy"].shape[1]), num_actions=env.num_actions,
                residual_scale=list(getattr(cfg, "residual_scale", ())),
                residual_offset=list(getattr(cfg, "residual_offset", None) or ()),
                history=cfg.student_history if student else cfg.history, recurrent=args.recurrent)
    (out / "policy.json").write_text(json.dumps(meta, indent=2))
    print(f"exported to {out}: {meta}")


if __name__ == "__main__":
    main()
