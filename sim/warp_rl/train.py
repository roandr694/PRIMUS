"""Trains the residual policy on the GPU (MuJoCo Warp + RSL-RL PPO).

    ~/test/isaac-rl/bin/python sim/warp_rl/train.py --num-envs 4096 --iterations 1500
    ~/test/isaac-rl/bin/python sim/warp_rl/train.py --task skills --iterations 4000     # skill teacher (skill_env.py)
    ~/test/isaac-rl/bin/python sim/warp_rl/train.py --task recovery --iterations 1500   # stand up after a fall
    ~/test/isaac-rl/bin/python sim/warp_rl/train.py --task student --teacher <teacher run>/logs/model_3999.pt

Each run creates experiments/<date>-warp-residual/ in the workspace (or in $PRIMUS_WORKSPACE): manifest
(arguments, Git revision, diff), checkpoints and TensorBoard in logs/. At the end it runs the evaluation
(eval.py: model-based alone vs model-based + residual) and records the videos (render.py).
"""
import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = Path(os.environ.get("PRIMUS_WORKSPACE", REPO.parent)).expanduser().resolve()
sys.path.insert(0, str(HERE))

PPO_CFG = {
    "seed": 0,
    "num_steps_per_env": 24,
    "save_interval": 100,
    "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "check_for_nan": True,
    "logger": "tensorboard",
    "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
              "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar",
                                   "std_range": [0.05, 0.3]}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
               "obs_normalization": True, "distribution_cfg": None},
    "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3,
                  "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.0, "desired_kl": 0.01,
                  "max_grad_norm": 1.0, "optimizer": "adam", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                  "clip_param": 0.2, "normalize_advantage_per_mini_batch": False, "share_cnn_encoders": False,
                  "rnd_cfg": None, "symmetry_cfg": None},
}

# student: distilled from the skill teacher (DAgger: the student drives, the teacher labels the actions)
DISTILL_CFG = {
    "seed": 0,
    "num_steps_per_env": 24,
    "save_interval": 100,
    "obs_groups": {"student": ["student"], "teacher": ["teacher"]},
    "check_for_nan": True,
    "logger": "tensorboard",
    "student": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
                "obs_normalization": True,
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}},
    "teacher": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
                "obs_normalization": True,
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar",
                                     "std_range": [0.1, 0.5]}},
    "algorithm": {"class_name": "Distillation", "num_learning_epochs": 1, "gradient_length": 15,
                  "learning_rate": 5e-4, "max_grad_norm": 1.0, "loss_type": "mse", "optimizer": "adam"},
}


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", choices=("residual", "skills", "recovery", "student"), default="residual")
    ap.add_argument("--teacher", default=None, help="--task student: the skill teacher checkpoint")
    ap.add_argument("--resume", default=None, help="continue from this checkpoint (weights + optimiser)")
    ap.add_argument("--name", default=None, help="run name (default: warp-residual / skill-teacher)")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    import torch
    from rsl_rl.runners import OnPolicyRunner
    if args.task == "skills":
        from skill_env import SkillCfg as EnvCfg, SkillEnv as PrimusWarpEnv
        args.name = args.name or "skill-teacher"
        args.no_eval = True               # eval.py / render.py are for the residual task; see skill_eval.py
    elif args.task == "recovery":
        from recovery_env import RecoveryCfg as EnvCfg, RecoveryEnv as PrimusWarpEnv
        args.name = args.name or "recovery"
        args.no_eval = True
    elif args.task == "student":
        from student_env import StudentCfg as EnvCfg, StudentEnv as PrimusWarpEnv
        args.name = args.name or "skill-student"
        args.no_eval = True
        assert args.teacher, "--task student needs --teacher <checkpoint>"
    else:
        from env import EnvCfg, PrimusWarpEnv
        args.name = args.name or "warp-residual"

    run = WORKSPACE / "experiments" / f"{dt.datetime.now():%Y-%m-%d_%H-%M-%S}-{args.name}"
    run.mkdir(parents=True)
    (run / "tracked-changes.patch").write_text(git("diff", "HEAD"))
    cfg = EnvCfg(num_envs=args.num_envs)
    ppo = copy.deepcopy(DISTILL_CFG if args.task == "student" else PPO_CFG)
    ppo["seed"] = args.seed
    if args.task == "skills":                 # more authority: keep exploring (v1 collapsed to std 0.07)
        ppo["actor"]["distribution_cfg"].update(init_std=0.3, std_range=[0.1, 0.5])
    elif args.task == "recovery":             # not a residual: absolute joint targets, explore widely
        ppo["actor"]["distribution_cfg"].update(init_std=0.5, std_range=[0.1, 0.8])
    manifest = dict(name=run.name, started=dt.datetime.now().isoformat(timespec="seconds"), args=vars(args),
                    git_revision=git("rev-parse", "HEAD").strip(), git_dirty=bool(git("status", "--porcelain").strip()),
                    untracked=git("ls-files", "--others", "--exclude-standard").split(),
                    env_cfg=asdict(cfg), ppo_cfg=ppo, status="training")
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2))
    print(f"run: {run}")

    torch.manual_seed(args.seed)
    env = PrimusWarpEnv(cfg, device=args.device, seed=args.seed)
    if args.task == "student":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, copy.deepcopy(ppo), log_dir=str(run / "logs"), device=args.device)
        runner.load(args.teacher)                      # PPO checkpoint -> teacher weights
        manifest["teacher"] = str(Path(args.teacher).resolve())
        t0 = time.time()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        manifest.update(status="trained", train_seconds=round(time.time() - t0, 1))
        (run / "experiment.json").write_text(json.dumps(manifest, indent=2))
        print(f"done: {run}")
        return
    runner = OnPolicyRunner(env, copy.deepcopy(ppo), log_dir=str(run / "logs"), device=args.device)
    # start exactly from the model-based controller: residual mean = 0 (only the exploration noise)
    head = [m for m in runner.alg.actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    if args.resume:
        runner.load(args.resume)
        manifest["resumed_from"] = str(Path(args.resume).resolve())
    else:
        with torch.no_grad():
            head.weight[:env.num_actions].zero_()
            head.bias[:env.num_actions].zero_()
            if args.task == "skills":
                head.bias[8] = -0.3           # jump trigger: rare random jumps (a > 0.5 ~ 1 per 5 s) to discover them
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    manifest.update(status="trained", train_seconds=round(time.time() - t0, 1),
                    checkpoint=str(sorted((run / "logs").glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]))
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2))
    del runner, env
    torch.cuda.empty_cache()

    if not args.no_eval:
        for script, extra in (("eval.py", []), ("render.py", [])):
            cmd = [sys.executable, str(HERE / script), "--checkpoint", manifest["checkpoint"], "--out", str(run), *extra]
            print("$", " ".join(cmd), flush=True)
            code = subprocess.call(cmd)
            manifest.setdefault("stages", []).append(dict(stage=script, exit_code=code))
        manifest["status"] = "completed" if all(s["exit_code"] == 0 for s in manifest["stages"]) else "failed"
        (run / "experiment.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {run}")


if __name__ == "__main__":
    main()
