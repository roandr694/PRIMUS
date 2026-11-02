"""Trains the residual policy on the GPU (MuJoCo Warp + RSL-RL PPO).

    ~/test/isaac-rl/bin/python sim/warp_rl/train.py --num-envs 4096 --iterations 1500

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
    ap.add_argument("--name", default="warp-residual")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from env import EnvCfg, PrimusWarpEnv

    run = WORKSPACE / "experiments" / f"{dt.datetime.now():%Y-%m-%d_%H-%M-%S}-{args.name}"
    run.mkdir(parents=True)
    (run / "tracked-changes.patch").write_text(git("diff", "HEAD"))
    cfg = EnvCfg(num_envs=args.num_envs)
    ppo = copy.deepcopy(PPO_CFG)
    ppo["seed"] = args.seed
    manifest = dict(name=run.name, started=dt.datetime.now().isoformat(timespec="seconds"), args=vars(args),
                    git_revision=git("rev-parse", "HEAD").strip(), git_dirty=bool(git("status", "--porcelain").strip()),
                    untracked=git("ls-files", "--others", "--exclude-standard").split(),
                    env_cfg=asdict(cfg), ppo_cfg=ppo, status="training")
    (run / "experiment.json").write_text(json.dumps(manifest, indent=2))
    print(f"run: {run}")

    torch.manual_seed(args.seed)
    env = PrimusWarpEnv(cfg, device=args.device, seed=args.seed)
    runner = OnPolicyRunner(env, copy.deepcopy(ppo), log_dir=str(run / "logs"), device=args.device)
    # start exactly from the model-based controller: residual mean = 0 (only the exploration noise)
    head = [m for m in runner.alg.actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        head.weight[:env.num_actions].zero_()
        head.bias[:env.num_actions].zero_()
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
