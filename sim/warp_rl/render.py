"""Videos de uma politica treinada (fisica no MuJoCo Warp; quadros renderizados pelo MuJoCo de CPU, EGL).

  video_overview.mp4  muitos robos no mesmo terreno (como nos videos do Isaac Lab); a camera passa dos
                      niveis faceis para os dificeis. Alguns nascem estacionados e levantam sozinhos.
  video_compare.mp4   em cima: so o controlador model-based (residual zero); embaixo: model-based + RL.
                      Mesmos ladrilhos dificeis, mesmos comandos, empurroes e parametros sorteados.

As poses sao gravadas na GPU durante o rollout e so depois desenhadas; nada e simulado no MuJoCo de CPU.

    MUJOCO_GL=egl ~/test/isaac-rl/bin/python sim/warp_rl/render.py --checkpoint <run>/logs/model_1999.pt --out <run>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as C  # noqa: E402
from eval import load_policy  # noqa: E402

MODE_PT = {0: "estacionado", 1: "levantando", 2: "operando", 3: "pousando"}


def make_env(per_tile, seed, p_parked, device):
    from env import EnvCfg, PrimusWarpEnv
    cfg = EnvCfg(num_envs=36 * per_tile, curriculum=False, random_tile_type=False, p_parked_start=p_parked,
                 episode_s=1e6, spawn_offset=0.25 if per_tile == 1 else 1.0)
    env = PrimusWarpEnv(cfg, device=device, seed=seed)
    idx = torch.arange(env.num_envs, device=env.device)
    tile = idx // per_tile
    env.ttype[:] = tile % 6
    env.level[:] = tile // 6
    env._reset(idx)
    env._compute_obs()
    return env


def record(env, policy, seconds, every=2):
    """Roda e grava qpos/modo/queda de todos os robos a cada `every` passos (50 Hz / 2 = 25 fps)."""
    qs, modes, fell = [], [], []
    obs = env.get_observations()
    falls = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        for k in range(int(seconds / env.dt)):
            a = policy(obs) if policy is not None else torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, _, dones, ex = env.step(a)
            falls |= dones & ~ex["time_outs"]
            if k % every == 0:
                qs.append(env.qpos.cpu().numpy().copy())
                modes.append(env.cs[:, C.S["mode"]].long().cpu().numpy().copy())
                fell.append(falls.cpu().numpy().copy())
                falls[:] = False
    return np.array(qs), np.array(modes), np.array(fell)


class MultiRenderer:
    """Desenha varias poses do mesmo robo numa cena so (terreno uma vez, robos como geoms dinamicos)."""

    def __init__(self, model, w, h):
        self.m, self.d = model, mujoco.MjData(model)
        self.r = mujoco.Renderer(model, h, w, max_geom=40000)
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def frame(self, qposes, lookat, distance, azimuth, elevation):
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = lookat, distance, azimuth, elevation
        for i, q in enumerate(qposes):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            if i == 0:
                self.r.update_scene(self.d, self.cam)
            else:
                mujoco.mjv_addGeoms(self.m, self.d, self.r._scene_option, self.pert,
                                    mujoco.mjtCatBit.mjCAT_DYNAMIC, self.r.scene)
        return self.r.render()

    def close(self):
        self.r.close()


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def label(img, lines, size=22, color=(255, 255, 255)):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = font(size)
    y = 8
    for text in lines:
        dr.text((10, y), text, font=f, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))
        y += size + 6
    return np.asarray(im)


class Video:
    def __init__(self, path, w, h, fps=25):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def add(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def overview(env, policy, out, seconds, tag):
    qs, modes, _ = record(env, policy, seconds)
    w, h = 1280, 720
    mr = MultiRenderer(env.mjm, w, h)
    centers = env.terrain["centers"]          # [nivel, tipo, 2]
    v = Video(out / "video_overview.mp4", w, h)
    T = len(qs)
    for k in range(T):
        s = k / max(T - 1, 1)
        lv = s * (env.cfg.terrain_levels - 1)
        y = np.interp(lv, np.arange(env.cfg.terrain_levels), centers[:, 0, 1])
        x = np.interp(np.sin(2 * np.pi * s) * 0.5 + 0.5, [0, 1], [centers[0, 0, 0] + 4, centers[0, -1, 0] - 4])
        img = mr.frame(qs[k], (x, y, 0.3), 13.0, 90 + 25 * np.sin(np.pi * s), -28)
        n_oper = int((modes[k] == 2).sum())
        img = label(img, [f"{tag} | {env.num_envs} robos no mesmo terreno (MuJoCo Warp, GPU)",
                          f"t = {k / 25:4.1f} s   nivel ~{lv:.1f} de 5   operando: {n_oper}/{env.num_envs}",
                          "tipos (esq. -> dir.): plano, irregular, rampa, ondas, degraus, blocos"], size=20)
        v.add(img)
    v.close()
    mr.close()


def compare(envs, policies, out, seconds, tiles):
    """tiles: lista de (tipo, nivel). Cada coluna segue um robo nesse ladrilho; linha 0 = LQR, 1 = LQR+RL."""
    recs = [record(e, p, seconds) for e, p in zip(envs, policies)]
    pw, ph = 640, 400
    mr = MultiRenderer(envs[0].mjm, pw, ph)
    types = envs[0].terrain["types"]
    names = {"flat": "plano", "rough": "irregular", "slope": "rampa", "waves": "ondas", "steps": "degraus", "blocks": "blocos"}
    v = Video(out / "video_compare.mp4", pw * len(tiles), ph * 2)
    falls = np.zeros((2, len(tiles)), int)
    per_tile = envs[0].num_envs // 36
    for k in range(len(recs[0][0])):
        rows = []
        for r, (qs, modes, fell) in enumerate(recs):
            panels = []
            for c, (t, lv) in enumerate(tiles):
                i = (lv * 6 + t) * per_tile
                falls[r, c] += int(fell[k, i])
                q = qs[k, i]
                yaw = np.degrees(2 * np.arctan2(q[6], q[3]))
                img = mr.frame([q], (q[0], q[1], q[2] - 0.15), 2.4, yaw + 90 + 35, -18)
                head = "SO LQR + VMC" if r == 0 else "LQR + VMC + RL residual"
                img = label(img, [head, f"{names[types[t]]} nivel {lv}  |  {MODE_PT[int(modes[k, i])]}",
                                  f"quedas: {falls[r, c]}"], size=20,
                            color=(255, 210, 120) if r == 0 else (150, 230, 255))
                panels.append(img)
            rows.append(np.concatenate(panels, 1))
        v.add(np.concatenate(rows, 0))
    v.close()
    mr.close()
    return falls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)

    env = make_env(per_tile=6, seed=args.seed, p_parked=0.3, device=args.device)
    overview(env, load_policy(env, args.checkpoint, args.device), out, args.seconds, "LQR + VMC + RL residual")
    del env
    torch.cuda.empty_cache()

    tiles = [(1, 5), (4, 5), (5, 5)]           # irregular, degraus e blocos no nivel mais dificil
    envs = [make_env(per_tile=1, seed=args.seed + 1, p_parked=0.0, device=args.device) for _ in range(2)]
    falls = compare(envs, [None, load_policy(envs[1], args.checkpoint, args.device)], out, args.seconds, tiles)
    print("quedas nos paineis (linha 0 = so LQR, linha 1 = LQR + RL):", falls.tolist())
    print("videos:", out / "video_overview.mp4", out / "video_compare.mp4")


if __name__ == "__main__":
    main()
