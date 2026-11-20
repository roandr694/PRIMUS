"""P2 test: point-to-point navigation on a floor with the tour map (localisation mode), scored on the truth.

    MUJOCO_GL=egl python3 sim/home_sim/eval_nav.py --map sim/models/house/robot_map --goals 20 [--video v.mp4]

Goals: random free cells (>= 0.5 m from obstacles in the ground truth) inside the indoor rooms of the floor; each
goal is a separate episode started from the previous goal (the robot keeps driving). Success: the true position
within 0.35 m of the goal when the navigator reports "arrived"; also counted: falls, collisions, time, replans.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from brain import persist  # noqa: E402
from runner import Scenario, run_stack  # noqa: E402
import gt_grid  # noqa: E402
from world import GT_PATH, HomeSim  # noqa: E402


class GoalTask:
    def __init__(self, goals, per_goal=75.0, tol=0.35):
        self.goals, self.per_goal, self.tol = goals, per_goal, tol
        self.results, self.message = [], ""

    def start(self, sim, stack):
        self.sim, self.k, self.t0 = sim, -1, None

    def _next(self, now, stack):
        self.k += 1
        if self.k < len(self.goals):
            name, xy = self.goals[self.k]
            stack.nav.go_to(xy, label=name)
            self.t0 = now

    def step(self, now, stack, ev):
        if ev.state.mode != 2:
            return
        if self.t0 is None:
            self._next(now, stack)
            return
        st = stack.nav.status
        if st == "arrived" or st.startswith("failed") or now - self.t0 > self.per_goal:
            name, xy = self.goals[self.k]
            err = float(np.linalg.norm(self.sim.truth().xy - xy))
            ok = st == "arrived" and err <= self.tol
            self.results.append(dict(goal=name, xy=[round(float(v), 2) for v in xy], ok=ok, status=st,
                                     err=round(err, 3), time=round(now - self.t0, 1),
                                     loc_err=round(float(np.linalg.norm(self.sim.truth().xy - np.array(stack.pose()[:2]))), 3)))
            print(f"  goal {self.k + 1}/{len(self.goals)} {name:14s} {'OK ' if ok else 'BAD'} {st:24s} err={err:.2f} m "
                  f"t={now - self.t0:5.1f}s loc_err={self.results[-1]['loc_err']:.2f}")
            self._next(now, stack)
        self.message = stack.nav.message

    def done(self):
        if self.k >= len(self.goals):
            n = sum(r["ok"] for r in self.results)
            return (n == len(self.goals), f"{n}/{len(self.goals)} goals")
        return None

    def extra(self):
        return dict(goals=self.results)


def random_goals(sim, floor, n, seed, min_clear=0.5):
    gt = json.loads(GT_PATH.read_text())
    grid = gt_grid.cached(sim.m, sim.d, floor, GT_PATH.parent)
    clear = grid.clearance()
    rng = np.random.default_rng(seed)
    rooms = [r for r in gt["rooms"] if r["floor"] == floor and r["indoor"]]
    out = []
    while len(out) < n:
        r = rooms[rng.integers(len(rooms))]
        x0, x1, y0, y1 = r["rects"][0]
        xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        i, j = grid.world_to_ij(xy)
        if grid.inside(i, j) and clear[i, j] >= min_clear:
            out.append((r["name"], xy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE.parents[1] / "sim/models/house/robot_map_sem"))
    ap.add_argument("--floor", default="social")
    ap.add_argument("--spawn", default="home")
    ap.add_argument("--goals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sim = HomeSim(spawn=a.spawn)
    goals = random_goals(sim, a.floor, a.goals, a.seed)
    stack = persist.load_stack(a.map, memory=False)          # floor maps + the camera layer of the tour
    tp = sim.truth()
    stack.on_state(sim.step().state)
    stack.slam.set_pose(tp.p[0], tp.p[1], tp.yaw + math.pi / 2)      # known start (relocalisation: eval_reloc)
    task = GoalTask(goals)
    res, sim = run_stack(Scenario("nav", spawn=a.spawn, timeout=a.goals * 80.0), stack, task, sim=sim, video=a.video,
                         verbose=False)
    print(res.line())
    R = res.extra["goals"]
    ok = [r for r in R if r["ok"]]
    print(f"success {len(ok)}/{len(R)}; mean time {np.mean([r['time'] for r in ok]) if ok else 0:.1f} s; "
          f"falls {res.falls}; collisions {res.collisions} {res.extra['collisions'][:6]}; nav stats {stack.nav.stats}")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(line=res.line(), goals=R), indent=1))
    sim.close()


if __name__ == "__main__":
    main()
