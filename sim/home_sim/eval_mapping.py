"""P1 test: tour one or more floors with the robot's own SLAM and score it against the ground truth.

    MUJOCO_GL=egl python3 sim/home_sim/eval_mapping.py --floor social [--save-map maps/cma] [--png out.png]

Scores: pose error (map pose vs true pose, the map frame starts at the true spawn pose), map precision (occupied
map cells within 10 cm of a ground-truth obstacle) and coverage (free ground-truth cells the map knows as free).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from brain.mapping.slam import FloorSLAM  # noqa: E402
from brain.runtime import Brain  # noqa: E402
from runner import Scenario, run  # noqa: E402
from tour import TourDriver  # noqa: E402
from world import HomeSim  # noqa: E402


class PoseLogger:
    """Wraps a driver to record (true pose, map pose) at 2 Hz."""

    def __init__(self, inner, brain, sim):
        self.inner, self.brain, self.sim, self.log, self.n = inner, brain, sim, [], 0
        for k in ("message", "on_rgbd"):
            if hasattr(inner, k):
                setattr(self, k, getattr(inner, k))

    def start(self, sim, brain):
        self.inner.start(sim, brain)

    def step(self, t, brain, ev):
        self.n += 1
        if self.n % 25 == 0 and ev.state.mode == 2:
            tp = self.sim.truth()
            self.log.append((t, tp.p[0], tp.p[1], tp.yaw + math.pi / 2, *brain.map_pose(), ev.state.x, ev.state.y))
        out = self.inner.step(t, brain, ev)
        self.message = getattr(self.inner, "message", "")
        return out

    def done(self):
        return self.inner.done()

    def extra(self):
        return self.inner.extra() if hasattr(self.inner, "extra") else {}


def score_map(slam, floor_id, gt):
    g = slam.floors[floor_id].grid
    occ_gt = ~gt.free
    dist_gt = ndimage.distance_transform_edt(~occ_gt) * gt.res               # distance to a GT obstacle
    occ = g.occupied()
    ii, jj = np.nonzero(occ)
    xy = g.xy(ii, jj)
    gi, gj = gt.world_to_ij(xy)
    ok = gt.inside(gi, gj)
    prec = float(np.mean(dist_gt[gi[ok], gj[ok]] <= 0.10)) if ok.any() else 0.0
    # coverage: GT free cells (with clearance) the map marks free
    fi, fj = np.nonzero(gt.free & (ndimage.distance_transform_edt(gt.free) * gt.res > 0.35))
    fxy = gt.ij_to_world(fi, fj)
    mi, mj = g.ij(fxy)
    okm = g.inside(mi, mj)
    cov = float(np.mean(g.free()[mi[okm], mj[okm]])) if okm.any() else 0.0
    return prec, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="social")
    ap.add_argument("--spawn", default="home")
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--save-map", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    sim = HomeSim(spawn=a.spawn)
    slam = FloorSLAM(mapping=True)
    brain = Brain(slam=slam)
    drv = PoseLogger(TourDriver(sim, a.floor, order=a.rooms), brain, sim)
    res, sim = run(Scenario(f"tour {a.floor}", spawn=a.spawn, timeout=a.timeout), drv, sim=sim, brain=brain,
                   video=a.video, verbose=False)
    print(res.line())
    L = np.array(drv.log)
    err = np.hypot(L[:, 1] - L[:, 4], L[:, 2] - L[:, 5])
    odo = np.hypot(L[:, 1] - L[:, 7], L[:, 2] - L[:, 8])
    eth = np.abs((L[:, 3] - L[:, 6] + np.pi) % (2 * np.pi) - np.pi)
    print(f"pose error: mean {err.mean() * 100:.1f} cm, max {err.max() * 100:.1f} cm, heading max "
          f"{math.degrees(eth.max()):.1f} deg; raw odometry error at the end {odo[-1] * 100:.1f} cm (max {odo.max() * 100:.1f})")
    print("slam stats:", slam.stats, "floors:", {k: (f.z, f.n_scans) for k, f in slam.floors.items()})
    import gt_grid
    from world import GT_PATH
    gt = gt_grid.cached(sim.m, sim.d, a.floor, GT_PATH.parent)
    fid = min(slam.floors, key=lambda k: abs(slam.floors[k].z - gt.floor_z))
    prec, cov = score_map(slam, fid, gt)
    print(f"map: precision {prec * 100:.1f} % (occupied cells within 10 cm of a real obstacle), "
          f"coverage {cov * 100:.1f} % of the free floor; rooms visited {res.extra.get('visited')}, failed {res.extra.get('failed')}")
    if a.save_map:
        slam.save(a.save_map)
        print("map saved to", a.save_map)
    if a.png:
        save_png(a.png, slam.floors[fid].grid, gt, L)
    sim.close()


def save_png(path, g, gt, L):
    from PIL import Image
    lo = gt.origin
    hi = gt.origin + np.array(gt.free.shape[::-1]) * gt.res
    i0, j0 = g.ij(lo)
    i1, j1 = g.ij(hi)
    sub = g.L[i0:i1, j0:j1]
    img = np.full(sub.shape + (3,), 128, np.uint8)
    img[sub < -0.5] = (235, 235, 235)
    img[sub > 0.8] = (20, 20, 20)
    gfree = gt.free[:sub.shape[0], :sub.shape[1]]
    img[(~gfree) & (sub <= 0.8)] = np.minimum(img[(~gfree) & (sub <= 0.8)].astype(int) + [60, 0, 0], 255)
    for r in L:
        for (x, y), col in (((r[1], r[2]), (0, 160, 0)), ((r[4], r[5]), (0, 0, 255))):
            a, b = g.ij(np.array([x, y]))
            a, b = a - i0, b - j0
            if 0 <= a < img.shape[0] and 0 <= b < img.shape[1]:
                img[a, b] = col
    Image.fromarray(img[::-1]).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)
    print("png:", path)


if __name__ == "__main__":
    main()
