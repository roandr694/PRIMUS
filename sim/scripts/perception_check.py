"""Detection accuracy of the perception stack on the skill terrain (kinematic robot, no physics).

The robot glides toward each feature from 2.5 m to 0.6 m at 0.5 m/s, legs at 0.30 m; the rendered D435i
(with noise) feeds the elevation map at 15 Hz; the detector output at the end is compared with the terrain:

    MUJOCO_GL=egl python3 scripts/perception_check.py
"""
import math
import sys

import numpy as np

import skill_world as SW
import mujoco
import five_bar
from perception.elevation_map import ElevationMap
from perception.features import FeatureDetector

CASES = (("stairs", "foot", "stairs_up", lambda lv: SW.ST.STAIR_RISE[lv], "rise"),
         ("stairs", "top", "stairs_down", lambda lv: SW.ST.STAIR_RISE[lv], "rise"),
         ("curbs", "top", "curb_down", lambda lv: SW.ST.CURB_RISE[lv], "rise"),
         ("gaps", "top", "gap", lambda lv: SW.ST.GAP_WIDTH[lv], "width"),
         ("overhang", "top", "overhang", lambda lv: SW.ST.BAR_BOTTOM[lv], "free"))


def feature_distance(tile, start):
    """distance from the start point to the feature edge along the start heading"""
    return {("stairs", "foot"): 0.9, ("stairs", "top"): 1.1, ("curbs", "top"): 0.8,
            ("gaps", "top"): 0.9, ("overhang", "top"): 1.45}[(tile, start)]


def main():
    ter = SW.ST.make_terrain(seed=0)
    _, spec = SW.M.build(ter, collision="full")
    sens = SW.SimSensors(spec)
    m = sens.model
    d = mujoco.MjData(m)
    q, z = five_bar.nominal_pose(0.30, wheel_radius=0.10751, clearance=0.0)
    det = FeatureDetector()
    print(f"{'feature':12s} {'level':>5s} {'true':>6s} {'detected':>9s} {'at [m]':>7s}  (true distance 0.60 m)  first feature seen")
    for tile, start, kind, truth, attr in CASES:
        for lv in range(6):
            emap = ElevationMap()
            xy0, h0 = SW.start_pose(ter, tile, lv, start)
            fwd = np.array([-math.sin(h0), math.cos(h0)])
            D = feature_distance(tile, start)
            ground0 = 0.0 if tile != "stairs" or start == "foot" else 5 * SW.ST.STAIR_RISE[lv]
            if tile == "curbs":
                ground0 = SW.ST.CURB_RISE[lv]
            back = max(0.0, 2.5 - D)
            for s in np.arange(-back, D - 0.6 + 1e-6, 0.5 / 15):
                d.qpos[:] = SW.M.parked_qpos()
                for name, v in q.items():
                    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if j >= 0:
                        d.qpos[m.jnt_qposadr[j]] = v
                d.qpos[0:2] = xy0 + s * fwd
                d.qpos[2] = z + ground0
                d.qpos[3:7] = [math.cos(h0 / 2), 0, 0, math.sin(h0 / 2)]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_camlight(m, d)                 # camera pose for the renderer
                pts, sig = sens.depth_points(d)
                emap.move_to(d.qpos[0:2])
                emap.integrate(pts, sig)
            feats = det.detect(emap, d.qpos[0:2], h0, 0.30, +1)
            hit = next((f for f in feats if f.kind == kind), None)
            got = abs(getattr(hit, attr)) if hit else float("nan")
            print(f"{kind:12s} {lv:5d} {truth(lv):6.2f} {got:9.2f} {hit.s if hit else float('nan'):7.2f}  "
                  f"{feats[0].text() if feats else '-'}{' (n=%d)' % hit.n if hit and hit.n > 1 else ''}")


if __name__ == "__main__":
    main()
