"""Top view of a saved robot map: lidar occupancy, the camera's voxel obstacles, rooms and objects.

    python3 sim/home_sim/view_map.py sim/models/apartments/kujiale_0026/map_vox --world kujiale_0026 --png out.png

Colours: light grey free (lidar), white free seen by the camera only, black lidar obstacle, orange camera obstacle
(between the wheels' step and the robot's top) the lidar does not see, blue outline the ground-truth obstacles,
coloured dots the confirmed objects of the memory.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))


def render(stack, world=None, robot_top=0.75, scale=3, gt_grid=None):
    from PIL import Image, ImageDraw
    fid = stack.slam.floor or next(iter(stack.slam.floors))
    g = stack.slam.floors[fid].grid
    vm = stack.voxels.get(fid)
    bb = g.known_bbox(pad=0.6)
    if bb is None:
        return None
    (x0, y0), (x1, y1) = bb
    i0, j0 = g.ij(np.array([x0, y0]))
    i1, j1 = g.ij(np.array([x1, y1]))
    sl = (slice(i0, i1), slice(j0, j1))
    img = np.full((i1 - i0, j1 - j0, 3), 110, np.uint8)
    img[g.free()[sl]] = (205, 205, 205)
    if vm is not None:
        img[vm.observed()[sl] & ~g.occupied()[sl]] = (235, 235, 235)
    img[g.occupied()[sl]] = (20, 20, 20)
    if vm is not None:
        cam = vm.columns(0.06, robot_top)[sl] & ~g.occupied()[sl]
        img[cam] = (235, 140, 30)
    if gt_grid is not None:
        from scipy import ndimage
        xs = g.xy(np.arange(i0, i1)[:, None].repeat(j1 - j0, 1), np.arange(j0, j1)[None, :].repeat(i1 - i0, 0))
        gi, gj = gt_grid.world_to_ij(xs)
        ok = gt_grid.inside(gi, gj)
        occ = np.zeros(ok.shape, bool)
        occ[ok] = ~gt_grid.free[gi[ok], gj[ok]]
        edge = occ & ~ndimage.binary_erosion(occ)
        img[edge] = (40, 90, 230)
    im = Image.fromarray(img[::-1]).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    H = img.shape[0]

    def px(xy):
        i, j = g.ij(np.asarray(xy))
        return ((j - j0 + 0.5) * scale, (H - 1 - (i - i0) + 0.5) * scale)
    if stack.memory is not None:
        for o in stack.memory.confirmed(fid):
            u, v = px(o.centroid[:2])
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(200, 30, 120))
            dr.text((u + 5, v - 5), o.cls, fill=(120, 0, 70))
    if stack.scene is not None and fid in stack.scene.rooms:
        for r in stack.scene.rooms[fid]:
            u, v = px(r.centroid)
            dr.text((u, v), r.label or r.name, fill=(0, 90, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--world", default=None)
    ap.add_argument("--png", required=True)
    a = ap.parse_args()
    from brain import persist
    stack = persist.load_stack(a.map)
    gt = None
    if a.world:
        from gt_grid import GTGrid
        from worlds import get_world
        w = get_world(a.world)
        f = w.gt_dir / f"gt_grid_{w.default_floor}_doors_open.npz"
        gt = GTGrid.load(f) if f.exists() else None
    render(stack, gt_grid=gt).save(a.png)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
