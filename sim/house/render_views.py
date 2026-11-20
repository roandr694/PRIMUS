"""Renders preview images of the CMA house scene (headless, EGL).

    MUJOCO_GL=egl python3 house/render_views.py [--out DIR] [--views name,name] [--width 1600 --height 900]

Views are named camera poses (look-at point, distance, azimuth, elevation) chosen to match the photographs.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "models" / "cena_casa_cma.xml"

# name: (lookat, distance, azimuth, elevation, fovy)
VIEWS = {  # azimuth = direction the camera looks, degrees from +X (east) towards +Y (north)
    "aerial_sea": ((4.0, 8.0, 0.0), 75.0, 0.0, -30.0, 45),          # from the sea, like photo 5
    "aerial_top": ((4.0, 8.0, 0.0), 105.0, 90.0, -89.9, 40),        # straight down, like photo 1
    "aerial_ne": ((10.0, 8.0, 2.0), 80.0, -140.0, -32.0, 45),       # from the hill over the parking
    "sea_deck": ((-16.0, 8.0, -1.0), 15.0, 0.0, -6.0, 65),          # photo 13
    "pool_house": ((1.0, 8.0, 2.5), 22.0, 0.0, -14.0, 60),          # photo 5 closer
    "varanda": ((3.2, 3.0, 1.3), 9.5, -90.0, -6.0, 70),             # photo 8
    "living": ((14.5, 13.0, 1.9), 6.5, 10.0, -4.0, 75),             # photo 12
    "stair_top": ((14.9, 9.0, 2.6), 6.0, 90.0, -25.0, 75),          # photo 9
    "kitchen": ((3.2, 1.2, 1.1), 5.5, -90.0, -8.0, 70),             # photo 10
    "upper_lounge": ((11.0, 8.0, 2.0), 8.85, 127.6, -22.0, 75),     # photo 14
    "garden": ((17.7, 10.0, 4.9), 7.5, 180.0, -6.0, 75),            # photo 11
    "lift": ((27.3, 17.0, 4.8), 6.5, 90.0, -8.0, 65),              # photo 6
    "suite": ((-1.6, 7.6, 4.8), 3.8, 190.0, -12.0, 80),
    "pavilion_in": ((-13.6, 10.2, -2.0), 3.6, 0.0, -10.0, 80),
    "robot": ((12.9, 9.0, 0.35), 2.4, 60.0, -12.0, 60),            # PRIMUS parked in the living room
}


def render(model, data, view, width, height):
    lookat, dist, az, el, fovy = view
    model.vis.global_.fovy = fovy
    r = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    r.update_scene(data, cam, opt)
    img = r.render()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "house" / "previews"))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--key", default="parked")
    a = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height)
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, a.key)
    mujoco.mj_resetDataKeyframe(model, data, max(kid, 0))
    mujoco.mj_forward(model, data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.views.split(","):
        img = render(model, data, VIEWS[name], a.width, a.height)
        Image.fromarray(img).save(out / f"{name}.jpg", quality=90)
        print(out / f"{name}.jpg")


if __name__ == "__main__":
    main()
