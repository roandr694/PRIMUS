"""Builds the CMA house scene for MuJoCo (and the data for the Isaac/USD export).

    cd sim
    python3 house/build.py                 # textures (download once) + geometry + sim/models/cena_casa_cma.xml
    python3 house/build.py --no-reflections   # faster rendering (water planes without mirror reflections)
    python3 house/build.py --usd           # + the USD stage for Isaac Sim (sim/isaac/usd/house/cma_house.usd)
    python3 house/render_views.py          # preview images of the generated scene
    python3 scripts/run_xbox_viewer.py --house    (or run_quest_viewer.py --world house)

Everything written here is generated (sim/models/house/, sim/models/cena_casa_cma.xml, the ground truth
sim/models/house/house_gt.json for sim/home_sim) and ignored by Git.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SIM = HERE.parent
MODELS = SIM / "models"

import geom  # noqa: E402
import materials  # noqa: E402
import mjcf  # noqa: E402
import textures  # noqa: E402

SCENE = MODELS / "cena_casa_cma.xml"


def robot_keys():
    """Robot qpos of the 'parked' and 'home' keyframes, read from the generated flat scene."""
    txt = (MODELS / "cena_plana.xml").read_text()
    keys = dict(re.findall(r'<key name="(\w+)" qpos="([^"]+)"', txt))
    return {k: [float(x) for x in v.split()] for k, v in keys.items()}


def yaw_quat(heading):
    """Robot base quaternion for a world heading (rad from +X); the base faces its local +Y."""
    a = (heading - math.pi / 2) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def collect_joints(scene):
    out = []

    def rec(b):
        if b.joint:
            out.append((f"{b.name}_j", scene.joint_defaults.get(b.name, 0.0)))
        for c in b.children:
            rec(c)
    for b in scene.bodies:
        rec(b)
    return out


def lights(s, sun_dir, min_elev_deg=38.0):
    """Sun (shadows) along the sky's sun azimuth; its elevation is raised to at least min_elev_deg so the
    interiors get daylight like in the photographs (the HDRI was shot in the late afternoon)."""
    az = math.atan2(sun_dir[1], sun_dir[0])
    el = max(math.asin(max(-1.0, min(1.0, sun_dir[2]))), math.radians(min_elev_deg))
    import numpy as np
    d = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    s.light(name="sun", directional=True, castshadow=True, pos=(6 - 40 * d[0], 8 - 40 * d[1], 40),
            dir=tuple(d), diffuse=(0.92, 0.88, 0.80), specular=(0.35, 0.33, 0.30), ambient=(0.0, 0.0, 0.0))
    s.light(name="sky_fill", directional=True, castshadow=False, pos=(6, 8, 50), dir=(0.15, -0.1, -1),
            diffuse=(0.22, 0.25, 0.30), specular=(0, 0, 0))
    # ground bounce: light coming from below fakes the inter-reflection that lights ceilings and soffits
    s.light(name="bounce", directional=True, castshadow=False, pos=(6, 8, -5), dir=(-0.2, 0.1, 1),
            diffuse=(0.30, 0.27, 0.22), specular=(0, 0, 0))
    warm = (0.55, 0.47, 0.36)
    for name, pos, att in (("living_lamp", (11.0, 10.0, 6.2), (1, 0.05, 0.02)),
                           ("varanda_lamp", (3.0, 8.0, 2.9), (1, 0.08, 0.04)),
                           ("kitchen_lamp", (10.5, 2.2, 2.9), (1, 0.12, 0.06)),
                           ("upper_lamp", (8.0, 6.0, 6.3), (1, 0.08, 0.03)),
                           ("pavilion_lamp", (-15.5, 7.0, -0.8), (1, 0.15, 0.06))):
        s.light(name=name, directional=False, castshadow=False, pos=pos, dir=(0, 0, -1), diffuse=warm,
                specular=(0.05, 0.05, 0.05), attenuation=att, cutoff=80, exponent=2)


def build(args):
    t0 = time.time()
    print("textures ...")
    if not args.skip_fetch:
        textures.fetch_all()
    tiles = textures.bake(force=args.force_textures)
    mats = materials.catalogue(tiles)
    if args.quality == "realtime":                   # the sea mirror re-renders the whole scene every frame
        mats["water_sea"].reflectance = 0.0
    s = geom.Scene(mats)
    import vegetation
    vegetation.LOD["level"] = "high" if args.quality == "full" else "low"

    import structure
    import semantics
    recorder = semantics.Recorder(s)
    with recorder.active():                          # ground truth for sim/home_sim (rooms, labelled objects)
        print("architecture ...")
        structure.build(s)
        if not args.no_interiors:
            import interiors
            print("interiors ...")
            interiors.furnish(s)
    if not args.no_site:
        import site_landscape
        print("site, terrain, vegetation ...")
        site_landscape.build(s, lite=args.lite)

    print("sky ...")
    hdr = textures.CACHE / "hdri" / "simons_town_rocks_4k.hdr"
    sky_png = textures.OUT / "sky.png"
    sun_dir, sun_elev = mjcf.make_skybox(hdr, sky_png, face=1024 if not args.lite else 512,
                                         sun_azimuth_deg=args.sun_azimuth)
    lights(s, sun_dir)

    rk = robot_keys()
    joints = collect_joints(s)
    stats = mjcf.write(s, MODELS, keyframes=[], sky_png=sky_png, sun_dir=sun_dir,
                       reflections=not args.no_reflections)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    nq_robot = len(rk["parked"])
    keys = []
    order = ["living"] + [k for k in s.spawns if k != "living"]
    for i, name in enumerate(order):
        x, y, zf, heading = s.spawns[name]
        for base_name in (("parked", "home") if i == 0 else ("parked",)):
            q = list(rk[base_name])
            q[0:3] = [x, y, zf + rk[base_name][2]]
            q[3:7] = yaw_quat(heading)
            full = model.qpos0.copy()
            full[:nq_robot] = q
            for jn, val in joints:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                full[model.jnt_qposadr[jid]] = val
            ctrl = [0.0] * model.nu
            for an, val in (("lift_gate_down", 1.45),):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
                if aid >= 0:
                    ctrl[aid] = val
            kname = base_name if (i == 0) else name
            keys.append(dict(name=kname, qpos=full, ctrl=ctrl))
    mjcf._write_scene(SCENE, "robo_bipedal.xml", s, keys, sun_dir)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    info = dict(stats, nq=model.nq, nv=model.nv, nu=model.nu, ngeom=model.ngeom, nbody=model.nbody,
                nmesh=model.nmesh, keyframes=[k["name"] for k in keys], sun_elevation_deg=sun_elev,
                build_seconds=round(time.time() - t0, 1),
                notes={k: v for k, v in s.notes.items() if k != "articulated"},
                articulated=len(s.notes.get("articulated", [])))
    if args.usd:
        import export_usd
        from render_views import VIEWS
        el = max(sun_elev, 38.0)
        az = math.atan2(sun_dir[1], sun_dir[0])
        light_dir = [math.cos(math.radians(el)) * math.cos(az), math.cos(math.radians(el)) * math.sin(az),
                     math.sin(math.radians(el))]
        print("USD export ...")
        info["usd"] = str(export_usd.export(s, hdr, light_dir, mjcf.make_skybox.rotation_deg, VIEWS))
    (MODELS / "house" / "build_info.json").write_text(json.dumps(info, indent=1))
    semantics.write(MODELS / "house" / "house_gt.json", s, recorder)
    print(json.dumps({k: v for k, v in info.items() if k != "notes"}, indent=1))
    print(f"wrote {SCENE}")
    return s, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-interiors", action="store_true")
    ap.add_argument("--no-site", action="store_true")
    ap.add_argument("--quality", choices=("realtime", "full"), default="realtime",
                    help="realtime (default): foliage and reflections sized for the interactive viewers; "
                         "full: dense forest, detailed leaves, sea reflections (for renders, videos, Isaac)")
    ap.add_argument("--lite", action="store_true", help="fast iteration: sparse vegetation, small sky")
    ap.add_argument("--no-reflections", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true", help="do not try to download missing textures")
    ap.add_argument("--force-textures", action="store_true")
    ap.add_argument("--usd", action="store_true", help="also export sim/isaac/usd/house/cma_house.usd for Isaac Sim")
    ap.add_argument("--sun-azimuth", type=float, default=150.0,
                    help="sun azimuth in degrees from +X (east) towards +Y (north); 150 = north-west, afternoon")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
