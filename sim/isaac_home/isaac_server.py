"""Isaac Sim render server: the robot's D435i rendered by RTX inside a realistic apartment (or the CMA house).

Started by camera_bridge.IsaacCamera in its own process (Isaac's SimulationApp owns the process and the GPU
context, MuJoCo keeps the physics in the client):

    OMNI_KIT_ACCEPT_EULA=YES ~/test/isaac-rl/bin/python sim/isaac_home/isaac_server.py --usd <stage> --port 6100

Protocol (multiprocessing.connection, pickled tuples):
    ("cam", name, width, height, fx, near, far)   -> "ok"      create a camera + render product (rgb, depth)
    ("render", name, pos[3], R[3,3], want_rgb)    -> (rgb uint8 [h, w, 3] or None, depth float32 [h, w])
         pos / R: world pose of the camera in the MuJoCo convention (-Z forward, +Y up; R columns = camera axes),
         which is also the USD camera convention
    ("hide", [prim paths])                        -> "ok"
    ("quit",)
Depth is distance to the image plane (like MuJoCo's depth buffer); 0 where nothing is hit within `far`.
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--hide", nargs="*", default=[])
    ap.add_argument("--renderer", default="RaytracedLighting")
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": a.renderer, "anti_aliasing": 0, "width": 640, "height": 360})
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    omni.usd.get_context().open_stage(a.usd)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        app.update()
    stage = omni.usd.get_context().get_stage()
    for p in a.hide:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    # no physics needed: Isaac only renders. Keep the timeline stopped.
    rep.orchestrator.set_capture_on_play(False)
    cams = {}

    def make_cam(name, w, h, fx, near, far):
        path = f"/World/PrimusCam_{name}"
        cam = UsdGeom.Camera.Define(stage, Sdf.Path(path))
        hap = 20.955
        cam.CreateHorizontalApertureAttr(hap)
        cam.CreateVerticalApertureAttr(hap * h / w)
        cam.CreateFocalLengthAttr(hap * fx / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(near, far))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        rp = rep.create.render_product(path, (w, h))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach(rp)
        dep.attach(rp)
        cams[name] = dict(op=op, rp=rp, rgb=rgb, dep=dep, w=w, h=h, far=far)

    def set_pose(c, pos, R):
        M = Gf.Matrix4d(1.0)
        for i in range(3):
            for j in range(3):
                M[i, j] = float(R[j, i])          # USD rows = camera axes (row-vector convention)
            M[3, i] = float(pos[i])
        c["op"].Set(M)

    def set_active(name):
        for n, c in cams.items():
            c["rp"].hydra_texture.set_updates_enabled(n == name)

    from multiprocessing.connection import Listener
    lst = Listener(("127.0.0.1", a.port), authkey=b"primus")
    print(f"[isaac_server] ready on {a.port}", flush=True)
    conn = lst.accept()
    warm = {}
    n_frames, t_render = 0, 0.0
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "quit":
            break
        if kind == "cam":
            make_cam(*msg[1:])
            conn.send("ok")
        elif kind == "hide":
            for p in msg[1]:
                prim = stage.GetPrimAtPath(p)
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
            conn.send("ok")
        elif kind == "render":
            name, pos, R, want_rgb = msg[1:]
            c = cams[name]
            t0 = time.time()
            set_active(name)
            set_pose(c, pos, R)
            steps = 1 if warm.get(name) else a.warmup                 # first frames: shaders / textures load
            for _ in range(steps):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            warm[name] = True
            d = np.asarray(c["dep"].get_data(), np.float32).reshape(c["h"], c["w"])
            d = np.where(np.isfinite(d) & (d > 0) & (d < c["far"]), d, 0.0).astype(np.float32)
            rgb = None
            if want_rgb:
                rgb = np.asarray(c["rgb"].get_data())[..., :3].astype(np.uint8).copy()
            t_render += time.time() - t0
            n_frames += 1
            conn.send((rgb, d))
        elif kind == "stats":
            conn.send(dict(frames=n_frames, ms=1000 * t_render / max(n_frames, 1)))
    conn.close()
    app.close()


if __name__ == "__main__":
    main()
