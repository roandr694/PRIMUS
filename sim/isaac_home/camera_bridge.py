"""Isaac Sim camera for HomeSim: MuJoCo keeps physics, controller, policy, lidar and the brain; Isaac renders the
D435i (colour + aligned depth for the semantic memory, and the decimated depth stream for the elevation map /
costmap) with RTX, in the same world, from the robot's camera pose (docs/home_autonomy_architecture.md §11).

    cam = IsaacCamera("kujiale_0026")
    sim = HomeSim(world="kujiale_0026", camera=cam, rgbd_hz=2.0)

The MuJoCo collision of an apartment is the voxelised copy of the same USD meshes (sim/apartments/convert.py), so
what the camera sees and what the robot bumps into agree to a few centimetres. Door leaves removed in MuJoCo are
hidden in the Isaac stage too.
"""
import atexit
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")      # pxr (USD) is imported from the Isaac install

HERE = Path(__file__).resolve().parent
PRIMUS = HERE.parents[1]
sys.path.insert(0, str(PRIMUS / "sim" / "home_sim"))
PYTHON = os.environ.get("ISAAC_PYTHON", str(Path.home() / "test" / "isaac-rl" / "bin" / "python"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def door_prims(world):
    """USD paths of the door leaves and handles removed from the apartment's collision."""
    from pxr import Usd
    st = Usd.Stage.Open(str(world.usd))
    out = []
    other = st.GetPrimAtPath("/Root/Meshes/other")
    if other.IsValid():
        for p in other.GetChildren():
            n = p.GetName()
            if n.startswith("door_") and not n.startswith("doorsill"):
                out.append(str(p.GetPath()))
    return out


class IsaacCamera:
    def __init__(self, world="kujiale_0026", log=None, timeout=600.0):
        from worlds import get_world
        self.world = get_world(world)
        self.port = free_port()
        hide = door_prims(self.world) if self.world.name != "cma" else []
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
        env.pop("MUJOCO_GL", None)
        self.log = open(log or os.devnull, "w")
        self.proc = subprocess.Popen([PYTHON, str(HERE / "isaac_server.py"), "--usd", str(self.world.usd),
                                      "--port", str(self.port), "--hide", *hide],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env)
        atexit.register(self.close)
        from multiprocessing.connection import Client
        t0 = time.time()
        while True:
            try:
                self.conn = Client(("127.0.0.1", self.port), authkey=b"primus")
                break
            except (ConnectionRefusedError, OSError):
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Isaac server exited ({self.proc.returncode}); see {log}")
                if time.time() - t0 > timeout:
                    raise TimeoutError("Isaac server did not start")
                time.sleep(1.0)
        self.cams = {}
        self.n, self.t = 0, 0.0

    def _cam(self, key, cam):
        """Camera with the intrinsics of a D435i model: square pixels at cam.fx, rendered tall enough for the
        vertical FOV, then cropped to cam.height."""
        if key not in self.cams:
            h = int(2 * math.ceil(cam.fx * math.tan(math.radians(cam.vfov_deg) / 2)))
            h = max(h, cam.height)
            self.conn.send(("cam", key, cam.width, h, cam.fx, 0.05, 12.0))
            assert self.conn.recv() == "ok"
            self.cams[key] = h
        return self.cams[key]

    def _render(self, sim, key, cam, want_rgb):
        h = self._cam(key, cam)
        pos, R = sim.camera_pose()
        Rc = self._optical_R(sim, cam, R)
        t0 = time.time()
        self.conn.send(("render", key, pos, Rc, want_rgb))
        rgb, depth = self.conn.recv()
        self.t += time.time() - t0
        self.n += 1
        o = (h - cam.height) // 2
        depth = depth[o:o + cam.height]
        if rgb is not None:
            rgb = rgb[o:o + cam.height]
        return rgb, depth

    @staticmethod
    def _optical_R(sim, cam, R):
        """The MuJoCo d435i camera frame (its fixed mount) is used as is: both use -Z forward, +Y up."""
        return R

    def depth(self, sim, cam):
        return self._render(sim, "depth", cam, False)[1]

    def rgbd(self, sim, cam):
        return self._render(sim, "color", cam, True)

    def stats(self):
        return dict(frames=self.n, ms=1000 * self.t / max(self.n, 1))

    def close(self):
        try:
            self.conn.send(("quit",))
        except Exception:
            pass
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
