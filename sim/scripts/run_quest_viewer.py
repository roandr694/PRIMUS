"""Pilot the simulated robot with a Meta Quest 3 over Wi-Fi (Linux and macOS).

The computer runs MuJoCo + the controller in real time and serves a WebXR page over HTTPS. In the Quest
browser the page shows a large screen with the robot's first-person camera (chase camera in the corner),
a pilot panel, and sends the Quest controllers back.

    cd sim
    python3 scripts/run_quest_viewer.py [--flat]

On the Quest (same Wi-Fi): browser -> the https://<ip>:8443 address printed here -> accept the self-signed
certificate warning (Advanced -> proceed) -> "Enter VR". The same page works in a desktop browser with the
keyboard (W/S, A/D, E/Q, space, P, R) for testing without the headset.

Quest controls: left stick = speed, right stick = turn, triggers = leg height (latches), A = hard brake,
B = soft stop, X = land <-> stand up, Y = restart, both grips = re-centre the screen.
Needs: mujoco, numpy, pillow, aiohttp, and the system `openssl` (self-signed certificate, made once in
~/.cache/primus-quest).
"""
import argparse
import asyncio
import io
import json
import math
import os
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering without a window (macOS default works)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import robot_model as rm  # noqa: E402
from pilot import Pilot  # noqa: E402
from repo_control_policy import MAX_LEG_LENGTH, MIN_LEG_LENGTH, RepoControlPolicy, body_rpy  # noqa: E402

WEB = Path(__file__).resolve().parent / "quest_web" / "index.html"
CERT_DIR = Path.home() / ".cache" / "primus-quest"


def local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))     # sends nothing; just picks the LAN interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def ensure_cert(ip):
    """Self-signed certificate for the LAN IP (WebXR on the Quest requires HTTPS)."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, cfg = CERT_DIR / f"cert-{ip}.pem", CERT_DIR / f"key-{ip}.pem", CERT_DIR / "openssl.cnf"
    if not cert.exists():
        cfg.write_text("[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=PRIMUS sim\n"
                       f"[ext]\nsubjectAltName=IP:{ip},DNS:localhost\n")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout",
                        str(key), "-out", str(cert), "-config", str(cfg)], check=True, capture_output=True)
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(cert, key)
    return ctx


class Shared:
    """State shared by the simulation (main thread) and the web server (asyncio thread)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inputs, self.inputs_time = {}, 0.0
        self.frame, self.frame_seq, self.telemetry, self.clients = None, 0, {}, 0


def build_model(flat):
    """Project scene + a first-person camera at the front of the chassis (looking +Y, 12 deg down)."""
    spec = mujoco.MjSpec.from_file(str(rm.FLAT_MODEL_PATH if flat else rm.MODEL_PATH))
    tilt = math.radians(12.0)
    spec.body("base").add_camera(name="fpv", pos=[0.0, 0.19, 0.11], fovy=95.0,
                                 xyaxes=[1, 0, 0, 0, math.sin(tilt), math.cos(tilt)])
    model = spec.compile()
    data = mujoco.MjData(model)
    rm.reset(mujoco, model, data)
    return model, data


def simulate(shared, args):
    """Real-time physics + controller; renders and JPEG-encodes the video while a pilot is connected."""
    model, data = build_model(args.flat)
    policy = RepoControlPolicy(mujoco, model)
    pilot = Pilot(args.max_speed, args.max_yaw_rate)
    W, H = args.width, args.height
    pw, ph = W * 3 // 10, H * 3 // 10
    fpv, chase = mujoco.Renderer(model, H, W), mujoco.Renderer(model, ph, pw)
    cam = mujoco.MjvCamera()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    dt = model.opt.timestep

    def restart():
        rm.reset(mujoco, model, data)
        policy.reset(data)
        return data.time, time.perf_counter()

    sim_t0, wall_t0 = restart()
    next_frame = next_tel = 0.0
    fallen_since, falls, out, rtf, rtf_win = None, 0, {}, 1.0, (time.perf_counter(), data.time)
    while True:
        wall = time.perf_counter() - wall_t0
        sim_t0 = min(sim_t0, data.time - wall + 0.25)      # computer too slow: lag at most 0.25 s, don't catch up
        steps = 0
        while data.time - sim_t0 < wall and steps < 50:     # catch up with real time, then render
            with shared.lock:
                inp, age = shared.inputs, time.perf_counter() - shared.inputs_time
            cmd, reset = pilot.command(inp, dt, stale=age > 0.5)     # no headset data for 0.5 s: stop
            if reset:
                (sim_t0, wall_t0), fallen_since = restart(), None
                break
            out = policy.apply(data, cmd)
            mujoco.mj_step(model, data)
            steps += 1

        roll, pitch, _ = body_rpy(data)
        if abs(pitch) > math.radians(50) or abs(roll) > math.radians(50) or data.qpos[2] < 0.15:
            if fallen_since is None:
                fallen_since, falls = time.perf_counter(), falls + 1
            elif time.perf_counter() - fallen_since > 4.0:      # fell: restart by itself
                (sim_t0, wall_t0), fallen_since = restart(), None
                continue
        else:
            fallen_since = None

        now = time.perf_counter()
        if now - rtf_win[0] > 1.0:
            rtf, rtf_win = (data.time - rtf_win[1]) / (now - rtf_win[0]), (now, data.time)
        if shared.clients and now >= next_frame:
            next_frame = now + 1.0 / args.fps
            fpv.update_scene(data, camera="fpv")
            img = fpv.render().copy()
            R = data.xmat[base].reshape(3, 3)
            cam.lookat[:] = data.xpos[base] + np.array([0, 0, 0.1])
            cam.distance, cam.azimuth, cam.elevation = 1.8, math.degrees(math.atan2(R[1, 1], R[0, 1])), -18.0
            chase.update_scene(data, cam)
            img[H - ph - 14:H - 10, W - pw - 14:W - 10] = 255                 # frame of the chase view
            img[H - ph - 12:H - 12, W - pw - 12:W - 12] = chase.render()
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=args.quality)
            with shared.lock:
                shared.frame, shared.frame_seq = buf.getvalue(), shared.frame_seq + 1
        if now >= next_tel and out:
            next_tel, c = now + 1.0 / 15, out["controls"]
            r3 = lambda v: round(float(v), 3)
            with shared.lock:
                shared.telemetry = dict(
                    mode=out["mode"], gear_down=bool(out["gear_down"]), t=r3(data.time), rtf=r3(rtf),
                    speed=r3(out["dx"]), speed_cmd=r3(cmd.speed_cmd), speed_target=r3(out["target_speed"]),
                    yaw_rate=r3(data.sensordata[policy.imu.adr_g + 2]), yaw_cmd=r3(cmd.yaw_speed_cmd),
                    leg=r3(0.5 * (out["left_leg_length"] + out["right_leg_length"])), leg_cmd=r3(pilot.leg_length),
                    leg_min=MIN_LEG_LENGTH, leg_max=MAX_LEG_LENGTH, pitch=r3(math.degrees(pitch)), roll=r3(math.degrees(roll)),
                    hip_torque=r3(max(abs(c[k]) for k in c if k.startswith("hip"))),
                    wheel_torque=r3(max(abs(c["wheel_xp"]), abs(c["wheel_xm"]))), torque_limit=18.0,
                    contact_l=r3(out["tire_force"]["xm"]), contact_r=r3(out["tire_force"]["xp"]),
                    park=pilot.park, brake=cmd.brake, fallen=fallen_since is not None, falls=falls,
                    pilot=time.perf_counter() - shared.inputs_time < 0.5)
        if steps == 0:
            time.sleep(0.0005)


def serve(shared, port, ssl_ctx, ready):
    """HTTPS page + one WebSocket per pilot: video (binary JPEG), telemetry and pong (text) out; inputs in."""
    from aiohttp import WSMsgType, web

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=1 << 20, heartbeat=10)
        await ws.prepare(request)
        shared.clients += 1
        print(f"pilot connected: {request.remote}")

        async def sender():
            seq, last_tel = -1, None
            while not ws.closed:
                with shared.lock:
                    frame, fseq, tel = shared.frame, shared.frame_seq, shared.telemetry
                if frame is not None and fseq != seq:
                    seq = fseq
                    await ws.send_bytes(frame)
                if tel is not last_tel:
                    last_tel = tel
                    await ws.send_str(json.dumps({"type": "telemetry", **tel}))
                await asyncio.sleep(0.008)

        task = asyncio.ensure_future(sender())
        try:
            async for msg in ws:
                m = json.loads(msg.data) if msg.type == WSMsgType.TEXT else {}
                if m.get("type") == "input":
                    with shared.lock:
                        shared.inputs, shared.inputs_time = m, time.perf_counter()
                elif m.get("type") == "ping":
                    await ws.send_str(json.dumps({"type": "pong", "ts": m.get("ts")}))
        finally:
            task.cancel()
            shared.clients -= 1
            print("pilot disconnected")
        return ws

    async def main():
        app = web.Application()
        app.router.add_get("/", lambda _: web.FileResponse(WEB, headers={"Cache-Control": "no-store"}))
        app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port, ssl_context=ssl_ctx).start()
        ready.set()
        await asyncio.Event().wait()

    asyncio.run(main())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--http", action="store_true", help="no HTTPS (only for a browser on this computer)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=75, help="video JPEG quality")
    ap.add_argument("--max-speed", type=float, default=1.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.8)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    ip, shared, ready = local_ip(), Shared(), threading.Event()
    ssl_ctx = None if args.http else ensure_cert(ip)
    threading.Thread(target=serve, args=(shared, args.port, ssl_ctx, ready), daemon=True).start()
    ready.wait(10)
    scheme = "http" if args.http else "https"
    print(f"\n  On the Quest (same Wi-Fi): {scheme}://{ip}:{args.port}\n  On this computer:          {scheme}://localhost:{args.port}")
    if not args.http:
        print("  Accept the self-signed certificate warning (Advanced -> proceed).")
    print("  Ctrl+C to quit.\n")
    try:
        simulate(shared, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
