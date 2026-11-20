"""Pilot the simulated robot with a Meta Quest 3 over Wi-Fi (Linux and macOS).

The computer runs MuJoCo + the controller in real time and serves a WebXR page over HTTPS. In the Quest
browser the page shows a large screen with the robot's first-person camera (chase camera in the corner),
a pilot panel, and sends the Quest controllers back.

    cd sim
    python3 scripts/run_quest_viewer.py [--flat]
    python3 scripts/run_quest_viewer.py --world skills --tile stairs --level 3 --start foot [--policy <exported>]
    python3 scripts/run_quest_viewer.py --world house [--spawn varanda]     # the CMA house (house/README.md)

On the Quest (same Wi-Fi): browser -> the https://<ip>:8443 address printed here -> accept the self-signed
certificate warning (Advanced -> proceed) -> "Enter VR". The same page works in a desktop browser with an
Xbox controller (Gamepad API) or the keyboard, for piloting without the headset.

Autonomy (perception/): the simulated D435i and RPLIDAR A1 feed the elevation map; the edge guard (on by
default) stops the robot before drops, stairs, gaps, low overhangs and obstacles; when a skill is offered on the
panel, A starts it (the robot aligns and crosses; with --policy the skill policy drives, otherwise only the
model-based controller); B brakes and cancels, holding B for 1 s turns the edge guard off / on.

Quest controls: left stick = speed, right stick = turn, triggers = leg height (latches), A = start the offered
skill, B = brake / cancel (hold: edge guard), X = land <-> stand up, Y = restart, both grips = re-centre the
screen, right / left stick click = responsiveness up / down. Xbox (desktop browser): sticks, triggers or D-pad,
A skill, B brake / cancel, X land, Y soft stop, Back edge guard, Start restart, LB / RB responsiveness.
Keyboard: W/S, A/D, E/Q, Enter skill, space brake / cancel, G edge guard, P land, R restart, 1/2/3.
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
from dataclasses import replace
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering without a window (macOS default works)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import skill_world as SW  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "house"))
from house_runtime import HouseRuntime  # noqa: E402
from perception.autonomy import Autonomy  # noqa: E402
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


FPV = ("fpv", (0.0, 0.19, 0.11), 95.0, 12.0)      # first-person camera: front of the chassis, 12 deg down


def simulate(shared, args):
    """Real-time physics + controller + autonomy; renders and JPEG-encodes the video while a pilot is connected."""
    world = SW.SkillWorld(args.world, torque=args.torque, extra_cameras=[FPV])
    model, data, sens = world.model, world.data, world.sens
    world.spawn = args.spawn
    world.place(args.tile, args.level, args.start)
    house_rt = HouseRuntime(mujoco, model) if args.world == "house" else None
    policy = RepoControlPolicy(mujoco, model)
    auto = Autonomy(args.policy, args.policy_mode)
    rec = SW.FallRecovery(world, policy, args.recovery) if args.recovery else None
    rec_msg = None
    scan = args.scan if world.truth is not None else "map"
    pilot = Pilot(args.max_speed, args.max_yaw_rate)
    W, H = args.width, args.height
    pw, ph = W * 3 // 10, H * 3 // 10
    fpv, chase = mujoco.Renderer(model, H, W), mujoco.Renderer(model, ph, pw)
    cam = mujoco.MjvCamera()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    dt = model.opt.timestep

    def restart():
        world.place(args.tile, args.level, args.start)
        policy.reset(data)
        auto.reset()
        if house_rt is not None:
            house_rt.reset(data)
        return data.time, time.perf_counter()

    sim_t0, wall_t0 = restart()
    next_frame = next_tel = 0.0
    fallen_since, falls, out, rtf, rtf_win = None, 0, {}, 1.0, (time.perf_counter(), data.time)
    next_cam = next_lidar = next_plan = 0.0
    cmd = None
    while True:
        wall = time.perf_counter() - wall_t0
        sim_t0 = min(sim_t0, data.time - wall + 0.25)      # computer too slow: lag at most 0.25 s, don't catch up
        steps = 0
        while data.time - sim_t0 < wall and steps < 50:     # catch up with real time, then render
            with shared.lock:
                inp, age = shared.inputs, time.perf_counter() - shared.inputs_time
            was_park = pilot.park
            pilot_cmd, reset = pilot.command(inp, dt, stale=age > 0.5)     # no headset data for 0.5 s: stop
            if reset:
                (sim_t0, wall_t0), fallen_since = restart(), None
                next_cam = next_lidar = next_plan = 0.0
                rec_msg = None
                break
            if rec is not None and fallen_since is not None and not rec.active and pilot.park != was_park:
                pilot.park = was_park                      # X while down: stand up instead of land / operate
                rec.start()
                rec_msg = "standing up..."
            if rec is not None and rec.active:             # stand-up policy drives the joints, controller off
                status = rec.step()
                mujoco.mj_step(model, data)
                steps += 1
                if status == "done":
                    pilot.park, fallen_since, rec_msg = False, None, None
                    auto.reset()
                elif status == "timeout":
                    rec_msg = "did not stand up - X to try again, Y to restart"
                continue
            if out and not args.no_autonomy:
                if data.time >= next_cam:                                   # D435i 15 Hz
                    next_cam = data.time + 1 / 15
                    L = 0.5 * (out["left_leg_length"] + out["right_leg_length"])
                    auto.on_depth(*sens.depth_points(data), world.axle(), out["yaw"], L)
                if data.time >= next_lidar:                                 # RPLIDAR A1 5.5 Hz
                    next_lidar = data.time + 1 / 5.5
                    auto.on_lidar(sens.lidar_points(data), world.contact_z(policy.wheel_radius), world.axle(), out["yaw"])
                if data.time >= next_plan or cmd is None:                   # planner + skill policy 50 Hz
                    next_plan = data.time + 0.02
                    buttons = {"skill": inp.get("skill"), "cancel": inp.get("brake"), "guard": inp.get("guard")}
                    cmd, _ = auto.step(0.02, pilot_cmd, buttons, out,
                                       lambda: world.policy_inputs(policy, out, auto, scan))
                else:
                    cmd = replace(cmd, park=pilot_cmd.park, leg_length=pilot_cmd.leg_length,
                                  max_accel=pilot_cmd.max_accel)
            else:
                cmd = pilot_cmd
            out = policy.apply(data, cmd)
            world.drive_limits()
            if house_rt is not None:                       # inclined lift automation (house/house_runtime.py)
                house_rt.step(data)
            mujoco.mj_step(model, data)
            steps += 1

        roll, pitch, _ = body_rpy(data)
        if abs(pitch) > math.radians(50) or abs(roll) > math.radians(50) or data.qpos[2] < 0.15:
            if fallen_since is None:
                fallen_since, falls = time.perf_counter(), falls + 1
            elif rec is None and time.perf_counter() - fallen_since > 4.0:      # fell: restart by itself
                (sim_t0, wall_t0), fallen_since = restart(), None
                continue
            elif rec is not None and not rec.active and rec_msg is None:
                rec_msg = "press X to stand up"
        elif rec is None or not rec.active:
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
                    wheel_torque=r3(max(abs(c["wheel_xp"]), abs(c["wheel_xm"]))), torque_limit=world.torque,
                    contact_l=r3(out["tire_force"]["xm"]), contact_r=r3(out["tire_force"]["xp"]),
                    park=pilot.park, brake=cmd.brake, fallen=fallen_since is not None, falls=falls,
                    recovery=rec_msg if rec is not None else None,
                    level=pilot.level_name, max_accel=cmd.max_accel,
                    lat_g=r3(out["a_lat"] / 9.81), turn_limited=abs(cmd.yaw_speed_cmd) > out["yaw_limit"] + 1e-6,
                    auto=None if args.no_autonomy else auto.status(),
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
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the arena (= --world flat)")
    ap.add_argument("--world", choices=("arena", "flat", "skills", "house"), default="arena",
                    help="skills: the skill terrain of sim/warp_rl/skill_terrain.py (stairs, curbs, gaps, bars); "
                         "house: the CMA house (build it first with house/build.py)")
    ap.add_argument("--spawn", default="parked",
                    help="--world house: start keyframe (parked = living room, varanda, kitchen, lawn, pool_deck, "
                         "sea_deck, pavilion, upper_lounge, suite1, garden, terrace, lift_entry, parking)")
    ap.add_argument("--tile", default="stairs", choices=SW.ST.TYPES, help="--world skills: start tile type")
    ap.add_argument("--level", type=int, default=2, help="--world skills: start tile difficulty 0-5")
    ap.add_argument("--start", default="foot", choices=("foot", "top", "center"))
    ap.add_argument("--policy", default=None, help="exported skill policy (sim/warp_rl/export_policy.py)")
    ap.add_argument("--scan", choices=("truth", "map"), default="map",
                    help="what the policy sees: the elevation map (as on the robot) or the true terrain (skills world)")
    ap.add_argument("--torque", type=float, default=None,
                    help="joint torque limit [N.m] with the motor speed droop (default: the model's, 18)")
    ap.add_argument("--recovery", default=None,
                    help="exported stand-up policy: after a fall, X stands the robot up (instead of a restart)")
    ap.add_argument("--policy-mode", choices=("skill", "always"), default="skill",
                    help="run the policy only while a skill executes (default) or all the time")
    ap.add_argument("--no-autonomy", action="store_true", help="pilot only: no perception, guard or skills")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--http", action="store_true", help="no HTTPS (only for a browser on this computer)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=75, help="video JPEG quality")
    ap.add_argument("--max-speed", type=float, default=1.5)
    ap.add_argument("--max-yaw-rate", type=float, default=1.8)
    args = ap.parse_args()
    if args.flat:
        args.world = "flat"
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
