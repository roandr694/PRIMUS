"""The Jetson side ("brain") main loop, independent of where the sensors come from (sim/home_sim or the drivers).

    sensors (odom frame) ─► elevation map + features + skill planner / edge guard (perception/)  ─► Command
                         └► (P1+) localisation, maps, navigator, semantic memory, agent  ─► pilot input of the planner

The navigator and the agent are "pilots": they produce the same speed / yaw-rate request as the gamepad, and the
existing skill planner and edge guard stay underneath them. Skills start either from the A button (pilot) or
programmatically (`request_skill`, used by the navigator when the route crosses a feature).
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIMUS = Path(__file__).resolve().parents[1]
for p in (PRIMUS, PRIMUS / "robot", PRIMUS / "control"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import link  # noqa: E402
from perception.autonomy import Autonomy  # noqa: E402
from perception.lidar import RPLidarA1  # noqa: E402
from repo_control_policy import RobotCommand  # noqa: E402


@dataclass
class PilotInput:
    """What a pilot (gamepad, navigator, agent behaviour) asks for."""
    speed: float = 0.0
    yaw_rate: float = 0.0
    leg_length: float = 0.30
    park: bool = False
    brake: bool = False
    max_accel: float = 1.5
    source: str = "nav"


class Brain:
    """Holds the robot-centric perception (elevation map, features, planner) and turns a pilot input into the
    Command for the Pi at 50 Hz, plus the ElevationPatch at 15 Hz."""

    def __init__(self, guard=True):
        self.auto = Autonomy(policy_dir=None)                     # the policy runs on the Pi
        self.auto.plan.guard_on = guard
        self.lidar = RPLidarA1()
        self.state, self.res, self.patch_due, self.hb = None, None, 0.0, 0
        self.skill_request = False
        self.dist = 0.0                                           # travelled distance (the planner's odometry x)
        self.last_scan_pts = np.zeros((0, 3))

    # ------------------------------------------------------------------------------------------------ sensors
    def axle_xy(self, pose):
        """Wheel axle midpoint from the base pose (the base origin is above it within a few cm when balancing);
        the robot-side estimator knows the legs, here the base xy projected is enough for the map window."""
        return pose.xy

    def on_depth(self, frame):
        cam = frame.cam
        pts, sig = cam.points(frame.depth, stride=2)
        world, _ = cam.to_world(pts, frame.pose.p, frame.pose.R)
        L = self.state.leg_length if self.state is not None else 0.3
        self.auto.on_depth(world, sig, self.axle_xy(frame.pose), frame.pose.yaw, L)

    def contact_z(self):
        """Ground height under the wheels from the state (base above the axle by the leg length, wheel radius
        0.1075 m; within ~1 cm of the true contact in sim)."""
        s = self.state
        return s.z - s.leg_length * math.cos(s.pitch) - 0.1075 if s is not None else 0.0

    def on_scan(self, scan, contact_z=None):
        contact_z = self.contact_z() if contact_z is None else contact_z
        pts, _ = self.lidar.points(scan.ranges, scan.pose.p, scan.pose.R)
        self.last_scan_pts = pts
        self.auto.on_lidar(pts, contact_z, self.axle_xy(scan.pose), scan.pose.yaw)

    def on_state(self, state: link.State):
        self.state = state

    def request_skill(self):
        """Start the offered skill as if A had been pressed (the navigator calls this at a skill edge)."""
        self.skill_request = True

    def cancel_skill(self):
        self.auto.plan.cancel()

    # ------------------------------------------------------------------------------------------------ 50 Hz
    def step(self, now, pilot: PilotInput):
        """-> (link.Command, link.ElevationPatch or None)"""
        s = self.state
        if s is not None:
            self.dist += s.speed * 0.02
        out = dict(yaw=s.yaw, x=self.dist, dx=s.speed, yaw_rate=s.yaw_rate) if s is not None else \
            dict(yaw=0.0, x=0.0, dx=0.0, yaw_rate=0.0)
        rc = RobotCommand(speed_cmd=pilot.speed, yaw_speed_cmd=pilot.yaw_rate, leg_length=pilot.leg_length,
                          park=pilot.park, brake=pilot.brake, max_accel=pilot.max_accel)
        buttons = {"skill": self.skill_request}
        self.skill_request = False
        if s is not None and s.mode == 2:
            rc, self.res = self.auto.step(0.02, rc, buttons, out, None)
        self.hb += 1
        src = link.SOURCES.index(pilot.source) if pilot.source in link.SOURCES else 0
        cmd = link.Command(stamp=now, speed=rc.speed_cmd, yaw_rate=rc.yaw_speed_cmd, leg_length=rc.leg_length,
                           pitch_ref=rc.pitch_ref, max_accel=rc.max_accel, brake=rc.brake, park=rc.park,
                           skill=self.res is not None and self.res.state == "EXECUTE", source=src, heartbeat=self.hb)
        patch = None
        if now >= self.patch_due and s is not None:
            patch = link.ElevationPatch.from_map(self.auto.emap, (s.x, s.y), now)
            self.patch_due = now + 1 / 15
        return cmd, patch

    @property
    def planner_state(self):
        return self.res.state if self.res is not None else "PILOT"

    def status(self):
        return self.auto.status()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
