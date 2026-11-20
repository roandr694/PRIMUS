"""Everything between the pilot and the controller, shared by the simulator viewers and the robot:

    depth points -> ElevationMap -> FeatureDetector (ahead / behind)
    lidar points -> obstacles in the corridor
    pilot command + buttons -> SkillPlanner (edge guard, skill offers, align, execute) -> RobotCommand
    (+ SkillPolicy residual when a policy is loaded. A v12 policy (obs_version 4) always runs: assist regime while
    the pilot drives, skill regime (flag on) while a skill executes, one continuous history. Older policies run
    only while a skill executes by default (policy_mode="always" runs them all the time).)

Buttons (logical): skill (A: start the offered skill), cancel (B: cancel / brake; hold 1 s toggles the edge
guard), guard (toggle the edge guard).
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .elevation_map import ElevationMap
from .features import FeatureDetector
from .lidar import corridor_obstacle, drop_ground
from .planner import SkillPlanner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))


class Autonomy:
    def __init__(self, policy_dir=None, policy_mode="skill"):
        self.det, self.policy, self.policy_mode = FeatureDetector(), None, policy_mode
        if policy_dir:
            from skill_policy import SkillPolicy
            self.policy = SkillPolicy(policy_dir)
        self.reset()

    def reset(self):
        guard = getattr(self, "plan", None) is None or self.plan.guard_on
        self.emap, self.plan = ElevationMap(), SkillPlanner()
        self.plan.guard_on = guard
        self.ahead, self.behind, self.obstacle, self.last = [], [], (math.inf, math.inf), None
        if self.policy is not None:
            self.policy.reset()

    # ---------------- sensors ----------------
    def on_depth(self, points, sigma, axle_xy, heading, leg_length):
        self.emap.move_to(axle_xy)
        self.emap.integrate(points, sigma)
        self.ahead = self.det.detect(self.emap, axle_xy, heading, leg_length, +1)
        self.behind = self.det.detect(self.emap, axle_xy, heading, leg_length, -1)

    def on_lidar(self, points, contact_z, axle_xy, heading):
        pts = drop_ground(points, lambda q: self.emap.lookup(q)[0], contact_z)
        self.obstacle = corridor_obstacle(pts, axle_xy, heading)

    def map_ground(self, xy, fallback):
        h = self.emap.lookup(xy)[0]
        return np.where(np.isfinite(h), h, fallback)

    def map_ceiling(self, xy):
        return self.emap.lookup(xy)[2]

    # ---------------- 50 Hz ----------------
    def step(self, dt, cmd, buttons, out, policy_inputs=None):
        """cmd: the pilot's RobotCommand; out: the last controller output; policy_inputs: callable returning the
        SkillPolicy.act keyword arguments (or None). -> (RobotCommand, PlannerOutput)"""
        if cmd.park and self.plan.state != "PILOT":
            self.plan.cancel()
        res = self.plan.step(dt, cmd.speed_cmd, cmd.yaw_speed_cmd, buttons, self.ahead, self.behind, out["yaw"],
                             out["x"], out.get("yaw_rate", 0.0), out["dx"], self.obstacle)
        new = replace(cmd, speed_cmd=res.speed, yaw_speed_cmd=res.yaw_rate, brake=cmd.brake or res.brake,
                      pitch_ref=res.pitch_ref, leg_length=min(cmd.leg_length, res.leg_max))
        always = self.policy is not None and (self.policy.always_on or self.policy_mode == "always")
        use = self.policy is not None and policy_inputs is not None and (always or res.state == "EXECUTE")
        if use:
            if not always and (self.last is None or self.last.state != "EXECUTE"):
                self.policy.reset()                       # a skill starts like a training episode: empty history
            kw = dict(skill=res.state == "EXECUTE") if self.policy.always_on else {}
            new = replace(new, residual=self.policy.act(cmd=new, **policy_inputs(), **kw))
        self.last = res
        return new, res

    def status(self):
        r = self.last
        if r is None:
            return dict(state="PILOT", message="", guard_on=self.plan.guard_on, guard_active=False, offer=None)
        return dict(state=r.state, message=r.message, guard_on=r.guard_on, guard_active=r.guard_active,
                    offer=r.offer.text() if r.offer is not None else None,
                    policy=self.policy is not None)
