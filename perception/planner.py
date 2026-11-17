"""Skill planner and edge guard: turns the pilot's input + the detected features into the policy commands.

    PILOT    the pilot drives (speed, turn rate, height). The edge guard (on by default) caps the speed so the
             robot stops `margin` before any guarded feature (drop, stairs, gap, wall, low overhang, obstacle
             from the lidar), in the direction of travel. The nearest crossable feature is offered to the pilot.
    ALIGN    after pressing the skill button (A) with a feature offered: stop and turn perpendicular to its edge.
    EXECUTE  drive across at the skill speed holding that heading; the skill policy does the rest. Ends after
             the feature (odometry) or on timeout; the cancel button (B) brakes and returns to PILOT anytime.
The guard toggle (Back on the gamepad, or holding B for 1 s) disables/enables the edge guard.
Looking: the camera is level and low, so the planner pitches the body to look — down (look_down) when stopped or
slow at a drop / unseen ground / stairs down / gap, and while aligning for such a skill (at least `peek_s` before
crossing, so the map fills in); up (look_up) when slow in front of a "wall" or a low overhang (a bar seen without
the floor under it). Back to level while a skill executes.
Commands use the controller conventions: speed [m/s] + forward, yaw rate [rad/s] + left (counter-clockwise).
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
import policy_interface as PI  # noqa: E402

# skill speeds: one table with the training environment (control/policy_interface.py)
SKILL_SPEED = {k: v["speed"] for k, v in PI.SKILLS.items()}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class PlannerOutput:
    speed: float
    yaw_rate: float
    state: str
    message: str
    guard_on: bool
    guard_active: bool              # the guard is cutting the pilot's speed now
    offer: object = None            # Feature offered to the pilot
    brake: bool = False             # hard brake (speed target snaps to zero)
    pitch_ref: float = 0.0          # body pitch to hold [rad] (+ = nose up): looking with the camera


class SkillPlanner:
    def __init__(self, brake_accel=0.8, latency=0.7, margin=0.35, heading_gain=1.5, max_yaw=1.5, align_tol_deg=4.0,
                 skill_timeout=15.0, offer_range=2.5, look_down=-0.32, look_up=0.18, look_rate=0.6, peek_s=1.0):
        # latency: perception + speed-tracking lag of the controller [s] (the real speed trails the command)
        self.brake_accel, self.latency, self.margin = brake_accel, latency, margin
        self.heading_gain, self.max_yaw = heading_gain, max_yaw
        self.align_tol = math.radians(align_tol_deg)
        self.skill_timeout, self.offer_range = skill_timeout, offer_range
        self.look_down, self.look_up, self.look_rate, self.peek_s = look_down, look_up, look_rate, peek_s
        self.pitch, self.speed = 0.0, 0.0
        self.state, self.guard_on = "PILOT", True
        self.target, self.goal_heading, self.start_x, self.t = None, 0.0, 0.0, 0.0
        self._prev = {}
        self._b_held = 0.0

    def _press(self, buttons, k):
        now = bool(buttons.get(k))
        hit = now and not self._prev.get(k, False)
        self._prev[k] = now
        return hit

    def cancel(self):
        self.state, self.target = "PILOT", None

    def _heading_rate(self, heading):
        return max(-self.max_yaw, min(self.max_yaw, self.heading_gain * wrap(self.goal_heading - heading)))

    def _cap(self, d):
        """largest speed that still stops `margin` before d: v*latency + v^2 / (2 a) = d - margin"""
        aT = self.brake_accel * self.latency
        return -aT + math.sqrt(aT * aT + 2.0 * self.brake_accel * max(0.0, d - self.margin))

    def step(self, dt, speed_cmd, yaw_cmd, buttons, ahead, behind, heading, x, yaw_rate, speed,
             obstacle=(math.inf, math.inf)):
        """ahead/behind: Feature lists (nearest first) from FeatureDetector; x, speed: odometry [m, m/s];
        obstacle: (distance ahead, distance behind) from the lidar. -> PlannerOutput"""
        self.speed = speed
        out = self._step(dt, speed_cmd, yaw_cmd, buttons, ahead, behind, heading, x, yaw_rate, obstacle)
        # where to look (rate limited), from the state and what is ahead
        want = 0.0
        near = next((f for f in ahead if f.guard and f.s < 1.5), None)
        slow = abs(self.speed) < 0.35
        down_kinds = ("cliff", "curb_down", "stairs_down", "gap")
        if out.state == "ALIGN" and self.target is not None and self.target.kind in down_kinds:
            want = self.look_down
        elif out.state == "PILOT" and near is not None and slow:
            want = self.look_down if near.kind in down_kinds else self.look_up if near.kind in ("wall", "overhang") else 0.0
        step = self.look_rate * dt
        self.pitch += max(-step, min(step, want - self.pitch))
        out.pitch_ref = self.pitch
        return out

    def _step(self, dt, speed_cmd, yaw_cmd, buttons, ahead, behind, heading, x, yaw_rate, obstacle):
        skill_hit, cancel_hit = self._press(buttons, "skill"), self._press(buttons, "cancel")
        self._b_held = self._b_held + dt if buttons.get("cancel") else 0.0
        if self._press(buttons, "guard") or (self._b_held >= 1.0 and self._b_held - dt < 1.0):
            self.guard_on = not self.guard_on
        if cancel_hit and self.state != "PILOT":
            self.cancel()
            return PlannerOutput(0.0, 0.0, "PILOT", "skill cancelled", self.guard_on, False, brake=True)
        offer = next((f for f in ahead if f.feasible and f.s <= self.offer_range), None)
        first_guard = next((f for f in ahead if f.guard), None)
        if first_guard is not None and offer is not None and first_guard.s < offer.s - 0.05:
            offer = None                    # something we cannot cross comes first
        if self.state == "PILOT":
            if skill_hit and offer is not None:
                self.state, self.target, self.t = "ALIGN", offer, 0.0
                self.goal_heading = offer.edge_heading if math.isfinite(offer.edge_heading) else heading
            else:
                return self._pilot(speed_cmd, yaw_cmd, ahead, behind, obstacle, offer)
        self.t += dt
        if self.state == "ALIGN":
            err = wrap(self.goal_heading - heading)
            peeking = self.target.kind in ("cliff", "curb_down", "stairs_down", "gap") and \
                (self.t < self.peek_s or abs(self.pitch - self.look_down) > 0.05)
            if abs(err) < self.align_tol and abs(yaw_rate) < 0.2 and not peeking:
                self.state, self.start_x, self.t = "EXECUTE", x, 0.0
            elif self.t > 6.0:
                self.cancel()
                return PlannerOutput(0.0, 0.0, "PILOT", "could not align", self.guard_on, False)
            else:
                return PlannerOutput(0.0, self._heading_rate(heading), "ALIGN",
                                     f"aligning for {self.target.text()}", self.guard_on, False, self.target)
        # EXECUTE
        f = self.target
        if x - self.start_x >= f.end + 0.6 or self.t > self.skill_timeout:
            done = x - self.start_x >= f.end + 0.6
            self.cancel()
            return PlannerOutput(0.0, 0.0, "PILOT", "skill done" if done else "skill timeout", self.guard_on, False)
        v = SKILL_SPEED.get(f.kind, 0.5)
        if obstacle[0] < 1.0:               # something (a person) walked in front: stop, keep the skill armed
            v = min(v, self._cap(obstacle[0]))
        return PlannerOutput(v, self._heading_rate(heading), "EXECUTE", f"{f.skill}: {f.text()}", self.guard_on,
                             False, f)

    def _pilot(self, speed_cmd, yaw_cmd, ahead, behind, obstacle, offer):
        msg = f"{offer.text()} - A to {offer.skill}" if offer is not None else ""
        if not self.guard_on:
            return PlannerOutput(speed_cmd, yaw_cmd, "PILOT", msg or "edge guard OFF", False, False, offer)
        feats, obst = (ahead, obstacle[0]) if speed_cmd >= 0 else (behind, obstacle[1])
        d = min([f.s for f in feats if f.guard] + [obst])
        cap = self._cap(d)
        active = abs(speed_cmd) > cap
        v = math.copysign(min(abs(speed_cmd), cap), speed_cmd)
        # the real speed is above what stops in time (controller lag): brake now
        late = math.copysign(1.0, speed_cmd or 1.0) * self.speed > cap + 0.1
        if active and not msg:
            near = next((f for f in feats if f.guard and f.s == d), None)
            msg = f"edge guard: {near.text()}" if near is not None else f"edge guard: obstacle at {d:.1f} m"
        # brake only while still moving: the controller's brake drops the position hold (target = position),
        # so a stopped robot held with brake would slowly drift over the edge
        moving = abs(self.speed) > 0.1
        return PlannerOutput(v, yaw_cmd, "PILOT", msg, True, active, offer,
                             brake=active and moving and (late or d <= self.margin + 0.05))
