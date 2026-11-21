"""Behaviours: cancellable, multi-second actions the agent starts (docs/home_autonomy_architecture.md §4.5).
Each one produces a PilotInput every 20 ms (navigator, skill planner and edge guard stay underneath) and ends with a
short Portuguese result the agent can say.

    b = GoTo(goal); stack.behaviour = b         # HomeStack.step drives it; b.done, b.ok, b.result
"""
import math

import numpy as np

from .runtime import PilotInput

HEIGHTS = {"baixo": 0.20, "low": 0.20, "medio": 0.30, "médio": 0.30, "normal": 0.30, "medium": 0.30,
           "alto": 0.46, "high": 0.46, "maximo": 0.50, "máximo": 0.50, "minimo": 0.18, "mínimo": 0.18}
LEG_MIN, LEG_MAX = 0.18, 0.50


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Behaviour:
    name = "behaviour"

    def __init__(self):
        self.done, self.ok, self.result, self.started = False, False, "", None

    def finish(self, ok, result):
        self.done, self.ok, self.result = True, ok, result

    def start(self, now, stack):
        self.started = now

    def step(self, now, stack):
        if self.started is None:
            self.start(now, stack)
        return self.tick(now, stack)

    def tick(self, now, stack):
        return PilotInput(leg_length=stack.leg, source="agent")

    def cancel(self, stack):
        self.finish(False, "cancelado")

    def progress(self):
        return ""


class GoTo(Behaviour):
    name = "go_to"

    def __init__(self, goal, timeout=240.0):
        super().__init__()
        self.goal, self.timeout = goal, timeout

    def start(self, now, stack):
        super().start(now, stack)
        v = 0.35 if stack.carrying else 0.6
        stack.nav.go_to(self.goal.xy, heading=self.goal.heading, label=self.goal.label, v_max=v, leg=stack.leg)

    def tick(self, now, stack):
        p = stack.nav.step(now)
        st = stack.nav.status
        if st == "arrived":
            self.finish(True, f"cheguei: {self.goal.label}")
        elif st.startswith("failed"):
            why = st.split(":", 1)[1].strip() if ":" in st else st
            self.finish(False, f"não consegui chegar em {self.goal.label} ({why}). O caminho pode estar bloqueado.")
        elif now - self.started > self.timeout:
            stack.nav.cancel()
            self.finish(False, f"demorei demais para chegar em {self.goal.label}")
        p.source = "agent"
        if stack.carrying:
            p.max_accel = 0.5
        return p

    def cancel(self, stack):
        stack.nav.cancel()
        super().cancel(stack)

    def progress(self):
        return f"indo para {self.goal.label}"


class SetBody(Behaviour):
    name = "set_body"

    def __init__(self, leg=None, delta=None):
        super().__init__()
        self.leg, self.delta = leg, delta

    def start(self, now, stack):
        super().start(now, stack)
        L = stack.leg + self.delta if self.delta is not None else self.leg
        stack.leg = float(np.clip(L, LEG_MIN, LEG_MAX if not stack.carrying else 0.40))
        self.target = stack.leg

    def tick(self, now, stack):
        L = stack.brain.state.leg_length
        if (now - self.started > 0.5 and abs(L - self.target) < 0.02) or now - self.started > 4.0:
            if abs(L - self.target) > 0.03:
                self.finish(False, f"só consegui chegar a {L * 100:.0f} cm de perna (algo acima de mim limita a altura)")
                return PilotInput(leg_length=stack.leg, source="agent")
            word = "mais baixo" if self.target < 0.27 else ("mais alto" if self.target > 0.36 else "na altura normal")
            self.finish(True, f"pronto, estou {word} (perna {self.target * 100:.0f} cm)")
        return PilotInput(leg_length=stack.leg, source="agent")


class Park(Behaviour):
    name = "park"

    def __init__(self, park=True):
        super().__init__()
        self.park = park

    def tick(self, now, stack):
        mode = stack.brain.state.mode
        if self.park and mode == 0 and now - self.started > 1.0:
            self.finish(True, "estacionei")
        if not self.park and mode == 2 and now - self.started > 0.5:
            self.finish(True, "estou de pé")
        if now - self.started > 12.0:
            self.finish(False, "não consegui " + ("estacionar" if self.park else "levantar"))
        return PilotInput(leg_length=stack.leg, park=self.park, source="agent")


class Stop(Behaviour):
    name = "stop"

    def tick(self, now, stack):
        stack.nav.cancel()
        stack.brain.cancel_skill()
        if now - self.started > 0.6:
            self.finish(True, "parei")
        return PilotInput(leg_length=stack.leg, brake=True, source="agent")


class Turn(Behaviour):
    name = "turn"

    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def start(self, now, stack):
        super().start(now, stack)
        self.target = wrap(stack.pose()[2] + self.angle)

    def tick(self, now, stack):
        e = wrap(self.target - stack.pose()[2])
        if abs(e) < 0.08 or now - self.started > 10:
            self.finish(True, "virei")
            return PilotInput(leg_length=stack.leg, source="agent")
        return PilotInput(yaw_rate=float(np.clip(2.0 * e, -0.9, 0.9)), leg_length=stack.leg, source="agent")


class Cross(Behaviour):
    """'sobe aquela escada': start the skill the detector offers for that feature kind, wait until it ends."""
    name = "cross"
    KINDS = {"stairs_up": ("stairs up",), "stairs_down": ("stairs down",), "curb": ("curb",), "gap": ("gap",),
             "overhang": ("overhang",), "any": ("",)}

    def __init__(self, kind="any", timeout=40.0):
        super().__init__()
        self.kind, self.timeout, self.phase = kind, timeout, "wait_offer"

    def tick(self, now, stack):
        if stack.carrying:
            self.finish(False, "não vou pular nem subir degraus carregando algo")
            return PilotInput(leg_length=stack.leg, source="agent")
        st = stack.brain.status()
        state = stack.brain.planner_state
        if self.phase == "wait_offer":
            offer = st.get("offer") or ""
            if offer and any(k in offer for k in self.KINDS.get(self.kind, ("",))):
                stack.brain.request_skill()
                self.phase = "running"
                self.offer = offer
            elif now - self.started > 6.0:
                self.finish(False, "não vejo esse obstáculo à minha frente; me coloque de frente para ele")
        elif self.phase == "running":
            if state in ("ALIGN", "EXECUTE"):
                self.phase = "executing"
            elif now - self.started > 8.0:
                self.finish(False, "o planejador não começou a manobra")
        elif self.phase == "executing" and state == "PILOT":
            self.finish(True, f"feito ({self.offer.split(' - ')[0]})")
        if now - self.started > self.timeout:
            stack.brain.cancel_skill()
            self.finish(False, "a manobra demorou demais, cancelei")
        return PilotInput(leg_length=stack.leg, source="agent")


class Follow(Behaviour):
    """Follow the tracked person at 1.0-1.5 m; go to the last seen position when the track is lost."""
    name = "follow"

    def __init__(self, distance=1.2, timeout=600.0):
        super().__init__()
        self.distance, self.timeout = distance, timeout
        self.t_goal, self.lost_since = -1e9, None

    def tick(self, now, stack):
        people = stack.people
        tr = people.target(now) if people is not None else None
        x, y, th = stack.pose()
        if tr is None:
            if self.lost_since is None:
                self.lost_since = now
            last = people.last_target() if people is not None else None
            if last is not None and now - self.lost_since < 12.0:
                if now - self.t_goal > 1.0:
                    stack.nav.go_to(last.xy, label="onde te vi por último", v_max=0.6, leg=stack.leg)
                    self.t_goal = now
                return stack.nav.step(now)
            if now - self.lost_since > 12.0:
                stack.nav.cancel()
                self.finish(False, "perdi você de vista")
            return PilotInput(yaw_rate=0.5, leg_length=stack.leg, source="agent")   # look around
        self.lost_since = None
        d = np.asarray(tr.xy) - np.array([x, y])
        dist = float(np.linalg.norm(d))
        face = math.atan2(d[1], d[0])
        if dist < self.distance + 0.25:                    # close enough: just face the person
            stack.nav.cancel()
            e = wrap(face - th)
            return PilotInput(yaw_rate=float(np.clip(1.8 * e, -1.0, 1.0)), leg_length=stack.leg, source="agent")
        if now - self.t_goal > 0.6 or stack.nav.status != "driving":
            goal = np.asarray(tr.xy) - d / max(dist, 1e-6) * self.distance
            stack.nav.go_to(goal, heading=face, label="pessoa", v_max=min(0.9, 0.4 + 0.5 * (dist - self.distance)),
                            leg=stack.leg, tol=0.3)
            self.t_goal = now
        p = stack.nav.step(now)
        if now - self.started > self.timeout:
            self.finish(True, "parei de seguir")
        p.source = "agent"
        return p

    def cancel(self, stack):
        stack.nav.cancel()
        super().cancel(stack)

    def progress(self):
        return "seguindo você"


class Find(Behaviour):
    """Find an object: the memory first, else visit the rooms where it is likely and look around."""
    name = "find"

    def __init__(self, cls, label, rooms=None):
        super().__init__()
        self.cls, self.label, self.rooms, self.sub = cls, label, list(rooms or []), None

    def tick(self, now, stack):
        if self.sub is None:
            hits = stack.memory.query(self.cls) if stack.memory else []
            hits = [(s, o) for s, o in hits if s > 0.27 or o.cls == self.cls]
            if hits:
                o = hits[0][1]
                xy, h = stack.scene.approach(stack, o, stack.pose()[:2])
                if xy is not None:
                    from .semantic.scene_graph import Goal
                    self.sub = GoTo(Goal(xy, h, o.floor, self.label, "object"))
                    self.found = o
            elif self.rooms:
                r = self.rooms.pop(0)
                from .semantic.scene_graph import Goal
                self.sub = GoTo(Goal(stack.scene.room_point(r), None, r.floor, r.label or r.pt, "room"))
                self.found = None
            else:
                self.finish(False, f"procurei mas não encontrei {self.label}")
                return PilotInput(leg_length=stack.leg, source="agent")
        p = self.sub.step(now, stack)
        if self.sub.done:
            if self.found is not None:
                room = f" ({self.found.room})" if self.found.room else ""
                self.finish(self.sub.ok, f"encontrei {self.label}{room}" if self.sub.ok else self.sub.result)
            self.sub = None
        return p

    def cancel(self, stack):
        if self.sub is not None:
            self.sub.cancel(stack)
        super().cancel(stack)


class Sequence(Behaviour):
    """Run behaviours one after another (multi-step requests); stops at the first failure."""
    name = "sequence"

    def __init__(self, items, say=None):
        super().__init__()
        self.items, self.k, self.results, self.say = list(items), 0, [], say

    def tick(self, now, stack):
        if self.k >= len(self.items):
            self.finish(True, "; ".join(self.results))
            return PilotInput(leg_length=stack.leg, source="agent")
        b = self.items[self.k]
        p = b.step(now, stack)
        if b.done:
            self.results.append(b.result)
            if not b.ok:
                self.finish(False, "; ".join(self.results))
            self.k += 1
        return p

    def cancel(self, stack):
        if self.k < len(self.items):
            self.items[self.k].cancel(stack)
        super().cancel(stack)

    def progress(self):
        return self.items[self.k].progress() if self.k < len(self.items) else ""
