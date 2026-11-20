"""The whole Jetson stack in one object: Brain (elevation map, features, skill planner, edge guard) + floor SLAM +
per-floor costmaps + navigator (+ semantic memory and the agent, attached later by their own modules).

    st = HomeStack.load("sim/models/house/robot_map")      # localise on the tour map (or HomeStack() to map)
    each 20 ms:   st.on_state(state); st.on_depth(frame); st.on_scan(scan)
                  cmd, patch = st.step(now)                # -> link.Command, link.ElevationPatch
    st.nav.go_to((x, y), heading)                          # or behaviours / the agent
"""
import math

import numpy as np

from .mapping.slam import FloorSLAM
from .nav.costmap import Costmap
from .nav.navigator import Navigator
from .runtime import Brain, PilotInput


class HomeStack:
    def __init__(self, slam=None, guard=True):
        self.slam = slam or FloorSLAM(mapping=True)
        self.brain = Brain(guard=guard, slam=self.slam)
        self.costmaps = {}
        self.nav = Navigator(self.brain, None)
        self.pilot_override = None              # a behaviour can drive directly (PilotInput) instead of the navigator
        self.behaviour = None
        self._terrain_due = 0.0
        self.memory = None                      # brain.semantic.memory.SemanticMemory (P3)
        self.scene = None                       # brain.semantic.scene_graph.SceneGraph
        self.people = None                      # brain.semantic.people.PeopleTracker
        self.leg = 0.30                         # body height the behaviours keep (set_body)
        self.carrying = False                   # carry mode: gentle driving, no jump skills
        self.message = ""

    @classmethod
    def load(cls, directory, guard=True):
        return cls(FloorSLAM.load(directory, mapping=False), guard=guard)

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def floor(self):
        return self.slam.floor

    def costmap(self, floor=None):
        f = floor or self.slam.floor
        if f is None:
            return None
        if f not in self.costmaps:
            self.costmaps[f] = Costmap(self.slam.floors[f].grid)
        return self.costmaps[f]

    def pose(self):
        return self.brain.map_pose()

    # ------------------------------------------------------------------------------------------------ sensors
    def on_state(self, state):
        self.brain.on_state(state)
        cm = self.costmap()
        if cm is not None and self.nav.cm is not cm:
            self.nav.cm = cm                     # floor changed (or first state)

    def on_depth(self, frame):
        self.brain.on_depth(frame)
        s = self.brain.state
        if frame.stamp >= self._terrain_due and self.costmap() is not None and not self.slam.in_transit() \
                and s is not None and s.mode == 2:
            self.costmap().update_terrain(self.brain.auto.emap, self.slam.T, robot_top=s.leg_length + 0.35)
            self._terrain_due = frame.stamp + 0.33

    def on_rgbd(self, frame):
        if self.memory is not None:
            self.memory.on_rgbd(frame, self)
            if self.people is not None:
                self.people.update(frame.stamp, self.memory.last_people)

    def on_scan(self, scan):
        self.brain.on_scan(scan)
        cm = self.costmap()
        s = self.brain.state
        if cm is None or self.slam.in_transit() or s is None or s.mode != 2 or abs(s.pitch) > 0.35:
            return                                # standing up / tilted: the floor would show up as obstacles
        pts = self.brain.last_scan_pts
        cz = self.brain.contact_z()
        pts = pts[(pts[:, 2] > cz + 0.10) & (pts[:, 2] < cz + 1.9)]
        T = self.slam.T
        c, s = math.cos(T.th), math.sin(T.th)
        mxy = np.stack([T.x + c * pts[:, 0] - s * pts[:, 1], T.y + s * pts[:, 0] + c * pts[:, 1]], 1)
        x, y, _ = self.pose()
        cm.update_live(mxy, (x, y), scan.stamp)

    # ------------------------------------------------------------------------------------------------ 50 Hz
    def step(self, now):
        if self.behaviour is not None:
            pilot = self.behaviour.step(now, self)
            if self.behaviour.done:
                self.message = self.behaviour.result
                self.behaviour = None
        elif self.pilot_override is not None:
            pilot = self.pilot_override
        else:
            pilot = self.nav.step(now) if self.nav.cm is not None else PilotInput(leg_length=self.leg)
        self.brain.skills_allowed = not self.carrying
        return self.brain.step(now, pilot)

    def start(self, behaviour):
        """Start a behaviour (cancels the running one)."""
        if self.behaviour is not None and not self.behaviour.done:
            self.behaviour.cancel(self)
        self.behaviour = behaviour
        return behaviour

    def cancel(self):
        if self.behaviour is not None:
            self.behaviour.cancel(self)
            self.behaviour = None
        self.nav.cancel()

    def status(self):
        s = self.brain.status()
        s.update(nav=self.nav.status, nav_message=self.nav.message, floor=self.slam.floor, pose=self.pose(),
                 lost=self.slam.lost, behaviour=getattr(self.behaviour, "name", None))
        return s
