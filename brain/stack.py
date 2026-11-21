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
        self.use_voxels = True                  # camera 3-D occupancy in the costmap (phase C)
        self.voxels = {}                        # floor id -> brain.mapping.voxels.VoxelMap
        self._vox_due = 0.0
        self.camera_match = False               # camera depth-to-voxel matches as EKF measurements (localisation on
                                                # a voxel map saved by the tour only; off until validated)
        self._cam_grid = {}
        self._cmatch_due = 0.0

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
        cm = self.costmaps[f]
        if self.use_voxels and cm.voxels is None:
            cm.voxels = self.voxel_map(f)
        return cm

    def voxel_map(self, floor=None):
        f = floor or self.slam.floor
        if f not in self.voxels:
            from .mapping.voxels import VoxelMap
            fl = self.slam.floors[f]
            self.voxels[f] = VoxelMap(fl.grid, fl.z)
        return self.voxels[f]

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
        if self.use_voxels and frame.stamp >= self._vox_due and self.slam.floor is not None \
                and not self.slam.in_transit() and s is not None and s.mode == 2 and not self.slam.lost:
            from .mapping.voxels import depth_to_map
            o, pts = depth_to_map(frame, self.slam.T)
            self.voxel_map().integrate(o, pts, max_range=frame.cam.max_range)
            cm = self.costmap()
            if cm is not None:
                cm.robot_top = s.leg_length + 0.35
            self._vox_due = frame.stamp + 0.2
            if self.camera_match and not self.slam.mapping and self.slam.ekf and frame.stamp >= self._cmatch_due \
                    and getattr(self.voxel_map(), "from_tour", False):
                self._camera_measurement(frame)
                self._cmatch_due = frame.stamp + 0.4
        if frame.stamp >= self._terrain_due and self.costmap() is not None and not self.slam.in_transit() \
                and s is not None and s.mode == 2:
            self.costmap().update_terrain(self.brain.auto.emap, self.slam.T, robot_top=s.leg_length + 0.35)
            self._terrain_due = frame.stamp + 0.33

    def _camera_measurement(self, frame):
        """Match the depth points against the voxel map's obstacle columns -> a pose measurement for the EKF."""
        from .mapping.matcher import match
        from .mapping.voxels import ColumnGrid, depth_local_points
        f = self.slam.floor
        vm = self.voxel_map(f)
        if f not in self._cam_grid:
            self._cam_grid[f] = ColumnGrid(vm)
        cg = self._cam_grid[f]
        if vm.version < 5:
            return
        local, od = depth_local_points(frame, self.slam.floors[f].z)
        if len(local) < 40:
            return
        cg.likelihood_field(0.08, now=frame.stamp)
        guess = self.slam.T.apply(*od)
        mp, score, C = match(cg, local, guess, win_xy=0.2, win_th=0.1, prior=0.08, cov=True)
        if score < 0.45:
            return
        # the measurement is for the pose at capture (od); express it as a correction of the current pose
        dx, dy, dth = mp[0] - guess[0], mp[1] - guess[1], mp[2] - guess[2]
        cur = self.slam.pose()
        self.slam.measure((cur[0] + dx, cur[1] + dy, cur[2] + dth), C * 3.0, kind="camera")

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
