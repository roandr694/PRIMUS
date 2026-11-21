"""Messages between the Raspberry Pi 5 (control, policy, safety) and the Jetson (perception, navigation, agent).

On the robot they travel as fixed little-endian structs over UDP on the Ethernet link; the simulator
(sim/home_sim) passes the same objects in-process, so both sides are written against this one interface.
See docs/home_autonomy_architecture.md §3.

    State           Pi -> Jetson  100 Hz   odometry, body state, battery, payload, gamepad
    Command         Jetson -> Pi   50 Hz   RobotCommand fields + skill flag + source + heartbeat
    ElevationPatch  Jetson -> Pi   15 Hz   local ground / known / ceiling grid in the odom frame

Frames: odom = the Pi's wheel-inertial odometry (continuous, drifts). Headings follow the controller: `yaw` = 0
when the robot faces +Y, counter-clockwise positive (math angle of the forward direction = yaw + pi/2).
"""
import math
import struct
from dataclasses import dataclass, field, fields

import numpy as np

MAGIC = 0x5052          # "PR"
VERSION = 1
T_STATE, T_COMMAND, T_PATCH = 1, 2, 3
HEADER = struct.Struct("<HBBId")           # magic, version, type, seq, stamp [s]

SOURCES = ("none", "pilot", "nav", "agent", "safety")


@dataclass
class State:
    stamp: float = 0.0
    x: float = 0.0              # odom pose [m]
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0            # [rad], 0 = facing +Y
    pitch: float = 0.0          # body pitch [rad], + = nose up
    roll: float = 0.0
    speed: float = 0.0          # forward [m/s]
    yaw_rate: float = 0.0       # [rad/s]
    leg_length: float = 0.0     # mean [m]
    mode: int = 0               # 0 parked, 1 standing up, 2 operating, 3 landing, 4 fallen, 5 recovering
    gear_down: bool = True
    contact: bool = True
    battery: float = 1.0        # state of charge 0..1
    payload_kg: float = 0.0     # estimated extra mass on top (carry mode)
    skill_active: bool = False  # the policy runs in the skill regime
    faults: int = 0             # bit field, see FAULTS
    pad_speed: float = 0.0      # gamepad (the pad is paired with the Pi): left stick Y, right stick X [-1, 1]
    pad_yaw: float = 0.0
    pad_buttons: int = 0        # bit field, see BUTTONS

    _fmt = struct.Struct("<9fB??ff?Iffi")       # every field after stamp


MODES = ("parked", "standing", "operating", "landing", "fallen", "recovering")
FAULTS = {"stale_command": 1, "stale_patch": 2, "fall": 4, "motor": 8, "imu": 16, "low_battery": 32}
BUTTONS = {"a": 1, "b": 2, "x": 4, "y": 8, "back": 16, "start": 32, "lb": 64, "rb": 128}


@dataclass
class Command:
    stamp: float = 0.0
    speed: float = 0.0          # [m/s]
    yaw_rate: float = 0.0       # [rad/s]
    leg_length: float = 0.30    # [m]
    pitch_ref: float = 0.0      # [rad]
    max_accel: float = 1.5      # [m/s^2]
    brake: bool = False
    park: bool = False
    skill: bool = False         # the planner is executing a skill (policy skill regime)
    source: int = 0             # index in SOURCES
    heartbeat: int = 0

    _fmt = struct.Struct("<5f???BI")


@dataclass
class ElevationPatch:
    """Square grid centred near the robot, in the odom frame. ground: height [m] (nan = unknown), ceiling: lowest
    overhead surface [m] (inf = open). Cell (i, j) covers x in origin_x + j*res .. +res, y in origin_y + i*res."""
    stamp: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    res: float = 0.04
    ground: np.ndarray = field(default_factory=lambda: np.full((60, 60), np.nan, np.float32))
    ceiling: np.ndarray = field(default_factory=lambda: np.full((60, 60), np.inf, np.float32))

    @classmethod
    def from_map(cls, emap, center_xy, stamp, size=2.4):
        """Cut a patch out of perception.ElevationMap around center_xy."""
        res = emap.res
        n = int(round(size / res))
        ox = (math.floor(center_xy[0] / res) - n // 2) * res
        oy = (math.floor(center_xy[1] / res) - n // 2) * res
        xs = ox + (np.arange(n) + 0.5) * res
        ys = oy + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        g, _, c = emap.lookup(np.stack([X, Y], -1))
        return cls(stamp, ox, oy, res, g.astype(np.float32), c.astype(np.float32))

    def _ij(self, xy):
        n = self.ground.shape[0]
        j = np.floor((xy[..., 0] - self.origin_x) / self.res).astype(np.int64)
        i = np.floor((xy[..., 1] - self.origin_y) / self.res).astype(np.int64)
        inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        return np.clip(i, 0, n - 1), np.clip(j, 0, n - 1), inside

    def ground_at(self, xy):
        i, j, inside = self._ij(np.asarray(xy, np.float64))
        return np.where(inside, self.ground[i, j], np.nan)

    def ceiling_at(self, xy):
        i, j, inside = self._ij(np.asarray(xy, np.float64))
        return np.where(inside, self.ceiling[i, j], np.inf)


# ---------------------------------------------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------------------------------------------
def _values(msg):
    return [getattr(msg, f.name) for f in fields(msg)]


def pack(msg, seq=0):
    if isinstance(msg, (State, Command)):
        typ = T_STATE if isinstance(msg, State) else T_COMMAND
        return HEADER.pack(MAGIC, VERSION, typ, seq, msg.stamp) + type(msg)._fmt.pack(*_values(msg)[1:])
    if isinstance(msg, ElevationPatch):
        n = msg.ground.shape[0]
        g = np.nan_to_num(msg.ground, nan=-1000.0).astype("<f2")       # float16: mm precision is plenty
        c = np.minimum(np.nan_to_num(msg.ceiling, posinf=1000.0), 1000.0).astype("<f2")
        head = struct.pack("<fffH", msg.origin_x, msg.origin_y, msg.res, n)
        return HEADER.pack(MAGIC, VERSION, T_PATCH, seq, msg.stamp) + head + g.tobytes() + c.tobytes()
    raise TypeError(type(msg))


def unpack(buf):
    """-> (message, seq)"""
    magic, ver, typ, seq, stamp = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a PRIMUS link message (or another version)")
    body = memoryview(buf)[HEADER.size:]
    if typ in (T_STATE, T_COMMAND):
        cls = State if typ == T_STATE else Command
        return cls(stamp, *cls._fmt.unpack(body)), seq
    if typ == T_PATCH:
        ox, oy, res, n = struct.unpack_from("<fffH", body, 0)
        a = np.frombuffer(body[14:], "<f2").astype(np.float32)
        g, c = a[:n * n].reshape(n, n), a[n * n:2 * n * n].reshape(n, n)
        g = np.where(g < -999.0, np.nan, g)
        c = np.where(c > 999.0, np.inf, c)
        return ElevationPatch(stamp, ox, oy, res, g, c), seq
    raise ValueError(f"unknown message type {typ}")

