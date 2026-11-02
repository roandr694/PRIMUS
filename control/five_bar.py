"""Analytic kinematics of the closed 5-bar leg (base YZ plane: Y forward, Z up), no simulator needed.

Geometry from sim/tools/robot_def.py (CAD pose = all angles zero). Each leg: hip Y+ and hip Y- on the base
(140 mm apart), 220 mm arms, 350 mm lower bars meeting at the wheel axle. hip_Yp turns the front arm,
hip_Ym the rear one, both about +X.

Used by the LQR design (pose for each L0), the MJCF generator (parked keyframe), the Isaac asset and the
GPU controller in sim/warp_rl (same equations in Warp).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "tools"))
from robot_def import LEGS  # noqa: E402

# Geometric range measured with the collision meshes (symmetric legs, gear retracted): below ~0.17 m the
# wheel hits the chassis; 0.565 m is the fully extended (singular) leg.
GEOMETRIC_LENGTH_RANGE = (0.17, 0.565)


def leg_points(side):
    """CAD points of leg `side` in the (y, z) plane [m]."""
    return {k: (v[1] / 1000.0, v[2] / 1000.0) for k, v in LEGS[side].items()}


def _rot(v, a):
    y, z = v
    return (math.cos(a) * y - math.sin(a) * z, math.sin(a) * y + math.cos(a) * z)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _ang(v):
    return math.atan2(v[1], v[0])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def closure(side, hip_p, hip_m):
    """Closes the 5-bar for the hip angles [rad].

    Returns (knee_Yp, knee_Ym, foot): passive knee angles [rad] and the wheel axle (y, z) in the CAD frame,
    or None if the loop cannot close. Picks the branch with the wheel below."""
    P = leg_points(side)
    hp, hm = P["hipYp"], P["hipYm"]
    kp = tuple(x + y for x, y in zip(hp, _rot(_sub(P["kneeYp"], hp), hip_p)))
    km = tuple(x + y for x, y in zip(hm, _rot(_sub(P["kneeYm"], hm), hip_m)))
    vp, vm = _sub(P["foot"], P["kneeYp"]), _sub(P["foot"], P["kneeYm"])
    rp, rm = math.hypot(*vp), math.hypot(*vm)
    dy, dz = _sub(km, kp)
    d = math.hypot(dy, dz)
    if d < 1e-9 or d > rp + rm or d < abs(rp - rm):
        return None
    along = (rp * rp - rm * rm + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, rp * rp - along * along))
    cy, cz = kp[0] + along * dy / d, kp[1] + along * dz / d
    foot = min([(cy - h * dz / d, cz + h * dy / d), (cy + h * dz / d, cz - h * dy / d)], key=lambda p: p[1])
    knee_p = _wrap(_ang(_sub(foot, kp)) - _ang(vp) - hip_p)
    knee_m = _wrap(_ang(_sub(foot, km)) - _ang(vm) - hip_m)
    return knee_p, knee_m, foot


def hip_mid(side):
    P = leg_points(side)
    return ((P["hipYp"][0] + P["hipYm"][0]) / 2.0, (P["hipYp"][1] + P["hipYm"][1]) / 2.0)


def leg_length(side, hip_p, hip_m):
    c = closure(side, hip_p, hip_m)
    if c is None:
        return None
    v = _sub(c[2], hip_mid(side))
    return math.hypot(*v)


def symmetric_angle_for_length(side, length):
    """Angle `a` (hip_Yp = +a, hip_Ym = -a) giving leg length `length`. L0 decreases monotonically with `a`
    in [-55, 90] deg; a = 0 is the CAD pose (0.394 m)."""
    lo, hi = math.radians(-55.0), math.radians(90.0)
    if not leg_length(side, hi, -hi) <= length <= leg_length(side, lo, -lo):
        raise ValueError(f"L0 = {length:.3f} m outside the mechanism range")
    for _ in range(60):
        a = 0.5 * (lo + hi)
        if leg_length(side, a, -a) > length:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)


def nominal_pose(length, wheel_radius=0.1075, clearance=0.002):
    """Both legs at `length` [m]: ({joint name: angle [rad]} for hips, knees and wheels, base height [m]
    above flat ground with the tires `clearance` above it)."""
    lo, hi = GEOMETRIC_LENGTH_RANGE
    if not lo <= length <= hi:
        raise ValueError(f"leg length {length:.3f} m outside the mechanism range [{lo}, {hi}] m")
    q, foot_z = {}, []
    for side in LEGS:
        a = symmetric_angle_for_length(side, length)
        knee_p, knee_m, foot = closure(side, a, -a)
        q.update({f"hip_Yp_{side}": a, f"hip_Ym_{side}": -a, f"knee_Yp_{side}": knee_p,
                  f"knee_Ym_{side}": knee_m, f"wheel_{side}": 0.0})
        foot_z.append(foot[1])
    return q, wheel_radius + clearance - min(foot_z)
