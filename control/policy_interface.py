"""v12 policy interface: one always-on residual policy with an assist and a skill regime.

Shared by the training environment (sim/warp_rl/skill_env.py), the mirror augmentation (sim/warp_rl/mirror.py),
the exporter and the robot-side wrapper (control/skill_policy.py), so that the layout lives in one place.

ACTION (12, each in [-1, 1]) as common-mode / differential pairs. "diff" = xm side minus xp side.
   0 height     common leg-length offset                       x 0.17 m
   1 roll       leg-length difference (-> body roll reference)  x 0.08 m (12 deg)
   2 lean       common leg-angle offset (shifts the LQR equilibrium)  x 0.35 rad
   3 split      leg-angle difference (one wheel ahead)          x 0.20 rad
   4 wheel      common wheel torque                             x 4 N.m
   5 wheel_diff wheel torque difference                         x 2 N.m
   6 force      common force along the legs                     x 150 N
   7 force_diff leg force difference                            x 60 N
   8 trigger    > 0.5 starts the jump primitive (skill regime only)
   9 ext, 10 push, 11 tuck   jump shape: 0.46 +- 0.06 m, 190 +- 60 N, 0.25 +- 0.07 m
In the assist regime the continuous actions (0-7) are scaled by ASSIST_AUTHORITY and the trigger is ignored.
The per-side residual of the controller (controller.py RESIDUAL) is side = common -+ diff / 2.

FRAME (one history step of the actor observation, FRAME_NAMES): proprioception, all pilot/planner commands, the
ramped speed target, the skill flag, the last action, the jump phase. Mirror symmetry (left <-> right, the body x
axis): FRAME_MIRROR gives, per entry, the index it comes from and its sign.
"""
SKILL_SCALE = (0.17, 0.08, 0.35, 0.20, 4.0, 2.0, 150.0, 60.0, 1.0, 0.06, 60.0, 0.07)
OFFSET = (0.0,) * 9 + (0.46, 190.0, 0.25)
ASSIST_AUTHORITY = 0.5
ACTION_NAMES = ("height", "roll", "lean", "split", "wheel", "wheel_diff", "force", "force_diff",
                "trigger", "ext", "push", "tuck")
DIFF_ACTIONS = (1, 3, 5, 7)
ACTION_SIGN = tuple(-1.0 if i in DIFF_ACTIONS else 1.0 for i in range(12))   # mirror: differential ones flip

FRAME_NAMES = ("grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y", "gyro_z",
               "L_xp", "L_xm", "th_xp", "th_xm", "dL_xp", "dL_xm", "dth_xp", "dth_xm",
               "wheel_xp", "wheel_xm", "dx", "operating",
               "cmd_speed", "cmd_yaw_rate", "cmd_leg", "cmd_pitch", "cmd_accel", "speed_target", "skill",
               *[f"a_{n}" for n in ACTION_NAMES], "jump_phase")
FRAME_DIM = len(FRAME_NAMES)          # 38
PRIV_NAMES = ("v_x", "v_y", "v_z", "base_height", "torque_limit", "friction", "mass_scale", "grounded")
SCAN_FWD, SCAN_LAT = 31, 5            # height scan: forward -0.30..1.20 every 5 cm (outer), lateral -0.30..0.30
CEIL_FWD, CEIL_LAT = 15, 3            # ceiling scan: forward -0.2..1.2 every 10 cm (outer), lateral -0.2, 0, 0.2
ACCEL_NORM = 5.0                      # cmd_accel = max accel [m/s^2] / 5


def _frame_mirror():
    idx, sign = list(range(FRAME_DIM)), [1.0] * FRAME_DIM
    n = FRAME_NAMES.index
    for a, b in (("L_xp", "L_xm"), ("th_xp", "th_xm"), ("dL_xp", "dL_xm"), ("dth_xp", "dth_xm"),
                 ("wheel_xp", "wheel_xm")):
        idx[n(a)], idx[n(b)] = n(b), n(a)
    # body x is lateral: gravity x flips; angular velocity is a pseudo-vector, so pitch rate (x) stays and roll (y)
    # and yaw (z) rates flip; the yaw-rate command flips; differential actions flip
    for k in ("grav_x", "gyro_y", "gyro_z", "cmd_yaw_rate"):
        sign[n(k)] = -1.0
    a0 = n("a_height")
    for i in DIFF_ACTIONS:
        sign[a0 + i] = -1.0
    return tuple(idx), tuple(sign)


FRAME_MIRROR = _frame_mirror()
PRIV_MIRROR = (tuple(range(8)), (-1.0,) + (1.0,) * 7)       # body velocity x flips


def scan_mirror(n_fwd, n_lat):
    """Index permutation of a forward-major, lateral-minor scan whose lateral points are symmetric about 0."""
    return tuple(f * n_lat + (n_lat - 1 - l) for f in range(n_fwd) for l in range(n_lat))


def to_sides(a, skill, xp=None):
    """Actions [..., 12] (already clipped to [-1, 1]) -> per-side residual [..., 12] in controller units.
    skill: [...] 0/1 (assist: reduced authority, trigger off). Works on torch tensors and numpy arrays (pass the
    module as `xp`)."""
    if xp is None:
        import numpy as xp
    s = [a[..., i] * SKILL_SCALE[i] for i in range(12)]
    k = ASSIST_AUTHORITY + (1.0 - ASSIST_AUTHORITY) * skill
    side = []
    for c, d in ((0, 1), (2, 3), (4, 5), (6, 7)):
        side += [k * (s[c] - 0.5 * s[d]), k * (s[c] + 0.5 * s[d])]
    trig = xp.where(skill > 0.5, s[8], s[8] * 0.0 - 1.0)
    side += [trig, s[9] + OFFSET[9], s[10] + OFFSET[10], s[11] + OFFSET[11]]
    return xp.stack(side, -1)


# --------------------------------------------------------------------------------------------------------------
# Skills: one table for the planner (perception/planner.py) and the training environment. Speeds are the planner's
# command during EXECUTE [m/s]; training draws commands in `train_speed`. `max_level` is the largest training level
# the policy is validated for (the planner should not offer a bigger feature); `approach_deg` is the heading error
# the planner aligns to and training spawns within.
SKILLS = {
    "stairs_up":   dict(speed=0.6, train_speed=(0.3, 1.2), max_level=5),
    "stairs_down": dict(speed=0.4, train_speed=(0.3, 1.2), max_level=5),
    "curb_up":     dict(speed=0.8, train_speed=(0.3, 1.2), max_level=5),
    "curb_down":   dict(speed=0.5, train_speed=(0.3, 1.2), max_level=5),
    "gap":         dict(speed=1.0, train_speed=(0.3, 1.2), max_level=5),
    "gap_wide":    dict(speed=1.2, train_speed=(0.8, 1.6), max_level=5),
    "hurdle":      dict(speed=1.2, train_speed=(0.8, 1.6), max_level=5),
    "overhang":    dict(speed=0.4, train_speed=(0.3, 1.2), max_level=5),
}
APPROACH_DEG = 10.0         # training spawns within +-10 deg of the feature axis; the planner aligns to 4 deg

# Remote envelope (sim/scripts/pilot.py) and what training covers (a margin above it)
REMOTE = dict(max_speed=1.5, max_yaw_rate=1.8, accel_levels=(0.5, 1.5, 5.0), look_down=-0.32)
TRAIN_ENVELOPE = dict(speed=(-1.8, 1.8), yaw_rate=(-1.8, 1.8), leg=(0.20, 0.48), pitch=(-0.35, 0.0),
                      accel_levels=(0.5, 1.5, 5.0))


def ramp_accel(cmd_accel):
    """Effective acceleration [m/s^2] of the controller's speed ramp for a responsiveness setting (measured:
    0 -> 1.5 m/s in ~2.7 / 1.2 / 0.7 s at 0.5 / 1.5 / 5.0)."""
    return 1.25 * (cmd_accel / 1.5) ** 0.6

