"""Architecture of the CMA house (Jacobsen Arquitetura, Angra dos Reis, 2018), rebuilt from the published floor
plans, section and photographs (see README.md for the source reading and what was extrapolated).

Levels (z = 0 is the social floor; the section gives 3.4 m social -> sea deck, 3.6 m social -> bedrooms):
    sea          -6.25   water surface
    pier         -5.40   boat landing
    sea deck     -3.40   deck, spa and games pavilion under the planted roof
    social        0.00   pool, lawn, varanda, double-height living room, kitchens, service, wine cellar
    upper         3.60   three suites, lounge over the kitchen, entrance garden with reflecting pool, study
    roof          6.50   timber soffit under a thin hipped metal roof
    parking      13.20   reached by the glass inclined lift ("plano inclinado"), street at 16.1

Plan frame: X east (towards the hill), Y north. X = 0 is the varanda column line, Y = 0 the south facade.
"""
import math

import numpy as np

import furniture as FU
from geom import Frame, rot_x, rot_y, rot_z

Z_SEA, Z_PIER, Z_DECK, Z_SOC, Z_UP = -6.25, -5.40, -3.40, 0.0, 3.60
Z_CEIL_SOC, Z_CEIL_UP, Z_ROOF_TOP = 3.0, 6.5, 7.45
Z_PARK, Z_STREET = 13.2, 16.1
LIFT_THETA = math.atan2(Z_PARK - Z_UP, 11.9)
LIFT_TRAVEL = math.hypot(Z_PARK - Z_UP, 11.9)
LIFT_X0, LIFT_Y = 27.3, 17.0          # cabin floor centre at the lower station


def y_north(x):
    """Slanted north property line (plan: 18.7 m at the pool, 20.6 m at the garden)."""
    return 19.0 + 0.06 * x


# ================================================================================================================
# generic helpers
# ================================================================================================================
def rect_minus(r, holes):
    """Rectangle (x0, x1, y0, y1) minus axis-aligned holes -> list of rectangles."""
    x0, x1, y0, y1 = r
    hs = [(max(a, x0), min(b, x1), max(c, y0), min(d, y1)) for a, b, c, d in holes
          if a < x1 and b > x0 and c < y1 and d > y0]
    xs = sorted({x0, x1, *[h[0] for h in hs], *[h[1] for h in hs]})
    ys = sorted({y0, y1, *[h[2] for h in hs], *[h[3] for h in hs]})
    cols = []
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if b - a < 1e-6:
            continue
        runs, cur = [], None
        for j in range(len(ys) - 1):
            c, d = ys[j], ys[j + 1]
            mx, my = (a + b) / 2, (c + d) / 2
            inside = any(h[0] <= mx <= h[1] and h[2] <= my <= h[3] for h in hs)
            if inside or d - c < 1e-6:
                if cur:
                    runs.append(cur)
                    cur = None
                continue
            cur = (cur[0], d) if cur else (c, d)
        if cur:
            runs.append(cur)
        cols.append((a, b, runs))
    out = []
    i = 0
    while i < len(cols):                      # merge neighbouring columns with identical runs
        a, b, runs = cols[i]
        j = i + 1
        while j < len(cols) and cols[j][2] == runs:
            b = cols[j][1]
            j += 1
        out += [(a, b, c, d) for c, d in runs]
        i = j
    return out


def finish(s, mat, x0, x1, y0, y1, z, t=0.02, uv_rot=0.0):
    """Visual floor/ceiling finish whose top face is at z (collision comes from the slab)."""
    if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
        return
    s.box(mat, ((x0 + x1) / 2, (y0 + y1) / 2, z - t / 2), ((x1 - x0) / 2, (y1 - y0) / 2, t / 2), collide=False,
          uv_rot=uv_rot, skip=("-z",))


def ceiling(s, mat, x0, x1, y0, y1, z, t=0.02, uv_rot=0.0):
    """Visual ceiling finish whose bottom face is at z."""
    s.box(mat, ((x0 + x1) / 2, (y0 + y1) / 2, z + t / 2), ((x1 - x0) / 2, (y1 - y0) / 2, t / 2), collide=False,
          uv_rot=uv_rot, skip=("+z",))


def solid(s, x0, x1, y0, y1, z0, z1, friction=(1.0, 0.02, 0.002), name=None):
    """Collision-only box (fill, slabs whose faces are drawn by finishes)."""
    s.collider("box", ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
               friction=friction, name=name)


def slab(s, x0, x1, y0, y1, ztop, thick, top=None, under=None, side="concrete_light", friction=None, holes=(),
         top_uv_rot=0.0, under_uv_rot=0.0):
    fr = friction or (s.materials[top].friction if top else (1.0, 0.02, 0.002))
    for (a, b, c, d) in rect_minus((x0, x1, y0, y1), holes):
        solid(s, a, b, c, d, ztop - thick, ztop, friction=fr)
        if top:
            finish(s, top, a, b, c, d, ztop, uv_rot=top_uv_rot)
        if under:
            ceiling(s, under, a, b, c, d, ztop - thick, uv_rot=under_uv_rot)
        if side:
            s.box(side, ((a + b) / 2, (c + d) / 2, ztop - thick / 2 - 0.01),
                  ((b - a) / 2, (d - c) / 2, thick / 2 - 0.01 - 0.0005), collide=False, skip=("+z", "-z"))


def wall(s, axis, c, a0, a1, z0, z1, t, mat_neg, mat_pos=None, openings=(), collide=True, cap=None):
    """Axis-aligned wall. axis='x': runs along X at y = c (faces -Y: mat_neg, +Y: mat_pos); axis='y': runs along
    Y at x = c (faces -X: mat_neg, +X: mat_pos). openings: (a0, a1, z_bottom, z_top) in wall coordinates."""
    mat_pos = mat_pos or mat_neg
    for (a, b, zb, zt) in rect_minus((a0, a1, z0, z1), openings):
        for k, (m, off) in enumerate(((mat_neg, -t / 4), (mat_pos, t / 4))):
            if m is None:
                continue
            if axis == "x":
                ctr, half = ((a + b) / 2, c + off, (zb + zt) / 2), ((b - a) / 2, t / 4, (zt - zb) / 2)
            else:
                ctr, half = (c + off, (a + b) / 2, (zb + zt) / 2), (t / 4, (b - a) / 2, (zt - zb) / 2)
            s.box(m, ctr, half, collide=False)
        if collide:
            if axis == "x":
                solid(s, a, b, c - t / 2, c + t / 2, zb, zt, friction=(0.8, 0.02, 0.002))
            else:
                solid(s, c - t / 2, c + t / 2, a, b, zb, zt, friction=(0.8, 0.02, 0.002))


def wall_seg(s, p0, p1, z0, z1, t, mat, collide=True):
    """Wall between two arbitrary plan points (rotated box)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    R = rot_z(math.atan2(d[1], d[0]))
    c = ((p0 + p1) / 2).tolist() + [(z0 + z1) / 2]
    s.box(mat, c, (L / 2, t / 2, (z1 - z0) / 2), R=R, collide=collide)


def glass_wall(s, axis, c, a0, a1, z0, z1, frame="wood_dark", spacing=1.25, t=0.025, mullion=(0.05, 0.08),
               collide=True, sill=True):
    """Floor-to-ceiling glazing with mullions and head/sill profiles."""
    L = a1 - a0
    n = max(1, int(round(L / spacing)))
    for i in range(n + 1):
        a = a0 + L * i / n
        if axis == "x":
            s.box(frame, (a, c, (z0 + z1) / 2), (mullion[0] / 2, mullion[1] / 2, (z1 - z0) / 2), collide=False)
        else:
            s.box(frame, (c, a, (z0 + z1) / 2), (mullion[1] / 2, mullion[0] / 2, (z1 - z0) / 2), collide=False)
    for z, h in ((z1 - 0.03, 0.03),) + (((z0 + 0.02, 0.02),) if sill else ()):
        if axis == "x":
            s.box(frame, ((a0 + a1) / 2, c, z), (L / 2, mullion[1] / 2, h), collide=False)
        else:
            s.box(frame, (c, (a0 + a1) / 2, z), (mullion[1] / 2, L / 2, h), collide=False)
    if axis == "x":
        s.box("glass", ((a0 + a1) / 2, c, (z0 + z1) / 2), (L / 2, t / 2, (z1 - z0) / 2), collide=False)
        if collide:
            solid(s, a0, a1, c - 0.04, c + 0.04, z0, z1, friction=(0.45, 0.01, 0.001))
    else:
        s.box("glass", (c, (a0 + a1) / 2, (z0 + z1) / 2), (t / 2, L / 2, (z1 - z0) / 2), collide=False)
        if collide:
            solid(s, c - 0.04, c + 0.04, a0, a1, z0, z1, friction=(0.45, 0.01, 0.001))


def glass_rail(s, axis, c, a0, a1, z, h=1.1):
    """Frameless glass guard on a steel shoe."""
    if axis == "x":
        s.box("steel_dark", ((a0 + a1) / 2, c, z + 0.05), ((a1 - a0) / 2, 0.04, 0.05), collide=False)
        s.box("glass", ((a0 + a1) / 2, c, z + h / 2 + 0.05), ((a1 - a0) / 2, 0.01, h / 2), collide=False)
        solid(s, a0, a1, c - 0.04, c + 0.04, z, z + h + 0.05)
    else:
        s.box("steel_dark", (c, (a0 + a1) / 2, z + 0.05), (0.04, (a1 - a0) / 2, 0.05), collide=False)
        s.box("glass", (c, (a0 + a1) / 2, z + h / 2 + 0.05), (0.01, (a1 - a0) / 2, h / 2), collide=False)
        solid(s, c - 0.04, c + 0.04, a0, a1, z, z + h + 0.05)


def handrail(s, pts, h=0.95, mat="steel_dark", posts=True):
    pts = [np.asarray(p, float) for p in pts]
    for p, q in zip(pts[:-1], pts[1:]):
        s.tube(mat, p + [0, 0, h], q + [0, 0, h], 0.02)
        if posts:
            n = max(1, int(np.linalg.norm(q - p) / 1.2))
            for i in range(n + 1):
                r = p + (q - p) * i / n
                s.tube(mat, r, r + [0, 0, h], 0.015)


def column(s, x, y, z0, z1, r=0.1, mat="steel_dark"):
    s.cylinder(mat, (x, y, (z0 + z1) / 2), r, (z1 - z0) / 2, collide=True, seg=20)


def stair(s, x, y, width, direction, z0, n, riser, tread, mat, style="solid", thick=0.08, side=None,
          z_base=None, nosing=0.02):
    """Straight stair. (x, y): centre of the bottom edge of the first riser; direction: '+x', '-x', '+y', '-y'
    (the way up). n risers; tread k (k = 0..n-2) has its top at z0 + (k+1) * riser. style 'solid' fills each step
    down to z_base (default z0), 'floating' draws thick cantilevered treads (open risers). Returns the plan point
    at the top edge."""
    d = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[direction]
    zb = z0 if z_base is None else z_base
    for k in range(n - 1):
        top = z0 + (k + 1) * riser
        r0, r1 = k * tread - nosing, (k + 1) * tread
        if style == "solid":
            lo = zb
        else:
            lo = top - thick
        a, b = sorted((r0 * d[0] + r0 * d[1] * 0, r1 * d[0]))
        if d[0] != 0:
            xa, xb = sorted((x + d[0] * r0, x + d[0] * r1))
            ya, yb = y - width / 2, y + width / 2
        else:
            ya, yb = sorted((y + d[1] * r0, y + d[1] * r1))
            xa, xb = x - width / 2, x + width / 2
        s.box(mat, ((xa + xb) / 2, (ya + yb) / 2, (lo + top) / 2), ((xb - xa) / 2, (yb - ya) / 2, (top - lo) / 2),
              collide=True, friction=s.materials[mat].friction, uv_world=True)
    L = (n - 1) * tread
    return (x + d[0] * L, y + d[1] * L)


def spiral_stair(s, cx, cy, r_out, z0, z1, n, a0=0.0, sweep=1.85 * math.pi, mat="steel_dark", tread_mat="wood_dark"):
    """Service spiral stair (plans: next to the laundry, both levels). n risers; treads are wedges."""
    riser = (z1 - z0) / n
    r_in = 0.09
    s.cylinder(mat, (cx, cy, (z0 + z1 + 1.0) / 2), r_in, (z1 + 1.0 - z0) / 2, collide=True)
    for k in range(n - 1):
        a = a0 + sweep * k / (n - 1)
        b = a + sweep / (n - 1) * 1.15
        top = z0 + (k + 1) * riser
        poly = [(cx + r_in * math.cos(a), cy + r_in * math.sin(a))]
        for t in np.linspace(a, b, 6):
            poly.append((cx + r_out * math.cos(t), cy + r_out * math.sin(t)))
        poly.append((cx + r_in * math.cos(b), cy + r_in * math.sin(b)))
        s.prism(tread_mat, poly, top - 0.05, top, uv_world=True)
        m = (a + b) / 2
        rm = (r_in + r_out) / 2
        s.collider("box", (cx + rm * math.cos(m), cy + rm * math.sin(m), top - 0.025),
                   ((r_out - r_in) / 2, rm * (b - a) / 2, 0.025), R=rot_z(m),
                   friction=s.materials[tread_mat].friction)
        p = (cx + (r_out - 0.03) * math.cos(m), cy + (r_out - 0.03) * math.sin(m), top)
        s.tube(mat, p, (p[0], p[1], top + 0.9), 0.01, seg=6)
        if k:
            s.tube(mat, prev, (p[0], p[1], top + 0.9), 0.014, seg=6)
        prev = (p[0], p[1], top + 0.9)
    s.note("service", "spiral stair between laundry and upper service room")


def slat_screen(s, axis, c, a0, a1, z0, z1, pitch=0.11, slat=(0.045, 0.07), mat="wood_light", gaps=(),
                collide=True):
    """Vertical timber slat screen (mashrabiya-like), see-through between slats; one collision box per run."""
    for (a, b, zb, zt) in rect_minus((a0, a1, z0, z1), gaps):
        n = max(1, int((b - a) / pitch))
        for i in range(n + 1):
            p = a + (b - a) * i / n
            if axis == "x":
                s.box(mat, (p, c, (zb + zt) / 2), (slat[0] / 2, slat[1] / 2, (zt - zb) / 2), collide=False)
            else:
                s.box(mat, (c, p, (zb + zt) / 2), (slat[1] / 2, slat[0] / 2, (zt - zb) / 2), collide=False)
        for z in (zb + 0.02, zt - 0.02):
            if axis == "x":
                s.box("steel_dark", ((a + b) / 2, c, z), ((b - a) / 2, 0.02, 0.02), collide=False)
            else:
                s.box("steel_dark", (c, (a + b) / 2, z), (0.02, (b - a) / 2, 0.02), collide=False)
        if collide:
            if axis == "x":
                solid(s, a, b, c - slat[1] / 2, c + slat[1] / 2, zb, zt)
            else:
                solid(s, c - slat[1] / 2, c + slat[1] / 2, a, b, zb, zt)


def planter(s, x0, x1, y0, y1, z0, h, wall_mat="steel_dark", t=0.03, plants="mixed", seed=0, soil="planting_bed"):
    import vegetation as V
    for (a, b, c, d) in ((x0, x1, y0, y0 + t), (x0, x1, y1 - t, y1), (x0, x0 + t, y0, y1), (x1 - t, x1, y0, y1)):
        s.box(wall_mat, ((a + b) / 2, (c + d) / 2, z0 + h / 2), ((b - a) / 2, (d - c) / 2, h / 2), collide=False)
    finish(s, soil, x0 + t, x1 - t, y0 + t, y1 - t, z0 + h - 0.05)
    solid(s, x0, x1, y0, y1, z0, z0 + h)
    rng = np.random.default_rng(seed)
    area = (x1 - x0) * (y1 - y0)
    n = max(1, int(area / 0.5))
    for k in range(n):
        p = (rng.uniform(x0 + 0.2, x1 - 0.2) if x1 - x0 > 0.4 else (x0 + x1) / 2,
             rng.uniform(y0 + 0.2, y1 - 0.2) if y1 - y0 > 0.4 else (y0 + y1) / 2, z0 + h - 0.05)
        kind = plants if plants != "mixed" else ["ginger", "shrub", "grass", "shrub"][rng.integers(4)]
        if kind == "ginger":
            V.ginger_clump(s, p, 0.5, n=5, height=1.1, seed=seed * 97 + k)
        elif kind == "grass":
            V.grass_tuft(s, p, 0.35, height=0.6, seed=seed * 97 + k)
        elif kind == "fern":
            V.leaf_cluster(s, np.add(p, (0, 0, 0.3)), 0.5, n=18, mat="leaf_light", seed=seed * 97 + k, droop=0.6)
        else:
            V.leaf_cluster(s, np.add(p, (0, 0, 0.35)), 0.45, n=22, mat="leaf_dark", seed=seed * 97 + k)


def door(s, name, axis, c, a_hinge, width, z0, h=2.4, swing=1, mat="wood_panel", t=0.045, qpos0=0.0,
         frame_mat="wood_dark", glass=False, open_max=1.75):
    """Hinged door leaf in an opening of an axis-aligned wall.
    axis='x': wall along X at y=c, leaf spans X from a_hinge towards a_hinge + width*sign(width)."""
    sgn = 1 if width > 0 else -1
    w = abs(width)
    if axis == "x":
        fr = Frame((a_hinge, c, z0), yaw=0.0 if sgn > 0 else math.pi)
    else:
        fr = Frame((c, a_hinge, z0), yaw=math.pi / 2 if sgn > 0 else -math.pi / 2)
    rng = (0, open_max) if swing > 0 else (-open_max, 0)
    mass = 18.0 if not glass else 25.0
    b = s.body(name, fr, joint=dict(type="hinge", axis=(0, 0, 1), range=rng, damping=3.0, frictionloss=2.0,
                                    armature=0.2, qpos0=qpos0), mass=mass,
               inertia=(mass * h * h / 12, mass * (w * w + h * h) / 12, mass * w * w / 3))
    b.com = (w / 2, 0, h / 2)
    with s.in_body(b):
        if glass:
            for zz in (0.05, h - 0.05):
                s.box(frame_mat, (w / 2, 0, zz), (w / 2, t / 2, 0.05), collide=False)
            for xx in (0.05, w - 0.05):
                s.box(frame_mat, (xx, 0, h / 2), (0.05, t / 2, h / 2), collide=False)
            s.box("glass", (w / 2, 0, h / 2), (w / 2 - 0.1, 0.01, h / 2 - 0.1), collide=False)
            s.collider("box", (w / 2, 0, h / 2), (w / 2 - 0.01, t / 2, h / 2 - 0.01))
        else:
            s.box(mat, (w / 2, 0, h / 2), (w / 2 - 0.004, t / 2, h / 2 - 0.004), collide=True)
        s.box("steel_dark", (w - 0.08, t / 2 + 0.03, 1.0), (0.01, 0.01, 0.15), collide=False)
        s.box("steel_dark", (w - 0.08, -t / 2 - 0.03, 1.0), (0.01, 0.01, 0.15), collide=False)
    s.note("articulated", name)
    return b


def pivot_doors(s, prefix, axis, c, a0, a1, z0, h, n, qpos0=1.35, frame_mat="wood_light", t=0.05):
    """Row of tall glazed pivot doors with timber frames (photos 8, 9, 11, 12, 14), pivot at 1/4 width."""
    w = (a1 - a0) / n
    for i in range(n):
        a = a0 + w * i
        piv = a + w * 0.25
        yaw = 0.0 if axis == "x" else math.pi / 2
        fr = Frame((piv, c, z0) if axis == "x" else (c, piv, z0), yaw=yaw)
        mass = 45.0
        b = s.body(f"{prefix}{i}", fr, joint=dict(type="hinge", axis=(0, 0, 1), range=(-1.65, 1.65), damping=1.0,
                                                  frictionloss=0.3, armature=0.3, qpos0=qpos0), mass=mass,
                   inertia=(mass * h * h / 12, mass * (w * w + h * h) / 12, mass * w * w / 12 + mass * (w / 4) ** 2))
        b.com = (w * 0.25, 0, h / 2)
        with s.in_body(b):
            x0, x1 = -w * 0.25 + 0.01, w * 0.75 - 0.01
            for zz, hh in ((0.06, 0.06), (h - 0.06, 0.06)):
                s.box(frame_mat, ((x0 + x1) / 2, 0, zz), ((x1 - x0) / 2, t / 2, hh), collide=False)
            for xx in (x0 + 0.06, x1 - 0.06):
                s.box(frame_mat, (xx, 0, h / 2), (0.06, t / 2, h / 2), collide=False)
            s.box("glass", ((x0 + x1) / 2, 0, h / 2), ((x1 - x0) / 2 - 0.12, 0.008, h / 2 - 0.12), collide=False)
            s.collider("box", ((x0 + x1) / 2, 0, h / 2), ((x1 - x0) / 2, t / 2, h / 2 - 0.02))
        s.note("articulated", f"{prefix}{i}")


def sliding_glass(s, name, axis, c, a0, a1, z0, h, travel, frame_mat="wood_dark", qpos0=0.0, offset=0.06):
    """One sliding glass panel (slide joint along the wall), placed just inside the fixed glazing line."""
    w = a1 - a0
    fr = Frame(((a0 + a1) / 2, c + offset, z0) if axis == "x" else (c + offset, (a0 + a1) / 2, z0),
               yaw=0.0 if axis == "x" else math.pi / 2)
    b = s.body(name, fr, joint=dict(type="slide", axis=(1, 0, 0), range=(min(0, travel), max(0, travel)),
                                    damping=30.0, frictionloss=10.0, armature=2.0, qpos0=qpos0), mass=40.0,
               inertia=(15.0, 15.0, 4.0))
    b.com = (0, 0, h / 2)
    with s.in_body(b):
        for zz in (0.04, h - 0.04):
            s.box(frame_mat, (0, 0, zz), (w / 2, 0.025, 0.04), collide=False)
        for xx in (-w / 2 + 0.04, w / 2 - 0.04):
            s.box(frame_mat, (xx, 0, h / 2), (0.04, 0.025, h / 2), collide=False)
        s.box("glass", (0, 0, h / 2), (w / 2 - 0.08, 0.008, h / 2 - 0.08), collide=False)
        s.collider("box", (0, 0, h / 2), (w / 2, 0.025, h / 2 - 0.02))
    s.note("articulated", name)
    return b


# ================================================================================================================
# sea level: deck, pier, pavilion (spa + games room)
# ================================================================================================================
NET = (-23.8, -21.4, 8.6, 12.2)
DECK_PALMS = [(-19.4, 5.0), (-19.6, 9.0), (-19.8, 13.0), (-25.6, 15.4)]


def sea_deck(s):
    holes = [NET] + [(x - 0.45, x + 0.45, y - 0.45, y + 0.45) for x, y in DECK_PALMS]
    slab(s, -24.5, -17.5, -1.7, 17.2, Z_DECK, 0.3, top="deck_grey", side="deck_grey", holes=holes, top_uv_rot=0.0)
    slab(s, -27.0, -24.5, 6.0, 17.2, Z_DECK, 0.3, top="deck_grey", side="deck_grey", holes=holes)
    slab(s, -17.5, -13.4, 14.0, 17.2, Z_DECK, 0.3, top="deck_grey", side="deck_grey")
    slab(s, -27.0, -24.5, -1.7, 6.0, Z_DECK - 0.2, 0.3, top="deck_brown", side="deck_brown")       # lower step
    s.note("sea deck", "lower deck strip 0.20 m step (darker timber)")
    for x, y in DECK_PALMS:                                                                  # planting holes
        finish(s, "planting_bed", x - 0.45, x + 0.45, y - 0.45, y + 0.45, Z_DECK - 0.03)
        solid(s, x - 0.45, x + 0.45, y - 0.45, y + 0.45, Z_DECK - 0.33, Z_DECK - 0.03)
    # hammock net inset (photo 4): black net 8 cm below the deck, orange cushions
    x0, x1, y0, y1 = NET
    s.box("net_black", ((x0 + x1) / 2, (y0 + y1) / 2, Z_DECK - 0.09), ((x1 - x0) / 2, (y1 - y0) / 2, 0.005),
          collide=True, friction=(1.0, 0.03, 0.004))
    for (a, b, c, d) in ((x0 - 0.06, x1 + 0.06, y0 - 0.06, y0), (x0 - 0.06, x1 + 0.06, y1, y1 + 0.06),
                         (x0 - 0.06, x0, y0, y1), (x1, x1 + 0.06, y0, y1)):
        s.box("steel_dark", ((a + b) / 2, (c + d) / 2, Z_DECK - 0.04), ((b - a) / 2, (d - c) / 2, 0.04),
              collide=False)
    rng = np.random.default_rng(4)
    for k in range(13):
        cx, cy = rng.uniform(x0 + 0.35, x1 - 0.35), rng.uniform(y0 + 0.35, y1 - 0.35)
        big = rng.random() < 0.6
        hx, hy = (0.33, 0.3) if big else (0.2, 0.2)
        mat = "fabric_orange" if rng.random() < 0.75 else "fabric_yellow"
        s.rbox(mat, (cx, cy, Z_DECK - 0.03 + 0.06 * (k % 3)), (hx, hy, 0.07), 0.05, R=rot_z(rng.uniform(-0.6, 0.6)))
    s.note("sea deck", "hammock net with orange cushions")
    # long built-in bench beside the net (photo 1)
    s.box("deck_grey", (-21.1, 7.6, Z_DECK + 0.21), (0.22, 4.6, 0.21))
    # concrete piers under the deck, cleats on the sea edge
    for x in (-26.6, -23.0, -19.5):
        for y in np.arange(-1.0, 17.0, 3.0):
            s.cylinder("concrete_light", (x, y, Z_DECK - 2.2), 0.18, 1.9, collide=False)
    for y in (0.0, 4.0, 8.0, 12.0, 16.0):
        s.box("stainless", (-26.9, y, Z_DECK + 0.03), (0.04, 0.12, 0.03), collide=False)


def pier(s):
    """Boat landing 2 m below the deck, reached by a timber stair (photo 2)."""
    slab(s, -32.0, -27.0, 2.0, 10.0, Z_PIER, 0.3, top="deck_brown", side="deck_brown")
    slab(s, -32.0, -30.3, 10.0, 11.4, Z_PIER, 0.3, top="deck_brown", side="deck_brown")
    # stair from the pier (x = -30.3) up east to the deck edge (x = -27.0): 12 risers of 0.167 m
    stair(s, -30.3, 10.7, 1.4, "+x", Z_PIER, 12, (Z_DECK - Z_PIER) / 12, 3.3 / 11, "deck_brown", style="floating",
          thick=0.06)
    for x in np.arange(-31.8, -26.9, 2.0):
        for y in (2.2, 9.8):
            s.cylinder("wood_dark", (x, y, Z_PIER - 1.8), 0.14, 1.9, collide=False)
            s.cylinder("wood_dark", (x, y, Z_PIER + 0.25), 0.12, 0.25, collide=True)
    handrail(s, [(-30.3, 11.42, Z_PIER), (-27.0, 11.42, Z_DECK)], posts=True)
    s.note("pier", "boat landing with mooring posts, 12-riser timber stair")


def pavilion(s):
    """Spa + games pavilion under the planted roof, opening onto the sea deck (photo 13)."""
    X0, X1, Y0, Y1 = -17.5, -13.4, 0.0, 14.0
    zc = -0.65
    slab(s, X0, X1, Y0, Y1, Z_DECK, 0.35, top="granite_tiles", side="concrete_light")
    slab(s, X0 - 0.2, X1, Y0 - 0.2, Y1 + 0.2, -0.3, 0.35, top=None, under="wood_ceiling", side="wood_dark")
    finish(s, "planting_bed", X0 - 0.2, -13.6, Y0 - 0.2, Y1 + 0.2, -0.25)
    import vegetation as V
    rng = np.random.default_rng(11)
    for k in range(70):
        V.grass_tuft(s, (rng.uniform(X0, -13.7), rng.uniform(Y0, Y1), -0.25), 0.35, height=rng.uniform(0.4, 0.8),
                     seed=200 + k)
    # walls
    wall(s, "y", X1 - 0.1, Y0, Y1, Z_DECK, zc, 0.2, "white_boards", None)              # retaining wall (east)
    wall(s, "x", Y0 + 0.1, X0, X1, Z_DECK, zc, 0.2, "mashrabiya", "plaster")
    wall(s, "x", Y1 - 0.1, X0, X1, Z_DECK, zc, 0.2, "plaster", "mashrabiya")
    # west facade: slatted wall, pivot doors, sliding glass (photo 13)
    fz0, fz1 = Z_DECK, zc
    wall(s, "y", X0, 0.2, 1.0, fz0, fz1, 0.14, "mashrabiya", "plaster")
    wall(s, "y", X0, 5.8, 6.2, fz0, fz1, 0.14, "mashrabiya", "plaster")
    wall(s, "y", X0, 12.2, 13.8, fz0, fz1, 0.14, "mashrabiya", "plaster")
    door(s, "pav_spa_door", "y", X0, 1.05, 1.1, fz0, h=2.6, swing=-1, mat="mashrabiya", qpos0=-1.2, t=0.06)
    glass_wall(s, "y", X0, 2.2, 5.8, fz0, fz1, spacing=1.8)
    sliding_glass(s, "pav_spa_slide", "y", X0, 2.2, 4.0, fz0, fz1 - fz0 - 0.02, 1.7, qpos0=0.0, offset=0.08)
    door(s, "pav_game_door_a", "y", X0, 6.25, 1.15, fz0, h=2.6, swing=-1, mat="mashrabiya", qpos0=-1.45, t=0.06)
    door(s, "pav_game_door_b", "y", X0, 8.55, -1.15, fz0, h=2.6, swing=1, mat="mashrabiya", qpos0=1.45, t=0.06)
    glass_wall(s, "y", X0, 8.6, 12.2, fz0, fz1, spacing=1.8)
    sliding_glass(s, "pav_game_slide", "y", X0, 8.6, 10.4, fz0, fz1 - fz0 - 0.02, 1.7, qpos0=1.2, offset=0.08)
    # facade slat cladding band above the openings and roof fascia
    slat_screen(s, "y", X0 - 0.14, 0.0, 14.0, zc - 0.35, -0.3, collide=False)
    # interior partition between spa and games room, sauna
    wall(s, "x", 5.95, X0 + 0.1, -14.6, Z_DECK, zc, 0.1, "plaster")
    door(s, "pav_inner_door", "x", 5.95, -14.55, 0.85, Z_DECK, h=2.4, swing=1, qpos0=1.4)
    FU.sauna(s, Frame((-17.2, 0.25, Z_DECK), yaw=0.0), W=2.0, D=2.1, H=2.4)
    s.note("pavilion", "sauna, massage room, games room, bar")
    s.spawns["pavilion"] = (-15.6, 9.5, Z_DECK, math.pi)
    return X0, X1


# ================================================================================================================
# social level: plateau, pool, lawn, house
# ================================================================================================================
POOL = (-13.1, -10.0, -1.2, 14.8)


def plateau(s):
    zb = Z_DECK
    x0, x1, y0, y1 = POOL
    # fill (collision only; tops drawn by finishes)
    solid(s, -13.35, x0, -1.7, 15.0, zb, -0.06)                     # infinity wall
    solid(s, x0, x1, -1.7, y0, zb, 0.0)                             # south coping
    solid(s, x0, x1, y1, 15.0, zb, 0.0)                             # north coping
    solid(s, x1, -7.4, -1.7, 15.0, zb, 0.0, friction=s.materials["deck_grey"].friction)
    solid(s, -7.4, -5.9, -1.7, 15.0, zb, 0.0, friction=s.materials["lawn"].friction)
    solid(s, -5.9, -0.5, -1.7, 16.6, zb, 0.0, friction=s.materials["lawn"].friction)
    solid(s, -13.35, -5.9, 16.7, y_north(-9.0), zb, 0.0)            # north garden beyond the stair wall
    solid(s, -5.9, 5.0, 16.6, y_north(0.0), zb, 0.0)
    finish(s, "stone_blocks", -13.35, x0, -1.7, 15.0, -0.06)
    finish(s, "concrete_light", x0, x1, -1.7, y0, 0.0)
    finish(s, "concrete_light", x0, x1, y1, 15.0, 0.0)
    finish(s, "deck_grey", x1, -7.4, -1.7, 15.0, 0.0, uv_rot=math.pi / 2)
    finish(s, "lawn", -7.4, -5.9, -1.7, 15.0, 0.0)
    finish(s, "lawn", -5.9, -0.5, -1.7, 16.6, 0.0)
    finish(s, "planting_bed", -13.35, -5.9, 16.7, y_north(-9.0), 0.0)
    finish(s, "lawn", -5.9, 5.0, 16.6, y_north(0.0), 0.0)
    # catch channel of the infinity edge, west retaining face of the plateau (stone)
    s.box("water_dark", (-13.48, 7.0, -0.29), (0.12, 8.6, 0.02), collide=False)
    s.box("stone_wall", (-13.6, 7.0, -0.3), (0.02, 8.6, 0.25), collide=False)
    s.box("stone_wall", (-13.35, -1.45, -1.85), (0.02, 0.25, 1.6), collide=False)
    s.note("pool", "infinity pool 3.1 x 16 m, depth 1.4 m with a 0.45 m bench at the south end")


def pool(s):
    x0, x1, y0, y1 = POOL
    yb = 1.5                                               # end of the shallow bench
    # floor and walls (collision + mosaic)
    solid(s, x0, x1, y0, yb, -0.75, -0.45)
    solid(s, x0, x1, yb, y1, -1.7, -1.4)
    solid(s, x0, x1, yb - 0.02, yb + 0.0, -1.4, -0.45)
    finish(s, "pool_mosaic", x0, x1, y0, yb, -0.45)
    finish(s, "pool_mosaic", x0, x1, yb, y1, -1.4)
    s.box("pool_mosaic", ((x0 + x1) / 2, yb + 0.01, -0.93), ((x1 - x0) / 2, 0.01, 0.47), collide=False)
    for (cx, cy, hx, hy, z0) in (((x0 + x1) / 2, y0 - 0.01, (x1 - x0) / 2, 0.01, -0.45),
                                 ((x0 + x1) / 2, y1 + 0.01, (x1 - x0) / 2, 0.01, -1.4)):
        s.box("pool_mosaic", (cx, cy, (z0 - 0.0) / 2), (hx, hy, -z0 / 2), collide=False)
    for x in (x0 - 0.01, x1 + 0.01):
        s.box("pool_mosaic", (x, (y0 + yb) / 2, -0.225 - 0.03), (0.01, (yb - y0) / 2, 0.225 - 0.03),
              collide=False)
        s.box("pool_mosaic", (x, (yb + y1) / 2, -0.7 - 0.03), (0.01, (y1 - yb) / 2, 0.7 - 0.03), collide=False)
    # entry steps at the south-east corner, down to the bench
    for (xa, top) in ((x1 - 0.6, -0.15), (x1 - 1.2, -0.30)):
        s.box2("pool_mosaic", (xa, y0, -0.45), (xa + 0.6, y0 + 1.7, top))
    s.planes.append(dict(name="pool_water", mat="water_pool", pos=((x0 + x1) / 2, (y0 + y1) / 2, -0.07),
                         half=((x1 - x0) / 2, (y1 - y0) / 2)))
    # underwater lights (emissive)
    for y in (3.0, 7.0, 11.0):
        s.box("lamp_glow", (x1 - 0.005, y, -0.6), (0.004, 0.12, 0.06), collide=False)


def social_house(s):
    """Varanda, living room, kitchens, service, wine cellar: floors, walls, columns, doors."""
    # structural slab and finishes
    solid(s, -0.5, 21.2, 0.0, 16.6, -0.35, 0.0, friction=s.materials["granite_tiles"].friction)
    finish(s, "granite_tiles", -0.5, 6.2, 0.0, 16.6, 0.0)
    finish(s, "granite_tiles", 6.2, 15.0, 0.0, 4.1, 0.0)
    finish(s, "dark_tiles", 6.2, 16.0, 4.1, 16.4, 0.0)
    finish(s, "granite_tiles", 15.0, 21.2, 0.0, 10.1, 0.0)
    finish(s, "wood_floor", 16.2, 21.2, 10.1, 16.4, 0.0, uv_rot=math.pi / 2)
    s.box("stone_wall", (21.25, 8.3, -1.7), (0.05, 8.3, 1.65), collide=False)
    # south side path (pavers) along the boundary, planters
    solid(s, -0.5, 18.0, -1.7, 0.0, Z_DECK, 0.0, friction=s.materials["pavers"].friction)
    finish(s, "pavers", -0.5, 18.0, -1.7, 0.0, 0.0, uv_rot=math.pi / 2)
    for x in np.arange(1.0, 17.0, 3.2):
        planter(s, x, x + 1.4, -1.65, -1.15, 0.0, 0.5, wall_mat="concrete_light", seed=int(x * 10))
    # varanda columns (photo 8: dark round steel), interior columns on the pivot-door line
    for y in (0.15, 4.1, 8.2, 12.3, 16.3):
        column(s, 0.0, y, 0.0, Z_CEIL_SOC, 0.1)
    for y in (8.2, 12.3):
        column(s, 6.4, y, 0.0, Z_CEIL_SOC, 0.09)
    # ---- walls (z 0 .. 3.0 under the upper slab, 3.6 where they reach the upper floor)
    zc = Z_CEIL_SOC
    wall(s, "x", 0.1, 0.0, 6.2, 0.0, zc, 0.2, "wood_panel", "wood_panel")                     # gourmet south
    wall(s, "y", 6.3, 0.2, 4.1, 0.0, zc, 0.2, "wood_panel", "wood_panel", openings=[(2.6, 3.6, 0.0, 2.6)])
    door(s, "kitchen_door_w", "y", 6.3, 2.6, 1.0, 0.0, h=2.6, swing=1, qpos0=0.0, mat="wood_panel")
    wall(s, "x", 0.1, 6.2, 21.2, 0.0, zc, 0.2, "plaster", "plaster",
         openings=[(8.0, 13.4, 1.1, 2.5), (16.7, 17.6, 0.0, 2.4)])                                # south facade
    glass_wall(s, "x", 0.1, 8.0, 13.4, 1.1, 2.5, frame="steel_dark", spacing=1.35, sill=True)
    door(s, "service_entry", "x", 0.1, 16.7, 0.9, 0.0, h=2.4, swing=1, qpos0=0.0, mat="wood_panel")
    wall(s, "x", 4.2, 6.4, 16.0, 0.0, zc, 0.2, "wood_panel", "white_boards")                  # kitchen / living
    s.box("steel_dark", (11.0, 4.2, zc + 0.3), (4.9, 0.12, 0.3), collide=False)                # slab edge band
    wall(s, "y", 16.1, 4.3, 16.4, 0.0, Z_UP, 0.2, "wood_panel", "plaster",
         openings=[(13.6, 14.9, 0.0, 2.8)])                                                    # living east wall
    door(s, "cellar_pivot", "y", 16.1, 13.6, 1.3, 0.0, h=2.8, swing=1, qpos0=0.0, mat="wood_panel", t=0.06)
    wall(s, "x", 16.5, 6.2, 21.2, 0.0, Z_UP, 0.2, "white_boards", "stone_wall")                # north (TV) wall
    wall(s, "y", 21.1, 0.0, 16.6, 0.0, Z_UP, 0.2, "plaster", None)                              # east retaining
    # service block partitions
    wall(s, "y", 14.95, 0.2, 4.1, 0.0, zc, 0.12, "wood_panel", "plaster", openings=[(1.6, 2.5, 0.0, 2.3)])
    door(s, "kitchen_door_e", "y", 14.95, 1.6, 0.9, 0.0, h=2.3, swing=-1, qpos0=-0.0, mat="plaster")
    wall(s, "x", 3.55, 15.0, 21.0, 0.0, zc, 0.12, "plaster", openings=[(16.3, 17.3, 0.0, 2.3)])
    wall(s, "x", 4.85, 16.2, 21.0, 0.0, zc, 0.12, "plaster", openings=[(16.5, 17.3, 0.0, 2.2), (18.2, 19.0, 0.0, 2.3),
                                                                     (19.4, 20.2, 0.0, 2.2)])
    door(s, "lavabo_door", "x", 4.85, 16.5, 0.8, 0.0, h=2.2, swing=1, qpos0=0.0)
    door(s, "staffbath_door", "x", 4.85, 19.4, 0.8, 0.0, h=2.2, swing=1, qpos0=0.0)
    wall(s, "y", 18.2, 4.9, 7.45, 0.0, zc, 0.12, "plaster")
    wall(s, "y", 19.0, 4.9, 7.45, 0.0, zc, 0.12, "plaster")
    wall(s, "x", 7.45, 16.2, 21.0, 0.0, zc, 0.12, "plaster", openings=[(18.2, 19.0, 0.0, 2.3)])
    wall(s, "x", 10.08, 16.2, 21.0, 0.0, zc, 0.15, "plaster", openings=[(17.2, 18.1, 0.0, 2.3)])
    door(s, "pantry_cellar_door", "x", 10.08, 17.2, 0.9, 0.0, h=2.3, swing=1, qpos0=0.0)
    finish(s, "bath_stone", 16.2, 18.15, 4.9, 7.4, 0.004)
    finish(s, "bath_stone", 19.05, 21.0, 4.9, 7.4, 0.004)
    # ceilings
    ceiling(s, "wood_ceiling", -2.4, 7.8, 0.0, 16.4, zc, uv_rot=math.pi / 2)
    ceiling(s, "wood_ceiling", 7.8, 15.0, 0.0, 4.1, zc, uv_rot=math.pi / 2)
    ceiling(s, "plaster", 15.0, 21.2, 0.0, 16.4, zc)
    # pivot doors between varanda and living room (open as in the photos)
    pivot_doors(s, "living_pivot", "y", 6.25, 4.3, 16.3, 0.0, zc - 0.02, 10, qpos0=-1.35)
    s.box("wood_light", (6.25, 10.3, zc - 0.01), (0.08, 6.0, 0.01), collide=False)
    s.spawns["living"] = (12.9, 9.0, 0.0, math.pi / 2)
    s.spawns["varanda"] = (1.2, 11.0, 0.0, -math.pi / 2)
    s.spawns["kitchen"] = (10.5, 2.3, 0.0, 0.0)
    s.spawns["lawn"] = (-3.5, 8.0, 0.0, 0.0)


def living_stair(s):
    """Wide timber stair with deep cantilevered treads along the east wall (photos 9, 12, 14):
    22 risers of 0.1636 m, 21 treads of 0.4286 m, bottom at y = 13.1 rising south to the upper floor at y = 4.1."""
    n, riser, tread = 22, Z_UP / 22, 9.0 / 21
    for k in range(n - 1):
        top = (k + 1) * riser
        y1 = 13.1 - k * tread
        y0 = y1 - tread
        s.box("wood_dark", (14.97, (y0 + y1) / 2, top - 0.06), (0.97, tread / 2 + 0.01, 0.06), collide=True,
              friction=s.materials["wood_dark"].friction, uv_world=True)
    s.note("living room", "cantilevered timber stair, 22 x 16.4 cm risers, 43 cm treads, no railing (as built)")


def upper_level(s):
    zf, zc = Z_UP, Z_CEIL_UP
    # ---- slabs (collision) and the void over the living room
    solid(s, -2.4, 7.8, 0.0, 16.55, Z_CEIL_SOC, zf)                     # bedroom wing (cantilevers to x = -2.4)
    SPIRAL_HOLE = (18.9, 20.9, 0.85, 2.85)
    for (a, b, c, d) in rect_minus((7.8, 25.7, 0.0, 4.1), [SPIRAL_HOLE]):
        solid(s, a, b, c, d, Z_CEIL_SOC, zf)                            # lounge, service, study
    spiral_stair(s, 19.9, 1.85, 0.95, Z_SOC, zf, 18, a0=-0.5 * math.pi)
    for (a, b, c, d) in ((18.9, 20.9, 2.85, 2.9), (20.85, 20.9, 0.85, 2.9)):
        s.box("steel_dark", ((a + b) / 2, (c + d) / 2, zf + 0.5), ((b - a) / 2, (d - c) / 2, 0.5), collide=True)
    solid(s, 15.9, 17.75, 4.1, 16.55, Z_CEIL_SOC, zf)                   # east strip
    solid(s, 17.75, 25.7, 4.1, 16.55, Z_CEIL_SOC, zf, friction=s.materials["pavers"].friction)  # garden
    s.box("steel_dark", (-2.42, 8.2, Z_CEIL_SOC + 0.3), (0.03, 8.3, 0.3), collide=False)        # slab fascias
    s.box("steel_dark", (7.82, 10.3, Z_CEIL_SOC + 0.3), (0.03, 6.2, 0.3), collide=False)
    s.box("steel_dark", (2.7, 16.58, Z_CEIL_SOC + 0.3), (5.1, 0.03, 0.3), collide=False)
    s.box("steel_dark", (2.7, -0.02, Z_CEIL_SOC + 0.3), (5.1, 0.03, 0.3), collide=False)
    # ---- finishes
    for (y0, y1) in ((0.15, 4.97), (5.05, 10.15), (10.25, 16.4)):
        finish(s, "wood_floor", -2.3, 2.5, y0, y1, zf)
    finish(s, "wood_floor", 2.5, 6.0, 10.25, 13.3, zf)
    finish(s, "wood_floor", 2.5, 6.0, 3.05, 4.97, zf)
    finish(s, "wood_floor", 2.5, 6.0, 5.05, 6.95, zf)
    finish(s, "bath_stone", 2.5, 7.7, 13.3, 16.4, zf)
    finish(s, "bath_stone", 2.5, 6.0, 6.95, 10.15, zf)
    finish(s, "bath_stone", 2.5, 6.0, 0.15, 3.05, zf)
    finish(s, "granite_tiles", 6.0, 7.8, 0.15, 13.3, zf)
    for (a, b, c, d) in rect_minus((7.8, 20.8, 0.12, 4.1), [SPIRAL_HOLE]):
        finish(s, "granite_tiles", a, b, c, d, zf)
    finish(s, "wood_floor", 20.8, 25.7, 0.12, 4.1, zf)
    finish(s, "granite_tiles", 16.85, 17.75, 4.1, 16.55, zf)
    finish(s, "gravel_white", 15.9, 16.0, 4.1, 16.55, zf)
    # ---- void guards: glass rail on the lounge edge, slat screen (mashrabiya) along the bedroom corridor
    glass_rail(s, "x", 4.15, 7.85, 13.95, zf)
    slat_screen(s, "y", 7.75, 0.15, 13.3, zf, zc, gaps=[(0.9, 2.5, zf, zf + 2.5)])
    s.note("upper lounge", "glass guard on the void edge; timber slat screen along the bedroom corridor")
    # east strip planter beside the stair (photo 9)
    planter(s, 16.0, 16.85, 4.6, 16.2, zf, 0.6, wall_mat="steel_dark", plants="mixed", seed=5)
    # ---- facades
    # west (sea) facade of the suites: sliding glass, frameless guard inside, timber pivot shutters outside
    for (y0, y1) in ((0.15, 4.97), (5.05, 10.15), (10.25, 16.4)):
        glass_wall(s, "y", -2.35, y0, y1, zf, zc, frame="wood_dark", spacing=1.7)
        s.box("glass", (-2.2, (y0 + y1) / 2, zf + 0.55), (0.01, (y1 - y0) / 2 - 0.05, 0.5), collide=False)
    for k, y in enumerate(np.arange(0.4, 16.4, 1.3)):
        ang = [0.0, 1.2, 0.0, 1.45, 0.3][k % 5]
        R = rot_z(ang)
        c = np.array([-2.55, y, (zf + zc) / 2]) + R @ np.array([0, 0.6, 0]) - np.array([0, 0.6, 0])
        s.box("mashrabiya", c, (0.03, 0.6, (zc - zf) / 2 - 0.05), R=R, collide=False)
    wall(s, "x", 0.07, -2.4, 7.8, zf, zc, 0.14, "wood_panel", "plaster")                     # suite 3 south wall
    wall(s, "x", 16.48, -2.4, 7.8, zf, zc, 0.14, "plaster", "wood_panel")                    # master north wall
    # suite partitions and bathroom / wardrobe blocks
    wall(s, "x", 5.01, -2.3, 6.06, zf, zc, 0.12, "plaster")
    wall(s, "x", 10.2, -2.3, 7.75, zf, zc, 0.12, "plaster", openings=[(6.3, 7.2, zf, zf + 2.4)])
    door(s, "master_door", "x", 10.2, 6.3, 0.9, zf, h=2.4, swing=1, qpos0=1.2)
    wall(s, "y", 2.56, 0.15, 16.4, zf, zc, 0.12, "plaster", "plaster",
         openings=[(3.1, 4.2, zf, zf + 2.4), (5.75, 6.9, zf, zf + 2.4), (10.6, 12.9, zf, zf + 2.4)])
    wall(s, "y", 6.06, 0.15, 10.15, zf, zc, 0.12, "plaster", "plaster",
         openings=[(3.25, 4.1, zf, zf + 2.4), (5.9, 6.8, zf, zf + 2.4)])
    door(s, "suite3_door", "y", 6.06, 3.25, 0.85, zf, h=2.4, swing=-1, qpos0=-0.9)
    door(s, "suite2_door", "y", 6.06, 5.9, 0.85, zf, h=2.4, swing=-1, qpos0=-0.9)
    wall(s, "x", 13.36, 2.62, 7.75, zf, zc, 0.12, "plaster", "bath_stone", openings=[(4.0, 4.9, zf, zf + 2.4)])
    door(s, "master_bath_door", "x", 13.36, 4.0, 0.9, zf, h=2.4, swing=1, qpos0=0.0)
    wall(s, "y", 6.06, 10.25, 13.3, zf, zc, 0.12, "plaster", "plaster", openings=[(10.25, 13.3, zf, zf + 2.6)])
    wall(s, "x", 6.95, 2.62, 6.0, zf, zc, 0.12, "plaster", "bath_stone", openings=[(4.2, 5.0, zf, zf + 2.3)])
    door(s, "suite2_bath_door", "x", 6.95, 4.2, 0.8, zf, h=2.3, swing=1, qpos0=0.0)
    wall(s, "x", 3.05, 2.62, 6.0, zf, zc, 0.12, "bath_stone", "plaster", openings=[(4.2, 5.0, zf, zf + 2.3)])
    door(s, "suite3_bath_door", "x", 3.05, 4.2, 0.8, zf, h=2.3, swing=-1, qpos0=0.0)
    wall(s, "y", 7.7, 13.3, 16.4, zf, zc, 0.12, "bath_stone", "wood_panel")
    # lounge / service / study
    glass_wall(s, "x", 0.06, 7.8, 17.7, zf, zc, frame="wood_dark", spacing=1.65)
    wall(s, "y", 17.72, 0.12, 4.1, zf, zc, 0.12, "wood_panel", "plaster", openings=[(0.6, 1.5, zf, zf + 2.4)])
    door(s, "upper_service_door", "y", 17.72, 0.6, 0.9, zf, h=2.4, swing=1, qpos0=0.0, mat="wood_panel")
    wall(s, "x", 2.2, 17.78, 19.0, zf, zc, 0.1, "plaster", openings=[(18.0, 18.8, zf, zf + 2.3)])
    wall(s, "y", 19.0, 2.2, 4.1, zf, zc, 0.1, "plaster")
    wall(s, "y", 20.8, 0.12, 4.1, zf, zc, 0.12, "plaster", "plaster", openings=[(0.5, 1.4, zf, zf + 2.3)])
    door(s, "study_door", "y", 20.8, 0.5, 0.9, zf, h=2.3, swing=1, qpos0=0.0)
    wall(s, "x", 0.06, 17.7, 25.7, zf, 6.4, 0.12, "plaster", "plaster")
    wall(s, "x", 4.06, 17.75, 20.8, zf, zc, 0.12, "plaster", "wood_panel")
    glass_wall(s, "x", 4.06, 20.8, 25.7, zf, 6.4, frame="steel_dark", spacing=1.25)
    sliding_glass(s, "study_slide", "x", 4.06, 20.9, 22.1, zf, 2.7, 1.1, qpos0=0.9, offset=-0.07)
    glass_wall(s, "y", 25.64, 0.12, 4.1, zf, 6.4, frame="steel_dark", spacing=1.3)
    ceiling(s, "plaster", 20.8, 25.7, 0.12, 4.1, 6.4)
    slab(s, 20.7, 25.85, -0.05, 4.2, 6.75, 0.35, top="planting_bed", under=None, side="wood_dark")
    # east facade on the garden: glazed pivot doors (photos 9, 11)
    pivot_doors(s, "garden_pivot", "y", 17.7, 4.15, 16.45, zf, zc - zf - 0.02, 10, qpos0=-1.3)
    # north clerestory over the TV wall and north of the east strip
    glass_wall(s, "x", 16.5, 7.8, 17.75, zf, zc, frame="steel_dark", spacing=1.5)
    # columns carrying the roof
    for (x, y) in ((7.85, 4.15), (7.85, 16.45), (13.95, 4.2), (17.7, 4.12), (17.7, 16.5)):
        column(s, x, y, zf, zc, 0.08)
    ceiling(s, "wood_ceiling", -4.75, 20.7, -1.5, 17.9, zc, uv_rot=0.0)
    s.spawns["upper_lounge"] = (12.6, 3.3, zf, 0.0)
    s.spawns["suite1"] = (0.3, 7.6, zf, math.pi)
    s.spawns["garden"] = (19.5, 8.5, zf, 0.0)


def roof(s):
    """Thin hipped metal roof with big overhangs (photos 2, 3, 5, 11)."""
    x0, x1, y0, y1 = -4.75, 20.7, -1.5, 17.9
    ze = Z_CEIL_UP + 0.12
    solid(s, x0, x1, y0, y1, Z_CEIL_UP, ze)
    for (a, b, c, d) in ((x0, x1, y0, y0 + 0.06), (x0, x1, y1 - 0.06, y1), (x0, x0 + 0.06, y0, y1),
                         (x1 - 0.06, x1, y0, y1)):
        s.box("steel_dark", ((a + b) / 2, (c + d) / 2, Z_CEIL_UP + 0.07), ((b - a) / 2, (d - c) / 2, 0.07),
              collide=False)
    yc = (y0 + y1) / 2
    half_w = (y1 - y0) / 2
    rx0, rx1 = x0 + half_w, x1 - half_w
    zr = Z_ROOF_TOP
    P = {"a": (x0, y0, ze + 0.02), "b": (x1, y0, ze + 0.02), "c": (x1, y1, ze + 0.02), "d": (x0, y1, ze + 0.02),
         "e": (rx0, yc, zr), "f": (rx1, yc, zr)}
    tris = [("a", "b", "f"), ("a", "f", "e"), ("b", "c", "f"), ("c", "d", "e"), ("c", "e", "f"), ("d", "a", "e")]
    V = np.array([P[k] for t in tris for k in t], float)
    Fc = np.arange(len(V)).reshape(-1, 3)
    s.trimesh("roof_metal", V, Fc, smooth=False)
    # standing seams on the two long slopes, ending on the ridge or on the hip lines
    for x in np.arange(x0 + 0.5, x1 - 0.2, 0.5):
        t = min(1.0, (x - x0) / (rx0 - x0), (x1 - x) / (x1 - rx1))
        for sgn in (-1, 1):
            ye = yc + sgn * half_w
            p0 = (x, ye, ze + 0.035)
            p1 = (x, ye + (yc - ye) * t, ze + (zr - ze) * t + 0.015)
            if t > 0.02:
                s.tube("roof_metal", p0, p1, 0.01, seg=4)
    for (x, y) in ((4.0, 6.0), (8.5, 6.0), (12.5, 8.0)):                 # skylights (photo 2)
        s.box("glass", (x, y, zr - 0.25), (0.6, 0.35, 0.03), collide=False)
    s.note("roof", "hipped standing-seam roof, 25.5 x 19.4 m, timber soffit at 6.5 m, three skylights")


# ================================================================================================================
# outdoor stairs, upper garden, terrace, lift, parking, boundaries
# ================================================================================================================
def north_stairs(s):
    # social -> sea deck, between stone walls (photo 7): 21 risers x 0.162, 20 treads x 0.375
    stair(s, -13.4, 15.8, 1.2, "+x", Z_DECK, 21, (Z_SOC - Z_DECK) / 21, 7.5 / 20, "stone_blocks", style="solid",
          z_base=Z_DECK - 0.3)
    wall(s, "x", 15.1, -13.4, -5.9, Z_DECK, 0.45, 0.2, "stone_wall")
    wall(s, "x", 16.55, -13.4, -5.9, Z_DECK, 1.0, 0.3, "stone_wall")
    s.box("steel_dark", (-11.5, 15.8, 0.25), (0.4, 0.85, 0.35), collide=True)          # planter bridge (photo 7)
    import vegetation as V
    for k in range(6):
        V.grass_tuft(s, (-11.5, 15.15 + 0.27 * k, 0.6), 0.25, height=0.6, seed=400 + k)
    s.note("outdoor stairs", "social -> sea deck: 21 stone steps between rubble walls, planter bridge overhead")
    # social -> upper along the north facade: 22 risers x 0.1636, 21 treads x 0.286
    stair(s, 5.0, 17.3, 1.4, "+x", Z_SOC, 22, Z_UP / 22, 6.0 / 21, "stone_blocks", style="solid", z_base=Z_SOC)
    wall(s, "x", 18.15, 5.0, 11.0, Z_SOC, Z_UP + 0.6, 0.3, "stone_wall")
    solid(s, 5.0, 11.0, 18.3, y_north(8.0), Z_SOC, Z_UP)
    finish(s, "planting_bed", 5.0, 11.0, 18.3, y_north(8.0), Z_UP)
    finish(s, "planting_bed", 5.0, 11.0, 16.6, 16.62, Z_SOC)
    # upper north path (pavers) from the stair top to the garden, planting beyond
    solid(s, 11.0, 25.7, 16.55, y_north(18.0), Z_SOC, Z_UP, friction=s.materials["pavers"].friction)
    finish(s, "pavers", 11.0, 25.7, 16.55, 18.4, Z_UP)
    finish(s, "planting_bed", 11.0, 25.7, 18.4, y_north(18.0), Z_UP)
    s.note("outdoor stairs", "social -> upper: 22 stone steps along the north facade")
    # south service stair: social path -> east terrace, rough granite blocks (photo 15)
    stair(s, 18.0, -0.9, 1.4, "+x", Z_SOC, 21, Z_UP / 21, 7.5 / 20, "stone_blocks", style="solid", z_base=Z_SOC)
    wall(s, "x", -0.12, 18.0, 25.5, Z_SOC, Z_UP + 0.1, 0.2, "stone_wall")
    s.note("outdoor stairs", "social -> upper (south): 21 rough granite block steps")


def upper_garden(s):
    """Entrance garden: concrete pavers in gravel, reflecting pool, raised stone planter, frangipani (photo 11)."""
    import vegetation as V
    zf = Z_UP
    POOLR = (22.1, 25.5, 4.3, 10.15)
    finish(s, "gravel_white", 17.75, 25.7, 4.1, 16.55, zf - 0.002)
    rng = np.random.default_rng(21)
    y = 4.35
    while y < 14.6:
        xa = 17.85 + rng.uniform(0, 0.4)
        while xa < 25.4:
            L = rng.uniform(1.4, 3.2)
            xb = min(xa + L, 25.5)
            if not (xb > POOLR[0] - 0.1 and xa < POOLR[1] and y < POOLR[3] + 0.1 and y + 0.6 > POOLR[2]):
                finish(s, "pavers", xa, xb, y, y + 0.6, zf, t=0.03)
            xa = xb + rng.uniform(0.12, 0.6)
        y += 0.75
    # reflecting pool (hole in the garden slab: 0.25 m deep, dark stone bottom)
    x0, x1, y0, y1 = POOLR
    solid(s, x0, x1, y0, y1, zf - 0.6, zf - 0.3)
    finish(s, "stone_counter", x0, x1, y0, y1, zf - 0.3)
    for (a, b, c, d) in ((x0 - 0.12, x1 + 0.12, y0 - 0.12, y0), (x0 - 0.12, x1 + 0.12, y1, y1 + 0.12),
                         (x0 - 0.12, x0, y0, y1), (x1, x1 + 0.12, y0, y1)):
        s.box("stone_blocks", ((a + b) / 2, (c + d) / 2, zf - 0.15), ((b - a) / 2, (d - c) / 2, 0.15),
              collide=False)
    s.planes.append(dict(name="reflecting_pool", mat="water_dark", pos=((x0 + x1) / 2, (y0 + y1) / 2, zf - 0.05),
                         half=((x1 - x0) / 2, (y1 - y0) / 2)))
    s.note("upper garden", "reflecting pool 3.4 x 5.9 m, 25 cm deep")
    # raised stone planter with a frangipani, beds along the north and east edges
    planter(s, 23.7, 25.5, 10.45, 14.5, zf, 0.5, wall_mat="stone_blocks", plants="fern", seed=31, t=0.15)
    V.frangipani(s, (24.6, 12.5, zf + 0.45), 4.2, seed=3)
    V.frangipani(s, (19.4, 15.2, zf), 3.6, seed=8)
    for k, x in enumerate(np.arange(18.2, 25.4, 1.1)):
        V.ginger_clump(s, (x, 16.1, zf), 0.6, n=6, height=1.4, seed=60 + k)
    for k, y in enumerate(np.arange(10.8, 16.0, 1.4)):
        V.areca_clump(s, (25.35, y, zf), 3.2, stems=5, seed=70 + k)
    s.note("upper garden", "concrete pavers in white gravel, frangipani, ginger, areca palms")


def east_terrace_and_lift(s):
    """Terrace at the foot of the hill and the glass inclined lift (photo 6) to the parking."""
    zf = Z_UP
    pit = (26.25, 28.5, 16.15, 17.85)
    for (a, b, c, d) in rect_minus((25.7, 28.5, -1.7, 19.5), [pit]):
        solid(s, a, b, c, d, Z_SOC, zf, friction=s.materials["pavers"].friction)
        finish(s, "pavers", a, b, c, d, zf, uv_rot=math.pi / 2)
    wall(s, "y", 28.65, -1.7, 19.5, zf - 0.5, zf + 1.6, 0.3, "stone_wall", openings=[(16.1, 17.9, zf - 0.5, zf + 1.6)])
    # lift pit and rails
    th = LIFT_THETA
    u = np.array([math.cos(th), 0, math.sin(th)])
    base = np.array([LIFT_X0, LIFT_Y, zf])
    solid(s, pit[0], pit[1], pit[2], pit[3], zf - 1.9, zf - 1.6)
    finish(s, "concrete_light", pit[0], pit[1], pit[2], pit[3], zf - 1.6)
    for yy in (LIFT_Y - 0.55, LIFT_Y + 0.55):
        p0 = np.array([pit[0], yy, zf - 1.55])
        p1 = p0 + u * (LIFT_TRAVEL + 3.0)
        s.box("stainless", ((p0 + p1) / 2).tolist(), (np.linalg.norm(p1 - p0) / 2, 0.04, 0.06), R=rot_y(-th),
              collide=False)
        for k in range(0, int(LIFT_TRAVEL) + 3, 2):
            q = p0 + u * k
            s.box("concrete_light", (q[0], q[1], (q[2] - 0.06 + q[2] - 3.0) / 2), (0.15, 0.15, 1.5), collide=False)
    # glass barriers around the pit (open on the west side where the cabin is boarded)
    glass_rail(s, "x", pit[2] - 0.05, pit[0], pit[1], zf)
    glass_rail(s, "x", pit[3] + 0.05, pit[0], pit[1], zf)
    # cabin: slide joint along the rail, gravity compensated; position servo with velocity limit; two gates
    cab = s.body("lift_cabin", Frame(base), joint=dict(type="slide", axis=tuple(u), range=(0, LIFT_TRAVEL),
                                                       damping=400.0, armature=50.0), mass=450.0,
                 inertia=(300.0, 300.0, 300.0))
    cab.gravcomp = 1.0
    cab.com = (0, 0, 1.0)
    L, W, H = 1.9, 1.6, 2.4
    with s.in_body(cab):
        s.box("granite_tiles", (0, 0, -0.04), (L / 2, W / 2, 0.04), collide=True)
        # chassis wedge under the floor following the rail
        wedge = np.array([[-L / 2, -W / 2 + 0.1, -0.08], [L / 2, -W / 2 + 0.1, -0.08], [-L / 2, W / 2 - 0.1, -0.08],
                          [L / 2, W / 2 - 0.1, -0.08], [-L / 2, -W / 2 + 0.1, -0.08 - L * math.tan(th) - 0.3],
                          [-L / 2, W / 2 - 0.1, -0.08 - L * math.tan(th) - 0.3],
                          [L / 2, -W / 2 + 0.1, -0.38], [L / 2, W / 2 - 0.1, -0.38]])
        Fw = [(0, 2, 1), (1, 2, 3), (0, 1, 4), (1, 6, 4), (2, 5, 3), (3, 5, 7), (0, 4, 2), (2, 4, 5), (1, 3, 6),
              (3, 7, 6), (4, 6, 5), (5, 6, 7)]
        s.trimesh("steel_dark", wedge, Fw, smooth=False)
        for sy in (-1, 1):
            s.box("glass", (0, sy * (W / 2 - 0.01), H / 2), (L / 2, 0.01, H / 2), collide=False)
            s.collider("box", (0, sy * (W / 2 - 0.02), H / 2), (L / 2, 0.03, H / 2))
            for sx in (-1, 1):
                s.tube("stainless", (sx * (L / 2 - 0.03), sy * (W / 2 - 0.03), 0),
                       (sx * (L / 2 - 0.03), sy * (W / 2 - 0.03), H), 0.022)
            s.tube("stainless", (-L / 2, sy * (W / 2 - 0.03), H), (L / 2, sy * (W / 2 - 0.03), H), 0.022)
            s.tube("stainless", (-L / 2, sy * (W / 2 - 0.03), 1.0), (L / 2, sy * (W / 2 - 0.03), 1.0), 0.016)
        s.box("glass", (0, 0, H), (L / 2, W / 2, 0.01), collide=False)
        s.box("stainless", (L / 2 - 0.05, W / 2 - 0.05, 1.2), (0.03, 0.03, 0.1), collide=False)   # buttons
        for sx in (-1, 1):
            g = s.body(f"lift_gate_{'up' if sx > 0 else 'down'}", Frame((sx * (L / 2 - 0.04), 0, 0)),
                       joint=dict(type="slide", axis=(0, 1, 0), range=(0, W - 0.15), damping=60.0, armature=2.0,
                                  qpos0=(W - 0.15) if sx < 0 else 0.0), mass=12.0, inertia=(1.0, 1.0, 1.0),
                       parent=cab)
            g.com = (0, 0, 0.55)
            with s.in_body(g):
                s.box("glass", (0, 0, 0.55), (0.012, W / 2 - 0.1, 0.5), collide=False)
                s.box("stainless", (0, 0, 1.06), (0.02, W / 2 - 0.1, 0.02), collide=False)
                s.collider("box", (0, 0, 0.55), (0.03, W / 2 - 0.1, 0.55))
    s.actuators.append(f'<intvelocity name="lift" joint="lift_cabin_j" kp="60000" kv="6000" actrange="0 '
                       f'{LIFT_TRAVEL:.4f}" ctrlrange="-0.6 0.6"/>')
    s.actuators.append('<position name="lift_gate_up" joint="lift_gate_up_j" kp="2000" kv="200" ctrlrange="0 1.45"/>')
    s.actuators.append('<position name="lift_gate_down" joint="lift_gate_down_j" kp="2000" kv="200" '
                       'ctrlrange="0 1.45"/>')
    s.note("inclined lift", f"glass cabin on a {math.degrees(th):.0f} deg incline, {LIFT_TRAVEL:.1f} m travel, "
           "automatic gates (house/house_runtime.py)")
    s.spawns["terrace"] = (25.0, 15.0, zf, 0.0)
    s.spawns["lift_entry"] = (24.9, LIFT_Y, zf, 0.0)


def parking(s):
    """Parking deck at the top of the lift, carport, driveway up to the street."""
    zp = Z_PARK
    top_x = LIFT_X0 + LIFT_TRAVEL * math.cos(LIFT_THETA)
    xs = top_x + 0.95
    slab(s, xs, 48.0, 1.0, 19.6, zp, 0.4, top="concrete_light", side="stone_wall")
    slab(s, top_x - 1.3, xs, LIFT_Y + 0.85, 19.6, zp, 0.4, top="concrete_light", side="stone_wall")
    slab(s, top_x - 1.3, xs, 1.0, LIFT_Y - 0.85, zp, 0.4, top="concrete_light", side="stone_wall")
    solid(s, xs, 48.0, 1.0, 19.6, zp - 8.0, zp - 0.4)
    glass_rail(s, "y", top_x - 1.3, 1.0, LIFT_Y - 0.85, zp)
    glass_rail(s, "y", top_x - 1.3, LIFT_Y + 0.85, 19.6, zp)
    glass_rail(s, "x", LIFT_Y - 0.9, top_x - 1.3, xs, zp)
    glass_rail(s, "x", LIFT_Y + 0.9, top_x - 1.3, xs, zp)
    # carport
    for x in (xs + 0.6, 47.6):
        for y in (8.2, 13.6, 19.2):
            column(s, x, y, zp, zp + 2.7, 0.08)
    s.box("roof_metal", ((xs + 0.6 + 47.8) / 2, 13.7, zp + 2.78), ((47.8 - xs - 0.6) / 2 + 0.3, 5.8, 0.08),
          collide=True)
    s.box("wood_ceiling", ((xs + 0.6 + 47.8) / 2, 13.7, zp + 2.69), ((47.8 - xs - 0.6) / 2 + 0.3, 5.8, 0.01),
          collide=False)
    FU.car(s, Frame((44.0, 13.6, zp), yaw=0.0))
    FU.generator(s, Frame((42.0, 2.4, zp), yaw=math.pi / 2))
    # driveway ramp up to the street (20 %), street with kerbs
    x0r, x1r = 48.0, 62.0
    L = x1r - x0r
    ang = math.atan2(Z_STREET - zp, L)
    c = ((x0r + x1r) / 2, 14.5, (zp + Z_STREET) / 2 - 0.2)
    s.box("concrete_light", c, (math.hypot(L, Z_STREET - zp) / 2 + 0.05, 2.5, 0.2), R=rot_y(-ang), collide=True)
    for yy in (11.85, 17.15):
        wall_seg(s, (x0r, yy), (x1r, yy), zp - 1.0, Z_STREET + 0.4, 0.3, "stone_wall")
    slab(s, 62.0, 70.0, -40.0, 60.0, Z_STREET, 0.5, top="asphalt", side="curb")
    slab(s, 60.5, 62.0, -40.0, 11.7, Z_STREET + 0.15, 0.65, top="concrete_light", side="curb")
    slab(s, 60.5, 62.0, 17.3, 60.0, Z_STREET + 0.15, 0.65, top="concrete_light", side="curb")
    for y in np.arange(-38.0, 58.0, 6.0):
        s.box("white_gloss", (66.0, y, Z_STREET + 0.001), (0.06, 1.5, 0.001), collide=False)
    s.note("parking", "parking deck with carport, car, standby generator; 20 % driveway to the street")
    s.spawns["parking"] = (43.0, 6.0, zp, math.pi)


def boundaries(s):
    """Rubble stone boundary walls (photos 5, 7, 11): south straight, north slanted, stepped with the levels."""
    zones = [(-17.5, -0.5, Z_DECK, 1.6), (-0.5, 18.0, Z_SOC - 0.5, 1.8), (18.0, 28.5, Z_SOC, Z_UP + 1.4)]
    for (x0, x1, zb, zt) in zones:
        wall(s, "x", -1.85, x0, x1, zb, zt, 0.3, "stone_wall")
    pts = [(-13.4, 0.0, 1.0), (-0.5, 0.0, 1.8), (5.0, Z_SOC, Z_UP + 1.0), (25.7, Z_UP - 0.5, Z_UP + 1.4),
           (28.5, Z_UP, Z_UP + 1.6)]
    for (xa, zb, zt), (xb, _, _) in zip(pts[:-1], pts[1:]):
        wall_seg(s, (xa, y_north(xa) + 0.15), (xb, y_north(xb) + 0.15), min(zb, Z_DECK) if xa < -0.4 else zb, zt, 0.3,
                 "stone_wall")
    wall_seg(s, (-24.0, -1.85), (-17.5, -1.85), Z_DECK - 0.4, Z_DECK + 1.2, 0.3, "stone_wall")
    wall_seg(s, (-27.0, 17.35), (-13.4, 17.35), Z_DECK - 0.4, Z_DECK + 1.2, 0.3, "stone_wall")


def build(s):
    sea_deck(s)
    pier(s)
    pavilion(s)
    plateau(s)
    pool(s)
    social_house(s)
    living_stair(s)
    upper_level(s)
    roof(s)
    north_stairs(s)
    upper_garden(s)
    east_terrace_and_lift(s)
    parking(s)
    boundaries(s)
    s.spawns["sea_deck"] = (-22.5, 4.5, Z_DECK, math.pi / 2)
    s.spawns["pool_deck"] = (-8.6, 12.0, 0.0, -math.pi / 2)
