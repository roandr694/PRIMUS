"""Parametric furniture, fixtures and appliances, modelled after the pieces seen in the CMA house photographs.

Convention: every builder takes the scene `s` and a `Frame` `fr` whose origin is the centre of the footprint at
floor level; local +Y is the front of the piece (where you sit / stand), local +X its right-hand side seen from
behind. Visual detail is fine-grained; collisions use a few boxes that follow the solid volumes. Moving parts
(drawers, cabinet and appliance doors) are separate bodies with slide/hinge joints.
"""
import math

import numpy as np

from geom import Frame, rot_x, rot_y, rot_z

_counter = {"n": 0}


def uid(prefix):
    _counter["n"] += 1
    return f"{prefix}{_counter['n']}"


# ================================================================================================================
# seating
# ================================================================================================================
def sofa_sectional(s, fr, L=3.0, D=1.05, ret=0.0, fabric="fabric_white", H=0.42, back_h=0.78, note=None):
    """Deep modern sofa (photos 9, 12): low plinth, thick seat cushions, loose back cushions. ret > 0 adds a
    chaise/return of that length on the right-hand side."""
    plinth = "black_matte"
    s.box(plinth, fr.pt((0, 0, 0.04)), (L / 2 - 0.05, D / 2 - 0.05, 0.04), R=fr.R, collide=False)
    s.rbox(fabric, (0, 0, 0.18), (L / 2, D / 2, 0.11), 0.05, frame=fr)              # base upholstery
    n = max(2, int(round(L / 0.95)))
    w = L / n
    for i in range(n):
        x = -L / 2 + w * (i + 0.5)
        s.rbox(fabric, (x, 0.06, H - 0.05), (w / 2 - 0.01, D / 2 - 0.13, 0.07), 0.05, frame=fr)
        s.rbox(fabric, (x, -D / 2 + 0.14, back_h - 0.17), (w / 2 - 0.02, 0.11, 0.2), 0.08, frame=fr)
    for sx in (-1, 1):
        if ret and sx == 1:
            continue
        s.rbox(fabric, (sx * (L / 2 - 0.1), 0, 0.42), (0.1, D / 2, 0.16), 0.05, frame=fr)
    s.collider("box", (0, 0, H / 2), (L / 2, D / 2, H / 2), frame=fr, friction=(0.9, 0.02, 0.002))
    s.collider("box", (0, -D / 2 + 0.13, back_h / 2), (L / 2, 0.13, back_h / 2), frame=fr)
    if ret:
        rf = fr.child((L / 2 - D / 2, D / 2 + ret / 2 - 0.0, 0))
        s.rbox(fabric, (0, 0, 0.18), (D / 2, ret / 2, 0.11), 0.05, frame=rf)
        s.rbox(fabric, (0, 0, H - 0.05), (D / 2 - 0.03, ret / 2 - 0.03, 0.07), 0.05, frame=rf)
        s.rbox(fabric, (D / 2 - 0.1, 0, 0.42), (0.1, ret / 2, 0.16), 0.05, frame=rf)
        s.collider("box", (0, 0, H / 2), (D / 2, ret / 2, H / 2), frame=rf)
    for k, c in enumerate(("fabric_yellow", "fabric_teal", "fabric_beige")):
        x = -L / 2 + 0.45 + k * 0.55
        s.rbox(c, (x, -D / 2 + 0.3, H + 0.2), (0.22, 0.07, 0.2), 0.06, frame=fr, R=rot_x(-0.25))
    if note:
        s.note(note, f"sofa {L:.1f} m")


def sofa_builtin(s, fr, L=4.0, D=0.95, note=None):
    """Long built-in varanda sofa (photo 8): solid wood frame, white seat and back cushions, scatter pillows."""
    wood = "wood_light"
    s.box(wood, fr.pt((0, 0, 0.17)), (L / 2, D / 2, 0.17), R=fr.R, collide=False)
    s.box(wood, fr.pt((0, -D / 2 + 0.05, 0.38)), (L / 2, 0.05, 0.38), R=fr.R, collide=False)
    s.rbox("fabric_white", (0, 0.04, 0.42), (L / 2 - 0.02, D / 2 - 0.08, 0.08), 0.05, frame=fr)
    nb = int(L / 0.9)
    for i in range(nb):
        x = -L / 2 + L / nb * (i + 0.5)
        s.rbox("fabric_white", (x, -D / 2 + 0.2, 0.68), (L / nb / 2 - 0.02, 0.09, 0.2), 0.08, frame=fr,
               R=rot_x(-0.2))
    pillows = ["fabric_teal", "fabric_yellow", "fabric_white", "fabric_beige", "fabric_teal"]
    for k in range(min(len(pillows), nb + 1)):
        x = -L / 2 + 0.35 + k * (L - 0.7) / max(1, nb)
        s.rbox(pillows[k], (x, -D / 2 + 0.36, 0.72), (0.24, 0.07, 0.22), 0.06, frame=fr, R=rot_z(0.15 * (k % 2)) @ rot_x(-0.3))
    s.collider("box", (0, 0, 0.25), (L / 2, D / 2, 0.25), frame=fr)
    s.collider("box", (0, -D / 2 + 0.08, 0.45), (L / 2, 0.08, 0.45), frame=fr)
    if note:
        s.note(note, f"built-in sofa {L:.1f} m")


def armchair_woven(s, fr, note=None):
    """Rope-woven tub armchair (photo 8): a curved shell open at the front, seat drum and cushion."""
    ro, ri = 0.42, 0.33
    a0, a1 = math.radians(90 + 55), math.radians(90 - 55 + 360)
    ang = np.linspace(a0, a1, 28)
    poly = [(ro * math.cos(a), ro * math.sin(a)) for a in ang] + [(ri * math.cos(a), ri * math.sin(a))
                                                                   for a in ang[::-1]]
    s.prism("rope_weave", poly, 0.05, 0.72, frame=fr)
    s.cylinder("rope_weave", fr.pt((0, 0, 0.2)), 0.4, 0.17, R=fr.R, collide=False, seg=28)
    s.cylinder("fabric_white", fr.pt((0, 0.02, 0.43)), 0.34, 0.06, R=fr.R, collide=False, seg=28)
    s.collider("box", (0, 0, 0.36), (0.42, 0.4, 0.36), frame=fr)
    if note:
        s.note(note, "woven armchair")


def armchair_lounge(s, fr, fabric="fabric_sage", note=None):
    """Mid-century lounge armchair (photo 14): low wooden frame, plump seat and back cushions."""
    wood = "wood_dark"
    for sx in (-1, 1):
        s.box(wood, fr.pt((sx * 0.38, 0.0, 0.22)), (0.035, 0.42, 0.035), R=fr.R, collide=False)
        for sy in (-1, 1):
            s.box(wood, fr.pt((sx * 0.38, sy * 0.36, 0.1)), (0.03, 0.03, 0.1), R=fr.R, collide=False)
    s.rbox(fabric, (0, 0.05, 0.3), (0.34, 0.36, 0.09), 0.07, frame=fr)
    s.rbox(fabric, (0, -0.28, 0.58), (0.34, 0.12, 0.24), 0.09, frame=fr, R=rot_x(-0.35))
    for sx in (-1, 1):
        s.rbox(fabric, (sx * 0.38, 0.0, 0.4), (0.07, 0.3, 0.06), 0.05, frame=fr)
    s.collider("box", (0, 0, 0.3), (0.42, 0.44, 0.3), frame=fr)
    if note:
        s.note(note, "lounge armchair")


def deck_lounge_chair(s, fr, note=None):
    """Chunky teak lounge chair with a grey cushion (photo 13)."""
    wood = "deck_grey"
    for sx in (-1, 1):
        s.box(wood, fr.pt((sx * 0.37, 0, 0.27)), (0.06, 0.5, 0.27), R=fr.R, collide=False)
    s.box(wood, fr.pt((0, 0.05, 0.22)), (0.31, 0.42, 0.04), R=fr.R, collide=False)
    s.box(wood, fr.pt((0, -0.38, 0.5)), (0.31, 0.06, 0.3), R=fr.R @ rot_x(-0.35), collide=False)
    s.rbox("fabric_grey", (0, 0.07, 0.31), (0.3, 0.4, 0.07), 0.05, frame=fr)
    s.rbox("fabric_grey", (0, -0.3, 0.58), (0.3, 0.08, 0.26), 0.06, frame=fr, R=rot_x(-0.35))
    s.collider("box", (0, 0, 0.3), (0.43, 0.52, 0.3), frame=fr)
    if note:
        s.note(note, "deck lounge chair")


def chair_wood(s, fr, seat="rush", note=None, H=0.46):
    """Wooden dining chair with rush seat and slatted back."""
    wood = "wood_light"
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.box(wood, fr.pt((sx * 0.2, sy * 0.2, H / 2)), (0.018, 0.018, H / 2), R=fr.R, collide=False)
        s.box(wood, fr.pt((sx * 0.2, -0.2, H + 0.22)), (0.018, 0.018, 0.22), R=fr.R, collide=False)
    s.box(seat, fr.pt((0, 0, H)), (0.22, 0.22, 0.02), R=fr.R, collide=False)
    s.box(wood, fr.pt((0, -0.2, H + 0.36)), (0.2, 0.012, 0.07), R=fr.R, collide=False)
    s.box(wood, fr.pt((0, 0.0, 0.12)), (0.2, 0.012, 0.012), R=fr.R, collide=False)
    s.collider("box", (0, 0, H / 2), (0.22, 0.22, H / 2), frame=fr)
    s.collider("box", (0, -0.2, H + 0.22), (0.22, 0.03, 0.22), frame=fr)


def chair_windsor(s, fr):
    """Spindle-back wooden chair (photo 12)."""
    wood = "wood_dark"
    s.rbox(wood, (0, 0, 0.45), (0.24, 0.22, 0.02), 0.015, frame=fr)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.tube(wood, (sx * 0.17, sy * 0.15, 0.44), (sx * 0.22, sy * 0.2, 0.0), 0.016, frame=fr)
    for k in range(7):
        x = -0.18 + k * 0.06
        s.tube(wood, (x, -0.18, 0.46), (x * 1.1, -0.24, 0.95), 0.008, frame=fr)
    s.tube(wood, (-0.22, -0.24, 0.95), (0.22, -0.24, 0.95), 0.014, frame=fr)
    s.collider("box", (0, 0, 0.24), (0.24, 0.22, 0.24), frame=fr)


def bar_stool(s, fr, H=0.66):
    """Wooden stool with a woven rush seat (photo 10)."""
    wood = "wood_light"
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.box(wood, fr.pt((sx * 0.17, sy * 0.15, H / 2)), (0.022, 0.022, H / 2), R=fr.R, collide=False)
    for z in (0.18, 0.42):
        for sy in (-1, 1):
            s.box(wood, fr.pt((0, sy * 0.15, z)), (0.17, 0.015, 0.015), R=fr.R, collide=False)
        s.box(wood, fr.pt((0, 0, z + 0.06)), (0.015, 0.15, 0.015), R=fr.R, collide=False)
    s.rbox("rush", (0, 0, H + 0.02), (0.2, 0.17, 0.03), 0.02, frame=fr)
    s.collider("box", (0, 0, H / 2), (0.2, 0.18, H / 2), frame=fr)


def stool_ceramic(s, fr):
    """Glazed green ceramic box stool (photo 8)."""
    s.rbox("ceramic_green", (0, 0, 0.23), (0.17, 0.17, 0.23), 0.02, frame=fr, collide=True)
    s.box("black_matte", fr.pt((0, 0.171, 0.36)), (0.06, 0.002, 0.02), R=fr.R, collide=False)


def lounger(s, fr, mattress="fabric_turquoise", pillow="fabric_yellow"):
    """Pool lounger (photo 5): low teak frame, turquoise mattress, yellow pillow."""
    s.box("deck_grey", fr.pt((0, 0, 0.12)), (0.36, 0.98, 0.04), R=fr.R, collide=False)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.box("deck_grey", fr.pt((sx * 0.3, sy * 0.88, 0.05)), (0.04, 0.04, 0.05), R=fr.R, collide=False)
    s.rbox(mattress, (0, 0.1, 0.22), (0.34, 0.86, 0.06), 0.05, frame=fr)
    s.rbox(mattress, (0, -0.78, 0.36), (0.34, 0.2, 0.06), 0.05, frame=fr, R=rot_x(0.5))
    s.rbox(pillow, (0, -0.72, 0.5), (0.22, 0.08, 0.06), 0.04, frame=fr, R=rot_x(0.5))
    s.collider("box", (0, 0, 0.2), (0.37, 1.0, 0.2), frame=fr)


def parasol(s, fr, r=1.3, colour="fabric_turquoise", h=2.4):
    """Round woven parasol (photo 5)."""
    s.cylinder("concrete_light", fr.pt((0, 0, 0.04)), 0.25, 0.04, R=fr.R, collide=True)
    s.cylinder("stainless", fr.pt((0, 0, h / 2)), 0.022, h / 2, R=fr.R, collide=True)
    s.cylinder(colour, fr.pt((0, 0, h - 0.05)), r, 0.12, R=fr.R, r_top=0.12, collide=False, seg=32)
    s.cylinder(colour, fr.pt((0, 0, h - 0.18)), r, 0.008, R=fr.R, collide=False, seg=32)


def hammock(s, a, b, sag=0.45, width=0.9, mat="fabric_white"):
    """Fabric hammock hung between anchor points a and b (visual only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d[:2])
    side = np.array([-d[1], d[0], 0]) / (L + 1e-9)
    n = 18
    V, Fc = [], []
    for i in range(n + 1):
        t = i / n
        p = a + d * t
        p[2] -= sag * 4 * t * (1 - t)
        w = width / 2 * (math.sin(math.pi * t) ** 0.7)
        for sgn in (-1, 1):
            q = p + side * w * sgn
            q[2] += 0.06 * (1 - abs(sgn)) + 0.05 * math.sin(math.pi * t) * (abs(sgn))
            V.append(q)
    for i in range(n):
        k = 2 * i
        Fc += [(k, k + 2, k + 3), (k, k + 3, k + 1), (k, k + 3, k + 2), (k, k + 1, k + 3)]
    s.trimesh(mat, np.array(V), np.array(Fc), smooth=True)
    s.tube("rope", a, a + d * 0.12 - np.array([0, 0, 0.05]), 0.01)
    s.tube("rope", b, b - d * 0.12 - np.array([0, 0, 0.05]), 0.01)


# ================================================================================================================
# tables
# ================================================================================================================
def coffee_table_log(s, fr, L=1.2, W=0.55, H=0.38):
    """Solid timber block coffee table (photo 8: two of them side by side)."""
    s.rbox("wood_light", (0, 0, H / 2), (L / 2, W / 2, H / 2), 0.03, frame=fr, collide=True)


def coffee_table_round(s, fr, r=0.65, H=0.32):
    s.cylinder("wood_light", fr.pt((0, 0, H - 0.04)), r, 0.04, R=fr.R, collide=False, seg=40)
    s.cylinder("wood_dark", fr.pt((0, 0, (H - 0.08) / 2)), r * 0.55, (H - 0.08) / 2, R=fr.R, collide=False,
               seg=32)
    s.collider("cylinder", (0, 0, H / 2), (r, H / 2), frame=fr)
    s.cylinder("ceramic_white", fr.pt((0.2, 0.1, H + 0.03)), 0.14, 0.03, R=fr.R, collide=False)
    plant_small(s, fr.child((-0.25, -0.15, H)), 0.2)


def side_table_glass(s, fr, L=1.2, W=0.45, H=0.42):
    """Black steel frame, glass top (photo 9)."""
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.box("steel_dark", fr.pt((sx * (L / 2 - 0.01), sy * (W / 2 - 0.01), H / 2)), (0.008, 0.008, H / 2),
                  R=fr.R, collide=False)
        s.box("steel_dark", fr.pt((sx * (L / 2 - 0.01), 0, H - 0.01)), (0.008, W / 2, 0.008), R=fr.R, collide=False)
        s.box("steel_dark", fr.pt((sx * (L / 2 - 0.01), 0, 0.01)), (0.008, W / 2, 0.008), R=fr.R, collide=False)
    s.box("glass", fr.pt((0, 0, H)), (L / 2, W / 2, 0.006), R=fr.R, collide=False)
    s.box("glass", fr.pt((0, 0, 0.12)), (L / 2, W / 2, 0.006), R=fr.R, collide=False)
    s.collider("box", (0, 0, H / 2), (L / 2, W / 2, H / 2), frame=fr)
    s.cylinder("rattan", fr.pt((0.3, 0, 0.18)), 0.16, 0.05, R=fr.R, collide=False)
    s.box("book_cream", fr.pt((-0.3, 0, H + 0.03)), (0.15, 0.11, 0.025), R=fr.R, collide=False)


def side_table_round(s, fr, r=0.22, H=0.5, mat="wood_dark"):
    s.cylinder(mat, fr.pt((0, 0, H - 0.015)), r, 0.015, R=fr.R, collide=False)
    s.cylinder(mat, fr.pt((0, 0, (H - 0.03) / 2)), 0.03, (H - 0.03) / 2, R=fr.R, collide=False)
    s.cylinder(mat, fr.pt((0, 0, 0.01)), r * 0.7, 0.01, R=fr.R, collide=False)
    s.collider("cylinder", (0, 0, H / 2), (r, H / 2), frame=fr)


def dining_table(s, fr, L=3.2, W=1.05, H=0.75, mat="wood_light"):
    s.box(mat, fr.pt((0, 0, H - 0.03)), (L / 2, W / 2, 0.03), R=fr.R, collide=False)
    for sx in (-1, 1):
        s.box(mat, fr.pt((sx * (L / 2 - 0.35), 0, (H - 0.06) / 2)), (0.05, W / 2 - 0.12, (H - 0.06) / 2),
              R=fr.R, collide=False)
    s.collider("box", (0, 0, H - 0.03), (L / 2, W / 2, 0.03), frame=fr)
    for sx in (-1, 1):
        s.collider("box", (sx * (L / 2 - 0.35), 0, (H - 0.06) / 2), (0.05, W / 2 - 0.12, (H - 0.06) / 2), frame=fr)
    s.cylinder("ceramic_white", fr.pt((0, 0, H + 0.06)), 0.18, 0.06, R=fr.R, r_top=0.22, collide=False)


def table_with_chairs(s, fr, L, W, n_side, chair=chair_wood, H=0.75, mat="wood_light", ends=True):
    dining_table(s, fr, L, W, H, mat)
    for i in range(n_side):
        x = -L / 2 + L / n_side * (i + 0.5)
        chair(s, fr.child((x, W / 2 + 0.22, 0), yaw=math.pi))
        chair(s, fr.child((x, -W / 2 - 0.22, 0)))
    if ends:
        chair(s, fr.child((L / 2 + 0.25, 0, 0), yaw=math.pi / 2))
        chair(s, fr.child((-L / 2 - 0.25, 0, 0), yaw=-math.pi / 2))


def desk(s, fr, L=1.6, W=0.7):
    s.box("wood_light", fr.pt((0, 0, 0.73)), (L / 2, W / 2, 0.025), R=fr.R, collide=False)
    for sx in (-1, 1):
        s.box("steel_dark", fr.pt((sx * (L / 2 - 0.05), 0, 0.36)), (0.02, W / 2 - 0.04, 0.36), R=fr.R,
              collide=False)
    s.collider("box", (0, 0, 0.73), (L / 2, W / 2, 0.025), frame=fr)
    for sx in (-1, 1):
        s.collider("box", (sx * (L / 2 - 0.05), 0, 0.36), (0.02, W / 2 - 0.04, 0.36), frame=fr)
    s.box("black_glass", fr.pt((0.1, -0.15, 0.95)), (0.3, 0.01, 0.19), R=fr.R, collide=False)
    s.box("steel_dark", fr.pt((0.1, -0.15, 0.8)), (0.04, 0.04, 0.05), R=fr.R, collide=False)
    s.box("stainless", fr.pt((0.1, 0.05, 0.76)), (0.2, 0.07, 0.006), R=fr.R, collide=False)
    table_lamp(s, fr.child((-L / 2 + 0.2, -0.15, 0.755)))


def office_chair(s, fr):
    s.cylinder("black_matte", fr.pt((0, 0, 0.03)), 0.3, 0.02, R=fr.R, collide=False)
    s.cylinder("stainless", fr.pt((0, 0, 0.25)), 0.025, 0.2, R=fr.R, collide=False)
    s.rbox("leather_tan", (0, 0, 0.48), (0.25, 0.24, 0.05), 0.04, frame=fr)
    s.rbox("leather_tan", (0, -0.24, 0.78), (0.23, 0.04, 0.25), 0.04, frame=fr)
    s.collider("box", (0, 0, 0.3), (0.3, 0.3, 0.3), frame=fr)


# ================================================================================================================
# storage, shelving, decor
# ================================================================================================================
def books(s, fr, L, depth=0.2, rng=None, H=0.25):
    rng = rng or np.random.default_rng(len(s.colliders))
    x = -L / 2
    mats = ["book_red", "book_blue", "book_cream", "book_green", "book_cream", "book_cream"]
    while x < L / 2 - 0.05:
        w = rng.uniform(0.02, 0.05)
        h = rng.uniform(0.6, 1.0) * H
        s.box(mats[rng.integers(len(mats))], fr.pt((x + w / 2, 0, h / 2)), (w / 2 - 0.001, depth / 2 * rng.uniform(0.75, 1), h / 2),
              R=fr.R, collide=False)
        x += w
        if rng.random() < 0.08:
            x += rng.uniform(0.1, 0.3)


def vase(s, fr, h=0.3, r=0.08, mat="ceramic_white"):
    s.cylinder(mat, fr.pt((0, 0, h / 2)), r, h / 2, R=fr.R, r_top=r * 0.6, collide=False, seg=20)


def frame_picture(s, fr, w=0.4, h=0.5, art="art_1", lean=True):
    R = fr.R @ (rot_x(-0.12) if lean else np.eye(3))
    s.box("wood_dark", fr.pt((0, 0, h / 2)), (w / 2, 0.012, h / 2), R=R, collide=False)
    s.box("book_cream", fr.pt((0, 0.013, h / 2)), (w / 2 - 0.04, 0.002, h / 2 - 0.04), R=R, collide=False)
    s.box(art, fr.pt((0, 0.016, h / 2)), (w / 2 - 0.08, 0.002, h / 2 - 0.08), R=R, collide=False)


def shelf_wall(s, fr, L=3.0, d=0.28, t=0.05, decor=True):
    """Floating wooden shelf with objects (TV wall, photos 9 and 12)."""
    s.box("wood_light", fr.pt((0, 0, 0)), (L / 2, d / 2, t / 2), R=fr.R, collide=False)
    if not decor:
        return
    rng = np.random.default_rng(int(abs(fr.t[0] * 100 + fr.t[1] * 10)))
    x = -L / 2 + 0.2
    while x < L / 2 - 0.3:
        k = rng.integers(5)
        f2 = fr.child((x, 0, t / 2))
        if k == 0:
            books(s, f2, 0.35, 0.18, rng)
            x += 0.45
        elif k == 1:
            vase(s, f2, rng.uniform(0.15, 0.35), rng.uniform(0.04, 0.08),
                 ["ceramic_white", "ceramic_green", "wood_dark"][rng.integers(3)])
            x += 0.3
        elif k == 2:
            frame_picture(s, f2.child((0, -0.05, 0)), 0.3, 0.38, ["art_1", "art_2"][rng.integers(2)])
            x += 0.45
        elif k == 3:
            plant_small(s, f2, 0.25)
            x += 0.35
        else:
            s.cylinder("rattan", f2.pt((0, 0, 0.06)), 0.12, 0.06, R=f2.R, collide=False)
            x += 0.35
        x += rng.uniform(0.1, 0.4)


def bench_shelf(s, fr, L=5.0, d=0.45, H=0.34):
    """Low wooden bench running along the TV wall (photo 12)."""
    s.box("wood_light", fr.pt((0, 0, H - 0.025)), (L / 2, d / 2, 0.025), R=fr.R, collide=False)
    s.box("wood_light", fr.pt((0, -d / 2 + 0.03, (H - 0.05) / 2)), (L / 2 - 0.05, 0.03, (H - 0.05) / 2), R=fr.R,
          collide=False)
    s.collider("box", (0, 0, H / 2), (L / 2, d / 2, H / 2), frame=fr)
    s.box("wood_dark", fr.pt((-L / 2 + 0.6, 0, H + 0.08)), (0.12, 0.09, 0.08), R=fr.R, collide=False)
    s.cylinder("wood_dark", fr.pt((-L / 2 + 1.2, 0, H + 0.04)), 0.16, 0.04, R=fr.R, r_top=0.2, collide=False)
    frame_picture(s, fr.child((L / 2 - 0.8, -0.1, H)), 0.45, 0.55, "art_2")
    frame_picture(s, fr.child((L / 2 - 1.2, -0.12, H)), 0.4, 0.5, "art_1")
    vase(s, fr.child((L / 2 - 1.8, 0, H)), 0.3, 0.09, "ceramic_green")


def tv(s, fr, w=1.68, h=0.96):
    """Wall-mounted TV with soundbar. fr origin = centre of the screen on the wall surface."""
    s.box("black_glass", fr.pt((0, 0.025, 0)), (w / 2, 0.025, h / 2), R=fr.R, collide=False)
    s.box("screen_on", fr.pt((0, 0.051, 0)), (w / 2 - 0.01, 0.001, h / 2 - 0.01), R=fr.R, collide=False)
    s.box("black_matte", fr.pt((0, 0.05, -h / 2 - 0.08)), (w * 0.4, 0.05, 0.04), R=fr.R, collide=False)


def bookcase(s, fr, W=1.6, H=2.2, D=0.36, rows=5):
    mat = "wood_light"
    for sx in (-1, 1):
        s.box(mat, fr.pt((sx * (W / 2 - 0.015), 0, H / 2)), (0.015, D / 2, H / 2), R=fr.R, collide=False)
    s.box(mat, fr.pt((0, -D / 2 + 0.006, H / 2)), (W / 2, 0.006, H / 2), R=fr.R, collide=False)
    rng = np.random.default_rng(int(abs(fr.t[0] * 31 + fr.t[1] * 7)))
    for r in range(rows + 1):
        z = 0.05 + r * (H - 0.08) / rows
        s.box(mat, fr.pt((0, 0, z)), (W / 2, D / 2, 0.015), R=fr.R, collide=False)
        if r < rows:
            books(s, fr.child((0, 0.02, z + 0.015)), W - 0.1, D * 0.7, rng, H=min(0.3, (H - 0.1) / rows - 0.05))
    s.collider("box", (0, 0, H / 2), (W / 2, D / 2, H / 2), frame=fr)


def sideboard(s, fr, L=2.0, D=0.45, H=0.75, mat="wood_light"):
    s.box(mat, fr.pt((0, 0, H / 2 + 0.05)), (L / 2, D / 2, H / 2 - 0.05), R=fr.R, collide=False)
    s.box("black_matte", fr.pt((0, 0, 0.03)), (L / 2 - 0.05, D / 2 - 0.05, 0.03), R=fr.R, collide=False)
    n = int(L / 0.5)
    for i in range(1, n):
        s.box("wood_dark", fr.pt((-L / 2 + L / n * i, D / 2 + 0.001, H / 2 + 0.05)), (0.003, 0.002, H / 2 - 0.08),
              R=fr.R, collide=False)
    s.collider("box", (0, 0, H / 2), (L / 2, D / 2, H / 2), frame=fr)
    vase(s, fr.child((-L / 2 + 0.3, 0, H)), 0.4, 0.1, "ceramic_green")
    table_lamp(s, fr.child((L / 2 - 0.3, 0, H)))
    books(s, fr.child((0, 0, H)), 0.4, 0.22)


def wardrobe(s, fr, W=2.4, H=2.6, D=0.6, mat="wood_panel"):
    s.box(mat, fr.pt((0, 0, H / 2)), (W / 2, D / 2, H / 2), R=fr.R, collide=True)
    n = max(2, int(round(W / 0.6)))
    for i in range(1, n):
        s.box("black_matte", fr.pt((-W / 2 + W / n * i, D / 2 + 0.001, H / 2)), (0.003, 0.002, H / 2 - 0.02),
              R=fr.R, collide=False)
    for i in range(n):
        x = -W / 2 + W / n * (i + 0.5) + (0.2 if i % 2 == 0 else -0.2) * W / n
        s.box("steel_dark", fr.pt((x, D / 2 + 0.012, 1.1)), (0.006, 0.01, 0.25), R=fr.R, collide=False)


def wine_rack(s, fr, W=2.0, H=2.4, D=0.35):
    """Timber wine wall with bottles."""
    s.box("wood_dark", fr.pt((0, -D / 2 + 0.01, H / 2)), (W / 2, 0.01, H / 2), R=fr.R, collide=False)
    nx, nz = int(W / 0.12), int(H / 0.12)
    for i in range(nx + 1):
        s.box("wood_dark", fr.pt((-W / 2 + i * W / nx, 0, H / 2)), (0.008, D / 2, H / 2), R=fr.R, collide=False)
    for k in range(0, nz + 1, 4):
        s.box("wood_dark", fr.pt((0, 0, k * H / nz)), (W / 2, D / 2, 0.008), R=fr.R, collide=False)
    rng = np.random.default_rng(3)
    for i in range(nx):
        for k in range(nz):
            if rng.random() < 0.75:
                c = fr.pt((-W / 2 + (i + 0.5) * W / nx, 0.02, (k + 0.5) * H / nz))
                s.cylinder("wine_bottle", c, 0.036, D / 2 - 0.03, R=fr.R @ rot_x(math.pi / 2), collide=False, seg=10)
    s.collider("box", (0, 0, H / 2), (W / 2, D / 2, H / 2), frame=fr)


def rug(s, fr, L=3.0, W=2.2, mat="rug"):
    s.box(mat, fr.pt((0, 0, 0.006)), (L / 2, W / 2, 0.006), R=fr.R, collide=True, friction=(1.0, 0.03, 0.004))


def basket(s, fr, r=0.2, h=0.35):
    s.cylinder("rattan", fr.pt((0, 0, h / 2)), r, h / 2, R=fr.R, r_top=r * 1.1, collide=True)


# ================================================================================================================
# lighting fixtures (emissive shades; a few real lights are added by the house)
# ================================================================================================================
def table_lamp(s, fr, h=0.5):
    s.cylinder("ceramic_white", fr.pt((0, 0, 0.12)), 0.08, 0.12, R=fr.R, r_top=0.05, collide=False)
    s.cylinder("brass", fr.pt((0, 0, 0.3)), 0.008, 0.08, R=fr.R, collide=False)
    s.cylinder("lamp_paper", fr.pt((0, 0, h - 0.08)), 0.16, 0.1, R=fr.R, r_top=0.13, collide=False, caps=False)


def floor_lamp(s, fr, h=1.6):
    s.cylinder("black_matte", fr.pt((0, 0, 0.015)), 0.16, 0.015, R=fr.R, collide=False)
    s.cylinder("black_matte", fr.pt((0, 0, h / 2)), 0.012, h / 2, R=fr.R, collide=False)
    s.sphere("lamp_paper", fr.pt((0, 0, h + 0.15)), 0.22, collide=False, scale=(1, 1, 0.8))
    s.collider("cylinder", (0, 0, h / 2), (0.16, h / 2), frame=fr)


def pendant_wicker(s, fr, z_ceiling, z_bottom, r=0.28):
    """Bell-shaped wicker pendant (photo 14). fr: point under the lamp at floor level."""
    s.tube("black_matte", (0, 0, z_bottom + 0.55), (0, 0, z_ceiling), 0.004, frame=fr)
    s.cylinder("rattan", fr.pt((0, 0, z_bottom + 0.3)), r, 0.3, R=fr.R, r_top=0.05, collide=False, caps=False,
               seg=28)
    s.cylinder("lamp_glow", fr.pt((0, 0, z_bottom + 0.1)), 0.06, 0.04, R=fr.R, collide=False)


def pendant_linear(s, fr, z_ceiling, z, L=2.0):
    """Linear wooden pendant over the kitchen island (photo 10)."""
    s.box("wood_dark", fr.pt((0, 0, z)), (L / 2, 0.06, 0.035), R=fr.R, collide=False)
    s.box("lamp_glow", fr.pt((0, 0, z - 0.036)), (L / 2 - 0.05, 0.03, 0.002), R=fr.R, collide=False)
    for sx in (-1, 1):
        s.tube("black_matte", (sx * L / 3, 0, z + 0.035), (sx * L / 3, 0, z_ceiling), 0.003, frame=fr)


def track_light(s, a, b, z, n=4):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s.tube("black_matte", (a[0], a[1], z), (b[0], b[1], z), 0.012)
    for i in range(n):
        p = a + (b - a) * (i + 0.5) / n
        s.cylinder("black_matte", (p[0], p[1], z - 0.09), 0.03, 0.06, collide=False)


# ================================================================================================================
# plants in pots (vegetation.py has the larger ones)
# ================================================================================================================
def plant_small(s, fr, h=0.3):
    import vegetation
    vegetation.near_detail(True)
    s.cylinder("ceramic_white", fr.pt((0, 0, h * 0.25)), h * 0.22, h * 0.25, R=fr.R, r_top=h * 0.28,
               collide=False)
    vegetation.leaf_cluster(s, fr.pt((0, 0, h * 0.55)), h * 0.6, n=14, mat="leaf_dark",
                            seed=int(abs(fr.t[0] * 13 + fr.t[1] * 17)))
    vegetation.near_detail(False)


def plant_pot(s, fr, h=1.2, pot="ceramic_green"):
    import vegetation
    s.cylinder(pot, fr.pt((0, 0, 0.22)), 0.22, 0.22, R=fr.R, r_top=0.25, collide=True)
    vegetation.near_detail(True)
    vegetation.leaf_cluster(s, fr.pt((0, 0, 0.45 + h * 0.4)), h * 0.5, n=40, mat="leaf_dark",
                            seed=int(abs(fr.t[0] * 7 + fr.t[1] * 3)), elong=1.8)
    vegetation.near_detail(False)


# ================================================================================================================
# bedroom
# ================================================================================================================
def bed(s, fr, W=1.93, L=2.03, throw="fabric_yellow", note=None):
    """King bed: wooden platform, headboard panel, mattress, duvet, pillows, folded throw. fr at the foot
    centre? No: fr at the centre of the bed footprint, headboard at local -Y."""
    s.box("wood_dark", fr.pt((0, 0, 0.12)), (W / 2 + 0.05, L / 2 + 0.05, 0.12), R=fr.R, collide=False)
    s.box("wood_panel", fr.pt((0, -L / 2 - 0.06, 0.55)), (W / 2 + 0.35, 0.04, 0.55), R=fr.R, collide=True)
    s.rbox("fabric_white", (0, 0.0, 0.36), (W / 2, L / 2, 0.12), 0.05, frame=fr)
    s.rbox("fabric_white", (0, 0.15, 0.49), (W / 2 + 0.02, L / 2 - 0.12, 0.03), 0.03, frame=fr)
    s.rbox(throw, (0, L / 2 - 0.35, 0.53), (W / 2 + 0.03, 0.22, 0.015), 0.012, frame=fr)
    for sx in (-1, 1):
        s.rbox("fabric_white", (sx * W / 4, -L / 2 + 0.25, 0.58), (W / 4 - 0.04, 0.14, 0.08), 0.07, frame=fr,
               R=rot_x(0.35))
        s.rbox("fabric_beige", (sx * W / 5, -L / 2 + 0.42, 0.56), (0.25, 0.07, 0.16), 0.06, frame=fr,
               R=rot_x(0.3))
    s.collider("box", (0, 0, 0.26), (W / 2 + 0.05, L / 2 + 0.05, 0.26), frame=fr, friction=(0.9, 0.02, 0.002))
    if note:
        s.note(note, "king bed")


def nightstand(s, fr):
    s.box("wood_light", fr.pt((0, 0, 0.24)), (0.25, 0.2, 0.24), R=fr.R, collide=True)
    s.box("black_matte", fr.pt((0, 0.201, 0.3)), (0.2, 0.002, 0.004), R=fr.R, collide=False)
    table_lamp(s, fr.child((0.05, -0.03, 0.48)))
    books(s, fr.child((-0.12, 0.05, 0.48)), 0.12, 0.16, H=0.05)


def bench_end(s, fr, L=1.6):
    s.box("wood_light", fr.pt((0, 0, 0.42)), (L / 2, 0.2, 0.03), R=fr.R, collide=False)
    for sx in (-1, 1):
        s.box("wood_light", fr.pt((sx * (L / 2 - 0.05), 0, 0.2)), (0.03, 0.18, 0.2), R=fr.R, collide=False)
    s.collider("box", (0, 0, 0.225), (L / 2, 0.2, 0.225), frame=fr)
    s.rbox("fabric_grey", (0.3, 0, 0.47), (0.25, 0.18, 0.02), 0.015, frame=fr)


def curtain(s, a, b, z0, z1, mat="glass_frosted", folds=0.08):
    """Sheer curtain between floor points a and b (visual only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    n = max(4, int(L / 0.12))
    nrm = np.array([-d[1], d[0]]) / (L + 1e-9)
    V, Fc = [], []
    for i in range(n + 1):
        p = a + d * i / n + nrm * folds * math.sin(i * math.pi / 2) * 0.5
        V += [(p[0], p[1], z0), (p[0], p[1], z1)]
    for i in range(n):
        k = 2 * i
        Fc += [(k, k + 2, k + 3), (k, k + 3, k + 1), (k, k + 3, k + 2), (k, k + 1, k + 3)]
    s.trimesh(mat, np.array(V), np.array(Fc), smooth=True)


# ================================================================================================================
# bathroom
# ================================================================================================================
def vanity(s, fr, L=1.6, sinks=2, mirror=True):
    """Floating wood vanity, stone top, vessel sinks, wall taps and a mirror. fr on the wall."""
    s.box("wood_panel", fr.pt((0, 0.27, 0.62)), (L / 2, 0.25, 0.2), R=fr.R, collide=True)
    s.box("bath_stone", fr.pt((0, 0.28, 0.84)), (L / 2 + 0.01, 0.27, 0.02), R=fr.R, collide=False)
    for i in range(sinks):
        x = -L / 2 + L / sinks * (i + 0.5)
        s.cylinder("ceramic_white", fr.pt((x, 0.3, 0.92)), 0.2, 0.06, R=fr.R, r_top=0.21, collide=False, seg=28)
        s.cylinder("stone_counter", fr.pt((x, 0.3, 0.975)), 0.17, 0.004, R=fr.R, collide=False, seg=28)
        s.tube("brass", (x, 0.02, 1.1), (x, 0.18, 1.1), 0.012, frame=fr)
        if mirror:
            s.cylinder("mirror", fr.pt((x, 0.012, 1.55)), 0.35, 0.008, R=fr.R @ rot_x(math.pi / 2), collide=False,
                       seg=40)
            s.cylinder("brass", fr.pt((x, 0.006, 1.55)), 0.365, 0.006, R=fr.R @ rot_x(math.pi / 2), collide=False,
                       seg=40)


def toilet(s, fr):
    """Wall-hung toilet with concealed cistern and flush plate. fr on the wall."""
    s.rbox("ceramic_white", (0, 0.3, 0.33), (0.18, 0.27, 0.09), 0.08, frame=fr)
    s.rbox("ceramic_white", (0, 0.32, 0.43), (0.19, 0.28, 0.015), 0.012, frame=fr)
    s.box("stainless", fr.pt((0, 0.01, 1.0)), (0.12, 0.008, 0.08), R=fr.R, collide=False)
    s.collider("box", (0, 0.3, 0.33), (0.19, 0.28, 0.12), frame=fr)
    s.tube("brass", (0.4, 0.02, 0.75), (0.4, 0.1, 0.75), 0.01, frame=fr)
    s.cylinder("fabric_white", fr.pt((0.4, 0.1, 0.75)), 0.06, 0.055, R=fr.R @ rot_y(math.pi / 2), collide=False)


def shower(s, fr, W=1.2, D=1.0, door_side=1):
    """Walk-in shower: glass screen, rain head, linear drain. fr at the back-wall centre."""
    s.box("glass", fr.pt((door_side * (W / 2 - 0.004) * 0 + 0, D, 1.05)), (W / 2 - 0.2, 0.005, 1.05), R=fr.R,
          collide=True)
    s.box("stainless", fr.pt((0, D, 2.09)), (W / 2 - 0.2, 0.012, 0.012), R=fr.R, collide=False)
    s.cylinder("stainless", fr.pt((0, D / 2, 2.25)), 0.15, 0.006, R=fr.R, collide=False, seg=28)
    s.tube("stainless", (0, 0.02, 2.3), (0, D / 2, 2.3), 0.01, frame=fr)
    s.box("stainless", fr.pt((0, 0.08, 0.002)), (W / 2 - 0.05, 0.03, 0.002), R=fr.R, collide=False)
    s.box("stainless", fr.pt((0.3, 0.01, 1.15)), (0.05, 0.01, 0.12), R=fr.R, collide=False)
    s.box("bath_stone", fr.pt((-W / 2 + 0.25, 0.08, 0.45)), (0.25, 0.08, 0.02), R=fr.R, collide=False)


def bathtub(s, fr, L=1.75, W=0.8, H=0.55):
    """Freestanding tub: hollow shell made of rounded walls + bottom, with a floor-mounted spout."""
    t = 0.06
    s.rbox("ceramic_white", (0, 0, 0.05), (W / 2, L / 2, 0.05), 0.04, frame=fr)
    for sx in (-1, 1):
        s.rbox("ceramic_white", (sx * (W / 2 - t / 2), 0, H / 2), (t / 2, L / 2, H / 2), 0.03, frame=fr)
    for sy in (-1, 1):
        s.rbox("ceramic_white", (0, sy * (L / 2 - t / 2), H / 2), (W / 2, t / 2, H / 2), 0.03, frame=fr)
    s.box("water_pool", fr.pt((0, 0, H * 0.6)), (W / 2 - t, L / 2 - t, 0.002), R=fr.R, collide=False)
    s.tube("stainless", (W / 2 + 0.15, L / 2 - 0.25, 0), (W / 2 + 0.15, L / 2 - 0.25, 0.9), 0.015, frame=fr)
    s.tube("stainless", (W / 2 + 0.15, L / 2 - 0.25, 0.9), (W / 2 - 0.05, L / 2 - 0.25, 0.9), 0.012, frame=fr)
    s.collider("box", (0, 0, H / 2), (W / 2, L / 2, H / 2), frame=fr)


def towel_rail(s, fr, w=0.6):
    s.tube("brass", (-w / 2, 0.06, 1.2), (w / 2, 0.06, 1.2), 0.01, frame=fr)
    s.rbox("fabric_white", (0, 0.08, 1.0), (w / 2 - 0.05, 0.02, 0.22), 0.015, frame=fr)


# ================================================================================================================
# kitchen (articulated drawers and doors)
# ================================================================================================================
def _drawer(s, name, fr_front, w, h, depth=0.5, front="wood_panel", handle=True, travel=0.42):
    """Drawer body: fr_front = centre of the closed front face (local +Y outwards). Slides along +Y."""
    b = s.body(name, fr_front, joint=dict(type="slide", axis=(0, 1, 0), range=(0, travel), damping=8.0,
                                          frictionloss=2.0, armature=0.5), mass=3.0, inertia=(0.08, 0.06, 0.08))
    with s.in_body(b):
        s.box(front, (0, -0.0095, 0), (w / 2 - 0.002, 0.0095, h / 2 - 0.002), collide=True)
        s.box("white_gloss", (0, -depth / 2 - 0.02, -h / 2 + 0.08), (w / 2 - 0.03, depth / 2, 0.006), collide=False)
        for sx in (-1, 1):
            s.box("white_gloss", (sx * (w / 2 - 0.035), -depth / 2 - 0.02, -h / 2 + 0.08 + (h - 0.12) / 2 * 0.5),
                  (0.006, depth / 2, (h - 0.12) / 4 + 0.04), collide=False)
        s.box("white_gloss", (0, -depth - 0.02, -h / 2 + 0.08 + (h - 0.12) / 4),
              (w / 2 - 0.03, 0.006, (h - 0.12) / 4 + 0.04), collide=False)
        if handle:
            s.box("steel_dark", (0, 0.004, h / 2 - 0.03), (min(0.25, w / 3), 0.006, 0.006), collide=False)
    s.note("articulated", name)
    return b


def _door(s, name, fr_hinge, w, h, side=1, mat="wood_panel", t=0.019, open_max=1.9, handle=True, mass=4.0):
    """Hinged leaf. fr_hinge at the hinge line, bottom of the leaf, local +X along the leaf width (towards the
    free edge), local +Y outwards. side=+1 opens outwards with positive angle about +Z (left hinge)."""
    rng = (0, open_max) if side > 0 else (-open_max, 0)
    b = s.body(name, fr_hinge, joint=dict(type="hinge", axis=(0, 0, 1), range=rng, damping=1.5,
                                          frictionloss=0.8, armature=0.05), mass=mass,
               inertia=(mass * h * h / 12, mass * (w * w + h * h) / 12, mass * w * w / 3))
    b.com = (side * w / 2, -t / 2, h / 2)
    with s.in_body(b):
        s.box(mat, (side * w / 2, -t / 2, h / 2), (w / 2 - 0.002, t / 2, h / 2 - 0.002), collide=True)
        if handle:
            s.box("steel_dark", (side * (w - 0.04), 0.008, h / 2), (0.006, 0.006, min(0.15, h / 4)), collide=False)
    s.note("articulated", name)
    return b


def kitchen_run(s, fr, modules, D=0.62, H=0.9, top="stone_counter", front="wood_panel", splash=None,
                wall_units=False, prefix="k"):
    """Base cabinets along local +X starting at x = 0, fronts facing local +Y (fr at back-left corner, floor).

    modules: list of (kind, width) with kind in drawers3, drawers2, door, door2, sink, dishwasher, cooktop,
    oven, fridge (tall), tall (pantry), gap. Returns total length."""
    x = 0.0
    sink_xs, cook_xs = [], []
    for kind, w in modules:
        cx = x + w / 2
        if kind == "gap":
            x += w
            continue
        tall = kind in ("fridge", "tall", "oven_tall")
        hh = 2.4 if tall else H - 0.04
        s.box(front if tall else "black_matte", fr.pt((cx, D / 2 - 0.02, 0.05)), (w / 2, D / 2 - 0.03, 0.05),
              R=fr.R, collide=False)
        s.box("wood_light", fr.pt((cx, D / 2 - 0.03, 0.1 + (hh - 0.1) / 2)), (w / 2 - 0.004, D / 2 - 0.03, (hh - 0.1) / 2),
              R=fr.R, collide=False)
        s.collider("box", (cx, D / 2 - 0.02, hh / 2), (w / 2, D / 2 - 0.02, hh / 2), frame=fr)
        yfront = D - 0.02
        if kind in ("drawers3", "drawers2", "cooktop"):
            n = 3 if kind != "drawers2" else 2
            hs = [0.2, 0.28, 0.28] if n == 3 else [0.38, 0.38]
            z = 0.1
            for k, dh in enumerate(hs[::-1]):
                _drawer(s, uid(f"{prefix}_drawer"), fr.child((cx, yfront + 0.019, z + dh / 2)), w, dh,
                        depth=D - 0.12, front=front)
                z += dh
            if kind == "cooktop":
                cook_xs.append(cx)
        elif kind in ("door", "door2", "sink"):
            hd = H - 0.04 - 0.1
            if kind == "door":
                _door(s, uid(f"{prefix}_door"), fr.child((cx - w / 2, yfront + 0.019, 0.1)), w, hd, side=1, mat=front)
            else:
                _door(s, uid(f"{prefix}_door"), fr.child((cx - w / 2, yfront + 0.019, 0.1)), w / 2, hd, side=1,
                      mat=front)
                _door(s, uid(f"{prefix}_door"), fr.child((cx + w / 2, yfront + 0.019, 0.1)), w / 2, hd, side=-1,
                      mat=front)
            if kind == "sink":
                sink_xs.append(cx)
        elif kind == "dishwasher":
            b = s.body(uid(f"{prefix}_dishwasher"), fr.child((cx, yfront + 0.019, 0.1)),
                       joint=dict(type="hinge", axis=(1, 0, 0), range=(0, 1.5), damping=4.0, frictionloss=3.0),
                       mass=6.0, inertia=(0.3, 0.3, 0.1))
            b.com = (0, -0.01, 0.36)
            with s.in_body(b):
                s.box(front, (0, -0.0095, 0.38), (w / 2 - 0.002, 0.0095, 0.38), collide=True)
                s.box("steel_dark", (0, 0.006, 0.72), (w / 3, 0.006, 0.006), collide=False)
            s.note("articulated", "dishwasher")
        elif kind in ("oven_tall",):
            _door(s, uid(f"{prefix}_tall"), fr.child((cx - w / 2, yfront + 0.019, 1.75)), w, 0.65, side=1, mat=front)
            b = s.body(uid(f"{prefix}_oven"), fr.child((cx, yfront + 0.02, 0.85)),
                       joint=dict(type="hinge", axis=(1, 0, 0), range=(0, 1.55), damping=3.0, frictionloss=2.0),
                       mass=5.0, inertia=(0.2, 0.2, 0.05))
            b.com = (0, -0.01, 0.28)
            with s.in_body(b):
                s.box("black_glass", (0, -0.012, 0.29), (w / 2 - 0.01, 0.012, 0.29), collide=True)
                s.box("stainless", (0, 0.012, 0.52), (w / 2 - 0.08, 0.008, 0.01), collide=False)
                s.box("stainless", (0, -0.0, 0.57), (w / 2 - 0.01, 0.002, 0.012), collide=False)
            s.note("articulated", "oven door")
            s.box("black_glass", fr.pt((cx, yfront + 0.01, 1.55)), (w / 2 - 0.01, 0.01, 0.18), R=fr.R, collide=False)
            s.box("stainless", fr.pt((cx, yfront + 0.022, 1.66)), (w / 2 - 0.1, 0.004, 0.02), R=fr.R, collide=False)
            for k, dh in enumerate((0.36, 0.36)):
                _drawer(s, uid(f"{prefix}_drawer"), fr.child((cx, yfront + 0.019, 0.1 + 0.18 + k * 0.36)), w, 0.36,
                        depth=D - 0.12, front=front)
        elif kind == "fridge":
            for sx, hh0, hh1 in ((-1, 0.1, 2.38),):
                _door(s, uid(f"{prefix}_fridge"), fr.child((cx - w / 2, yfront + 0.019, 0.1)), w / 2, 2.28, side=1,
                      mat=front, mass=12.0)
                _door(s, uid(f"{prefix}_fridge"), fr.child((cx + w / 2, yfront + 0.019, 0.1)), w / 2, 2.28, side=-1,
                      mat=front, mass=12.0)
            s.note("articulated", "fridge doors")
        elif kind == "tall":
            _door(s, uid(f"{prefix}_tall"), fr.child((cx - w / 2, yfront + 0.019, 0.1)), w, 2.28, side=1, mat=front)
        x += w
    L = x
    # countertop with sink cut-outs
    xs = sorted(sink_xs)
    segs, cur = [], 0.0
    for sx_ in xs:
        segs.append((cur, sx_ - 0.4))
        cur = sx_ + 0.4
    segs.append((cur, L))
    tops = [m for m in modules]
    for a, b_ in segs:
        if b_ - a > 0.01:
            s.box(top, fr.pt(((a + b_) / 2, D / 2, H - 0.02)), ((b_ - a) / 2, D / 2 + 0.01, 0.02), R=fr.R,
                  collide=False)
    for sx_ in xs:
        s.box(top, fr.pt((sx_, 0.06, H - 0.02)), (0.4, 0.06, 0.02), R=fr.R, collide=False)
        s.box(top, fr.pt((sx_, D - 0.04, H - 0.02)), (0.4, 0.05, 0.02), R=fr.R, collide=False)
        s.box("stainless", fr.pt((sx_, D / 2 + 0.01, H - 0.12)), (0.38, 0.2, 0.004), R=fr.R, collide=False)
        for sgn in (-1, 1):
            s.box("stainless", fr.pt((sx_ + sgn * 0.38, D / 2 + 0.01, H - 0.07)), (0.004, 0.2, 0.05), R=fr.R,
                  collide=False)
            s.box("stainless", fr.pt((sx_, D / 2 + 0.01 + sgn * 0.2, H - 0.07)), (0.38, 0.004, 0.05), R=fr.R,
                  collide=False)
        s.tube("stainless", (sx_, 0.06, H), (sx_, 0.06, H + 0.38), 0.014, frame=fr)
        s.tube("stainless", (sx_, 0.06, H + 0.38), (sx_, 0.26, H + 0.3), 0.012, frame=fr)
        s.note("kitchen", "sink with tap")
    for cx_ in cook_xs:
        s.box("black_glass", fr.pt((cx_, D / 2, H + 0.003)), (0.38, 0.26, 0.003), R=fr.R, collide=False)
        for dx in (-0.18, 0.18):
            for dy in (-0.1, 0.1):
                s.cylinder("steel_dark", fr.pt((cx_ + dx, D / 2 + dy, H + 0.0065)), 0.08, 0.0005, R=fr.R,
                           collide=False)
        s.note("kitchen", "induction cooktop")
    s.collider("box", (L / 2, D / 2, H - 0.02), (L / 2, D / 2 + 0.01, 0.02), frame=fr)
    if splash:
        s.box(splash, fr.pt((L / 2, 0.01, H + 0.35)), (L / 2, 0.01, 0.35), R=fr.R, collide=False)
    if wall_units:
        s.box(front, fr.pt((L / 2, 0.18, H + 0.95)), (L / 2, 0.18, 0.38), R=fr.R, collide=True)
    return L


def island(s, fr, L=3.6, W=1.0, H=0.92, cooktop=True, sink=False, prefix="isl"):
    """Kitchen island: wood-clad sides, stone top, cooktop and drawers on the working side (local -Y)."""
    s.box("wood_panel", fr.pt((0, 0, (H - 0.04) / 2)), (L / 2, W / 2, (H - 0.04) / 2), R=fr.R, collide=False,
          uv_rot=0.0)
    s.box("stone_counter", fr.pt((0, 0, H - 0.02)), (L / 2 + 0.02, W / 2 + 0.02, 0.02), R=fr.R, collide=False)
    s.collider("box", (0, 0, H / 2), (L / 2 + 0.02, W / 2 + 0.02, H / 2), frame=fr)
    if cooktop:
        cx = L / 2 - 0.6
        s.box("black_glass", fr.pt((cx, 0, H + 0.003)), (0.42, 0.27, 0.003), R=fr.R, collide=False)
        for dx in (-0.2, 0.2):
            for dy in (-0.11, 0.11):
                s.cylinder("steel_dark", fr.pt((cx + dx, dy, H + 0.0065)), 0.085, 0.0005, R=fr.R, collide=False)
        s.note("kitchen", "island cooktop")
    back = fr.child((0, -W / 2, 0), yaw=math.pi)
    n = int(L / 0.6)
    w = L / n
    for i in range(n):
        x = -L / 2 + w * (i + 0.5)
        _drawer(s, uid(f"{prefix}_drawer"), back.child((x, 0.019, 0.62)), w, 0.24, depth=0.5)
        _drawer(s, uid(f"{prefix}_drawer"), back.child((x, 0.019, 0.3)), w, 0.38, depth=0.5)
    s.box("wood_dark", fr.pt((-L / 2 + 0.5, 0, H + 0.04)), (0.28, 0.2, 0.012), R=fr.R, collide=False)


def range_hood(s, fr, z0, z1, w=0.9):
    s.box("stainless", fr.pt((0, 0, z0 + 0.05)), (w / 2, 0.25, 0.05), R=fr.R, collide=False)
    s.box("stainless", fr.pt((0, -0.1, (z0 + z1) / 2 + 0.05)), (0.15, 0.12, (z1 - z0) / 2 - 0.05), R=fr.R,
          collide=False)


# ================================================================================================================
# service, leisure, outdoors
# ================================================================================================================
def washer(s, fr):
    s.rbox("white_gloss", (0, 0, 0.43), (0.3, 0.3, 0.43), 0.02, frame=fr, collide=True)
    s.cylinder("black_glass", fr.pt((0, 0.3, 0.45)), 0.17, 0.01, R=fr.R @ rot_x(math.pi / 2), collide=False)
    s.cylinder("stainless", fr.pt((0, 0.299, 0.45)), 0.2, 0.006, R=fr.R @ rot_x(math.pi / 2), collide=False)
    s.box("black_glass", fr.pt((0.1, 0.3, 0.78)), (0.15, 0.003, 0.04), R=fr.R, collide=False)


def foosball(s, fr):
    s.box("wood_dark", fr.pt((0, 0, 0.82)), (0.6, 0.37, 0.1), R=fr.R, collide=False)
    s.box("felt_green", fr.pt((0, 0, 0.86)), (0.58, 0.35, 0.002), R=fr.R, collide=False)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.box("wood_dark", fr.pt((sx * 0.5, sy * 0.3, 0.36)), (0.04, 0.04, 0.36), R=fr.R, collide=False)
    for k in range(8):
        x = -0.52 + k * 0.15
        s.tube("stainless", (x, -0.55, 0.88), (x, 0.55, 0.88), 0.008, frame=fr)
        for j in range(3):
            s.box("book_red" if k % 2 else "book_blue", fr.pt((x, -0.2 + j * 0.2, 0.87)), (0.012, 0.015, 0.05),
                  R=fr.R, collide=False)
    s.collider("box", (0, 0, 0.46), (0.6, 0.37, 0.46), frame=fr)


def massage_table(s, fr):
    s.rbox("leather_tan", (0, 0, 0.72), (0.35, 0.95, 0.05), 0.04, frame=fr)
    for sy in (-1, 1):
        s.box("wood_light", fr.pt((0, sy * 0.75, 0.34)), (0.3, 0.03, 0.34), R=fr.R, collide=False)
    s.collider("box", (0, 0, 0.38), (0.36, 0.96, 0.38), frame=fr)
    s.rbox("fabric_white", (0, -0.6, 0.79), (0.3, 0.2, 0.02), 0.015, frame=fr)


def sauna(s, fr, W=2.0, D=2.2, H=2.3):
    """Timber sauna cabin with two bench tiers and a glass door. fr at the front-left corner, interior +Y back."""
    s.box("wood_light", fr.pt((W / 2, D, H / 2)), (W / 2, 0.04, H / 2), R=fr.R, collide=True)
    for sx in (0, W):
        s.box("wood_light", fr.pt((sx, D / 2, H / 2)), (0.04, D / 2, H / 2), R=fr.R, collide=True)
    s.box("glass_frosted", fr.pt((W / 2, 0, H / 2)), (W / 2, 0.02, H / 2), R=fr.R, collide=True)
    s.box("wood_light", fr.pt((W / 2, D - 0.3, 0.45)), (W / 2 - 0.05, 0.3, 0.03), R=fr.R, collide=False)
    s.box("wood_light", fr.pt((W / 2, D - 0.25, 0.95)), (W / 2 - 0.05, 0.25, 0.03), R=fr.R, collide=False)
    s.box("steel_dark", fr.pt((0.35, 0.4, 0.4)), (0.2, 0.2, 0.4), R=fr.R, collide=False)
    for k in range(6):
        s.sphere("rock", fr.pt((0.25 + 0.07 * (k % 3), 0.33 + 0.07 * (k // 3), 0.84)), 0.05, scale=(1, 1, 0.7))


def car(s, fr, colour="car_paint"):
    """Simplified SUV parked (visual + box collider)."""
    s.rbox(colour, (0, 0, 0.75), (0.95, 2.35, 0.4), 0.2, frame=fr)
    s.rbox(colour, (0, -0.25, 1.32), (0.85, 1.4, 0.28), 0.18, frame=fr)
    s.rbox("black_glass", (0, -0.25, 1.34), (0.86, 1.3, 0.22), 0.15, frame=fr)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.cylinder("tyre", fr.pt((sx * 0.86, sy * 1.45, 0.38)), 0.38, 0.14, R=fr.R @ rot_y(math.pi / 2),
                       collide=False)
            s.cylinder("stainless", fr.pt((sx * 0.93, sy * 1.45, 0.38)), 0.24, 0.08, R=fr.R @ rot_y(math.pi / 2),
                       collide=False)
    s.box("lamp_glow", fr.pt((0, 2.34, 0.85)), (0.7, 0.01, 0.04), R=fr.R, collide=False)
    s.collider("box", (0, 0, 0.8), (0.97, 2.37, 0.8), frame=fr)


def generator(s, fr):
    """Yellow standby generator (photo 6)."""
    s.rbox("generator_yellow", (0, 0, 0.55), (0.55, 1.05, 0.55), 0.04, frame=fr, collide=True)
    s.box("black_matte", fr.pt((0.551, 0, 0.6)), (0.002, 0.7, 0.3), R=fr.R, collide=False)


def boat(s, fr):
    """Small runabout moored at the pier (visual only, floats on the sea)."""
    v = []
    f = []
    prof = [(-2.4, 0.0), (-2.4, 0.85), (1.0, 0.95), (2.6, 0.0)]
    n = 10
    for i in range(n + 1):
        t = i / n
        y = -2.4 + 5.0 * t
        w = 0.95 * min(1.0, (2.6 - y) / 1.6) if y > 1.0 else 0.95
        w = max(w, 0.05)
        zt, zb = 0.55, (0.0 if y < 1.8 else 0.25 * (y - 1.8) / 0.8)
        v += [(-w, y, zt), (w, y, zt), (w * 0.6, y, zb), (-w * 0.6, y, zb)]
    for i in range(n):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            f += [(a + k, a + k2, b + k2), (a + k, b + k2, b + k)]
    s.trimesh("boat_white", np.array(v), np.array(f), frame=fr, smooth=False)
    s.box("deck_brown", fr.pt((0, -0.4, 0.5)), (0.85, 1.6, 0.02), R=fr.R, collide=False)
    s.box("glass", fr.pt((0, 0.6, 0.85)), (0.8, 0.02, 0.3), R=fr.R @ rot_x(0.4), collide=False)
    s.rbox("fabric_white", (0, -1.4, 0.65), (0.7, 0.5, 0.12), 0.08, frame=fr)
