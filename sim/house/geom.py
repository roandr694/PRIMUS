"""Geometry builder for the house scene.

Everything is added to a `Scene`:
  * visual meshes, merged per material, with world-scale texture coordinates (1 texture tile = `tile` metres of the
    material), written as MuJoCo binary .msh files and reused by the USD export;
  * primitive colliders (boxes, cylinders, spheres, capsules) for the physics: invisible (alpha 0) world geoms in
    group 0, so the robot, the simulated lidar (which casts against group 0) and the depth camera (which renders
    the visual meshes) all see the same house;
  * articulated bodies (doors, drawers, appliance doors, the inclined lift) with their own visual meshes and
    colliders.

Frames: X east (towards the hill), Y north, Z up, metres; z = 0 is the social (living room) floor.
"""
import math
import struct
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

# collision bitmasks: robot geoms have contype 2 / conaffinity 1 (robo_bipedal.xml)
STATIC_CT, STATIC_CA = 1, 1          # house structure: collides with the robot, not with itself (same body)
MOVING_CT, MOVING_CA = 4, 2          # doors, drawers, lift: collide with the robot only


# ----------------------------------------------------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------------------------------------------------
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def mat2quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


class Frame:
    """Rigid transform used to place furniture: world = R @ local + t."""

    def __init__(self, pos=(0, 0, 0), yaw=0.0, R=None):
        self.t = np.asarray(pos, float)
        self.R = np.asarray(R, float) if R is not None else rot_z(yaw)

    def __mul__(self, other):
        return Frame(self.R @ other.t + self.t, R=self.R @ other.R)

    def pt(self, p):
        return self.R @ np.asarray(p, float) + self.t

    def child(self, pos=(0, 0, 0), yaw=0.0, R=None):
        return self * Frame(pos, yaw, R)


IDENT = Frame()


# ----------------------------------------------------------------------------------------------------------------
# mesh buffers
# ----------------------------------------------------------------------------------------------------------------
class MeshBuf:
    def __init__(self):
        self.v, self.n, self.t, self.f = [], [], [], []
        self.count = 0

    def add(self, v, n, t, f):
        v = np.asarray(v, np.float64)
        if len(v) == 0:
            return
        self.v.append(v)
        self.n.append(np.asarray(n, np.float64))
        self.t.append(np.asarray(t, np.float64))
        self.f.append(np.asarray(f, np.int64) + self.count)
        self.count += len(v)

    def arrays(self):
        if not self.v:
            return None
        return (np.concatenate(self.v), np.concatenate(self.n), np.concatenate(self.t), np.concatenate(self.f))

    def __len__(self):
        return self.count


def write_msh(path, v, n, t, f):
    with open(path, "wb") as fh:
        fh.write(struct.pack("4i", len(v), len(n), len(t), len(f)))
        fh.write(np.asarray(v, np.float32).tobytes())
        fh.write(np.asarray(n, np.float32).tobytes())
        fh.write(np.asarray(t, np.float32).tobytes())
        fh.write(np.asarray(f, np.int32).tobytes())


# ----------------------------------------------------------------------------------------------------------------
# primitive mesh generators (local frame); UVs in metres, scaled per material when added to a buffer
# ----------------------------------------------------------------------------------------------------------------
_FACES = [  # (axis, sign, tangent u axis, tangent v axis) with u x v = outward normal
    (2, +1, 0, 1), (2, -1, 1, 0), (0, +1, 1, 2), (0, -1, 2, 1), (1, +1, 2, 0), (1, -1, 0, 2)]


def box_mesh(h, skip=()):
    """Box with half sizes h centred at the origin. Returns v, n, uv(metres, local), f. skip: faces '+z','-x'..."""
    h = np.asarray(h, float)
    V, N, T, F = [], [], [], []
    names = {(2, 1): "+z", (2, -1): "-z", (0, 1): "+x", (0, -1): "-x", (1, 1): "+y", (1, -1): "-y"}
    for ax, sg, a, b in _FACES:
        if names[(ax, sg)] in skip:
            continue
        n = np.zeros(3)
        n[ax] = sg
        base = len(V)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[ax] = sg * h[ax]
            p[a] = su * h[a]
            p[b] = sv * h[b]
            V.append(p)
            N.append(n)
            # texture axes: horizontal faces use (x, y); vertical faces use (horizontal, -z) so images stand upright
            if ax == 2:
                T.append((p[0], p[1]))
            elif ax == 0:
                T.append((p[1] * sg, -p[2]))
            else:
                T.append((-p[0] * sg, -p[2]))
        F += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(V), np.array(N), np.array(T), np.array(F)


def cylinder_mesh(r, half_h, seg=24, caps=True, r_top=None, uv_scale_angle=True):
    r_top = r if r_top is None else r_top
    V, N, T, F = [], [], [], []
    slope = (r - r_top) / (2 * half_h) if half_h > 0 else 0
    for i in range(seg + 1):
        a = 2 * math.pi * i / seg
        c, s = math.cos(a), math.sin(a)
        n = np.array([c, s, slope])
        n /= np.linalg.norm(n)
        u = a * r if uv_scale_angle else i / seg
        V += [(r * c, r * s, -half_h), (r_top * c, r_top * s, half_h)]
        N += [n, n]
        T += [(u, half_h), (u, -half_h)]
    for i in range(seg):
        k = 2 * i
        F += [(k, k + 2, k + 3), (k, k + 3, k + 1)]
    if caps:
        for z, rr, sg in ((half_h, r_top, 1), (-half_h, r, -1)):
            if rr <= 0:
                continue
            c0 = len(V)
            V.append((0, 0, z))
            N.append((0, 0, sg))
            T.append((0, 0))
            for i in range(seg):
                a = 2 * math.pi * i / seg
                V.append((rr * math.cos(a), rr * math.sin(a), z))
                N.append((0, 0, sg))
                T.append((rr * math.cos(a), rr * math.sin(a)))
            for i in range(seg):
                j, k = c0 + 1 + i, c0 + 1 + (i + 1) % seg
                F.append((c0, j, k) if sg > 0 else (c0, k, j))
    return np.array(V, float), np.array(N, float), np.array(T, float), np.array(F)


def sphere_mesh(r, seg=16, rings=10, scale=(1, 1, 1), z_min=-1.0):
    V, N, T, F = [], [], [], []
    sc = np.asarray(scale, float)
    for j in range(rings + 1):
        th = math.pi * j / rings
        for i in range(seg + 1):
            ph = 2 * math.pi * i / seg
            d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            d[2] = max(d[2], z_min)
            p = r * d * sc
            n = d / sc
            n /= np.linalg.norm(n) + 1e-12
            V.append(p)
            N.append(n)
            T.append((ph * r, th * r))
    for j in range(rings):
        for i in range(seg):
            a = j * (seg + 1) + i
            b = a + seg + 1
            F += [(a, b, b + 1), (a, b + 1, a + 1)]
    return np.array(V), np.array(N), np.array(T), np.array(F)


def rounded_box_mesh(h, r, res=6):
    """Box with rounded edges (cushions, upholstery). h: half sizes, r: edge radius (<= min(h))."""
    h = np.asarray(h, float)
    r = min(r, *(h * 0.999))
    inner = h - r
    V, N, T, F = [], [], [], []
    for ax, sg, a, b in _FACES:
        base = len(V)
        for j in range(res + 1):
            for i in range(res + 1):
                p = np.zeros(3)
                p[ax] = sg * h[ax]
                p[a] = (-1 + 2 * i / res) * h[a]
                p[b] = (-1 + 2 * j / res) * h[b]
                q = np.clip(p, -inner, inner)
                d = p - q
                dn = np.linalg.norm(d)
                d = d / dn if dn > 1e-9 else np.eye(3)[ax] * sg
                pp = q + r * d
                V.append(pp)
                N.append(d)
                if ax == 2:
                    T.append((pp[0], pp[1]))
                elif ax == 0:
                    T.append((pp[1] * sg, -pp[2]))
                else:
                    T.append((-pp[0] * sg, -pp[2]))
        for j in range(res):
            for i in range(res):
                k = base + j * (res + 1) + i
                F += [(k, k + 1, k + res + 2), (k, k + res + 2, k + res + 1)]
    return np.array(V), np.array(N), np.array(T), np.array(F)


def triangulate(poly):
    """Ear clipping for a simple CCW polygon (list of (x, y)). Returns triangles as index triples."""
    pts = [tuple(p) for p in poly]
    idx = list(range(len(pts)))
    if _area(pts) < 0:
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue
            if any(_in_tri(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _area(pts):
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def _in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def prism_mesh(poly, z0, z1, top=True, bottom=True, sides=True):
    poly = [tuple(map(float, p)) for p in poly]
    if _area(poly) < 0:
        poly = poly[::-1]
    V, N, T, F = [], [], [], []
    tris = triangulate(poly)
    if top:
        b = len(V)
        for x, y in poly:
            V.append((x, y, z1)); N.append((0, 0, 1)); T.append((x, y))
        F += [(b + i, b + j, b + k) for i, j, k in tris]
    if bottom:
        b = len(V)
        for x, y in poly:
            V.append((x, y, z0)); N.append((0, 0, -1)); T.append((x, y))
        F += [(b + i, b + k, b + j) for i, j, k in tris]
    if sides:
        acc = 0.0
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-9:
                continue
            n = (( y1 - y0) / L, -(x1 - x0) / L, 0)
            b = len(V)
            V += [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
            N += [n] * 4
            T += [(acc, -z0), (acc + L, -z0), (acc + L, -z1), (acc, -z1)]
            F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
            acc += L
    return np.array(V, float), np.array(N, float), np.array(T, float), np.array(F)


def flat_normals(v, f):
    """Unshares vertices: per-face normals (faceted look)."""
    v = np.asarray(v, float)
    f = np.asarray(f)
    tri = v[f]                                       # (M,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    V = tri.reshape(-1, 3)
    N = np.repeat(n, 3, axis=0)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, N, F


def smooth_normals(v, f):
    v = np.asarray(v, float)
    f = np.asarray(f)
    n = np.zeros_like(v)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def box_uv(v, n):
    """Tri-planar UVs (metres) from positions and normals."""
    v = np.asarray(v, float)
    n = np.asarray(n, float)
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.where(ax[:, None] == 2, v[:, [0, 1]], np.where(ax[:, None] == 0, np.c_[v[:, 1], -v[:, 2]],
                                                           np.c_[v[:, 0], -v[:, 2]]))
    return uv


# ----------------------------------------------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------------------------------------------
class Body:
    """An articulated rigid body (door leaf, drawer, appliance door, lift cabin)."""

    def __init__(self, name, frame, joint, mass, inertia=None, parent=None):
        self.name, self.frame, self.joint, self.mass = name, frame, joint, mass
        self.inertia = inertia
        self.parent = parent
        self.children = []
        self.meshes = defaultdict(MeshBuf)
        self.colliders = []
        self.sites = []
        self.extra_xml = ""
        self.com = (0, 0, 0)
        self.gravcomp = 0.0


class Scene:
    def __init__(self, materials):
        self.materials = materials          # name -> Material (materials.py)
        self.meshes = defaultdict(MeshBuf)  # static visual meshes
        self.colliders = []
        self.bodies = []
        self.lights = []
        self.cameras = []
        self.planes = []                    # visual planes (water) rendered with reflections
        self.sites = []
        self.actuators = []
        self.notes = defaultdict(list)      # room / object inventory for the README and the spawn points
        self.spawns = {}                    # name -> (x, y, z_floor, yaw)
        self.target = None                  # current Body or None (static)
        self.joint_defaults = {}            # body name -> initial joint position for the keyframes
        self.hfields = []                   # terrain heightfields (collision)

    # ------------------------------------------------------------------ helpers
    def _buf(self, mat):
        if mat not in self.materials:
            raise KeyError(f"unknown material {mat}")
        return (self.target.meshes if self.target is not None else self.meshes)[mat]

    def _tile(self, mat):
        return self.materials[mat].tile

    def add_mesh(self, mat, v, n, uv_m, f, frame=IDENT, uv_world=False, uv_rot=0.0, uv_off=(0, 0)):
        """Adds a local-frame mesh. uv_m are metres in the local frame; uv_world recomputes them from world
        positions (continuous textures across adjacent pieces of floor/wall)."""
        v = np.asarray(v, float)
        n = np.asarray(n, float)
        Vw = v @ frame.R.T + frame.t
        Nw = n @ frame.R.T
        if uv_world:
            uv = box_uv(Vw, Nw)
        else:
            uv = np.asarray(uv_m, float)
        if uv_rot:
            c, s = math.cos(uv_rot), math.sin(uv_rot)
            uv = uv @ np.array([[c, s], [-s, c]])
        uv = (uv + np.asarray(uv_off, float)) / self._tile(mat)
        self._buf(mat).add(Vw, Nw, uv, f)

    def collider(self, kind, pos, size, R=None, friction=(1.0, 0.02, 0.002), name=None, solref=None, frame=IDENT):
        p = frame.pt(pos)
        Rw = frame.R @ (np.eye(3) if R is None else np.asarray(R, float))
        c = dict(kind=kind, pos=p, size=np.atleast_1d(np.asarray(size, float)), R=Rw, friction=friction,
                 name=name, solref=solref)
        (self.target.colliders if self.target is not None else self.colliders).append(c)
        return c

    # ------------------------------------------------------------------ primitives (visual + optional collider)
    def box(self, mat, center, half, frame=IDENT, R=None, collide=True, friction=None, uv_world=None, uv_rot=0.0,
            uv_off=(0, 0), skip=(), name=None):
        """Axis-aligned (in `frame`, optionally rotated by R) textured box. Static, axis-aligned boxes default to
        world UVs so floors/walls made of several boxes have continuous textures."""
        half = np.asarray(half, float)
        v, n, t, f = box_mesh(half, skip)
        Rl = np.eye(3) if R is None else np.asarray(R, float)
        fr = frame.child(center, R=Rl)
        if uv_world is None:
            uv_world = (self.target is None and R is None and np.allclose(frame.R, np.eye(3)))
        if not uv_world:
            uv_off = (uv_off[0] + 0.37 * fr.t[0] + 0.21 * fr.t[2], uv_off[1] + 0.29 * fr.t[1])
        self.add_mesh(mat, v, n, t, f, fr, uv_world=uv_world, uv_rot=uv_rot, uv_off=uv_off)
        if collide:
            fric = friction if friction is not None else self.materials[mat].friction
            self.collider("box", (0, 0, 0), half, friction=fric, name=name, frame=fr)

    def box2(self, mat, lo, hi, **kw):
        """Box from min/max corners (world or frame coordinates)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.box(mat, (lo + hi) / 2, np.abs(hi - lo) / 2, **kw)

    def cylinder(self, mat, center, r, half_h, frame=IDENT, R=None, collide=True, seg=24, r_top=None,
                 friction=None, caps=True):
        v, n, t, f = cylinder_mesh(r, half_h, seg, caps=caps, r_top=r_top)
        fr = frame.child(center, R=np.eye(3) if R is None else R)
        self.add_mesh(mat, v, n, t, f, fr)
        if collide:
            fric = friction if friction is not None else self.materials[mat].friction
            self.collider("cylinder", (0, 0, 0), (max(r, r_top or 0), half_h), friction=fric, frame=fr)

    def sphere(self, mat, center, r, frame=IDENT, collide=False, scale=(1, 1, 1), seg=16, rings=10, z_min=-1.0,
               R=None):
        v, n, t, f = sphere_mesh(r, seg, rings, scale, z_min)
        fr = frame.child(center, R=np.eye(3) if R is None else R)
        self.add_mesh(mat, v, n, t, f, fr)
        if collide:
            self.collider("sphere", (0, 0, 0), (r * max(scale),), friction=self.materials[mat].friction,
                          frame=fr)

    def rbox(self, mat, center, half, r, frame=IDENT, R=None, collide=False, res=5):
        v, n, t, f = rounded_box_mesh(half, r, res)
        fr = frame.child(center, R=np.eye(3) if R is None else R)
        self.add_mesh(mat, v, n, t, f, fr)
        if collide:
            self.collider("box", (0, 0, 0), half, friction=self.materials[mat].friction, frame=fr)

    def prism(self, mat, poly, z0, z1, frame=IDENT, collide=False, top=True, bottom=True, sides=True,
              uv_world=True):
        v, n, t, f = prism_mesh(poly, z0, z1, top, bottom, sides)
        self.add_mesh(mat, v, n, t, f, frame, uv_world=uv_world and self.target is None)

    def tube(self, mat, p0, p1, r, frame=IDENT, collide=False, seg=12):
        """Cylinder between two points (rails, rods, legs)."""
        p0, p1 = frame.pt(p0), frame.pt(p1)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-9:
            return
        z = d / L
        x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.c_[x, y, z]
        self.cylinder(mat, (p0 + p1) / 2, r, L / 2, R=R, collide=collide, seg=seg)

    def quad(self, mat, corners, frame=IDENT, double=True):
        """Flat quad from 4 local corners (CCW seen from the front); double-sided by default."""
        c = np.asarray(corners, float)
        n = np.cross(c[1] - c[0], c[2] - c[0])
        n /= np.linalg.norm(n) + 1e-12
        v = np.r_[c, c] if double else c
        nn = np.r_[[n] * 4, [-n] * 4] if double else np.array([n] * 4)
        f = [(0, 1, 2), (0, 2, 3)] + ([(4, 6, 5), (4, 7, 6)] if double else [])
        self.add_mesh(mat, v, nn, box_uv(v, nn), f, frame)

    def trimesh(self, mat, v, f, frame=IDENT, smooth=True, uv=None):
        v = np.asarray(v, float)
        f = np.asarray(f)
        if smooth:
            n = smooth_normals(v, f)
            t = uv if uv is not None else box_uv(v, n)
            self.add_mesh(mat, v, n, t, f, frame)
        else:
            V, N, F = flat_normals(v, f)
            self.add_mesh(mat, V, N, box_uv(V, N) if uv is None else uv[f.reshape(-1)], F, frame)

    # ------------------------------------------------------------------ bodies
    def body(self, name, frame, joint=None, mass=5.0, inertia=None, parent=None):
        """joint: dict(type='hinge'|'slide', axis=(..), range=(lo, hi), damping, stiffness, springref, ref, qpos0,
        frictionloss, armature) in the body frame; geometry added inside `with scene.in_body(b)` is body-local.
        With `parent`, `frame` is relative to the parent body."""
        b = Body(name, frame, joint, mass, inertia, parent)
        (parent.children if parent is not None else self.bodies).append(b)
        self.joint_defaults[name] = (joint or {}).get("qpos0", 0.0)
        return b

    @contextmanager
    def in_body(self, b):
        prev = self.target
        self.target = b
        try:
            yield b
        finally:
            self.target = prev

    def light(self, **kw):
        self.lights.append(kw)

    def note(self, room, item):
        self.notes[room].append(item)
