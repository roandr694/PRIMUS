"""Procedural tropical vegetation (visual meshes; trunks get cylinder colliders): coconut palms, areca palm
clumps, frangipani, broadleaf trees, variegated ginger/heliconia clumps (photos 6, 7, 15), bamboo, ornamental
grasses and shrubs. Everything is deterministic for a given seed.
"""
import math

import numpy as np

# Level of detail of the outdoor foliage (set by build.py from --quality): "high" keeps every leaf segment and
# midrib; "low" (default, real-time viewers) uses 3-segment flat leaves and fewer, larger leaves per cluster.
LOD = {"level": "low"}


def _leaf(length, width, bend, n=5, midrib=True):
    """Lanceolate leaf along +X in the XY plane, bent down along its length. Returns vertices, faces."""
    if LOD["level"] == "low" and not _leaf.near:
        n, midrib = 3, False
    V = []
    for i in range(n + 1):
        t = i / n
        w = width * math.sin(math.pi * min(1.0, t * 1.05)) ** 0.8 / 2
        x = length * t
        z = -bend * length * t * t
        V += [(x, -w, z), (x, w, z), (x, 0, z + 0.08 * width)]
    F = []
    for i in range(n):
        a, b = 3 * i, 3 * (i + 1)
        if midrib:
            F += [(a, b, b + 2), (a, b + 2, a + 2), (a + 2, b + 2, b + 1), (a + 2, b + 1, a + 1)]
        else:
            F += [(a, b, b + 1), (a, b + 1, a + 1)]
    return np.array(V, float), F


_leaf.near = False          # True while building plants seen up close (pots, planters): always full detail


def near_detail(flag):
    _leaf.near = flag


def _frame_from_dir(d, roll=0.0):
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([1.0, 0, 0])
    y = np.cross(up, d)
    y /= np.linalg.norm(y)
    z = np.cross(d, y)
    c, s = math.cos(roll), math.sin(roll)
    y, z = c * y + s * z, -s * y + c * z
    return np.c_[d, y, z]


def _add_double(s, mat, V, F):
    """Adds a two-sided leaf mesh in world coordinates (faces duplicated with reversed winding)."""
    F = np.asarray(F)
    F2 = np.r_[F, F[:, ::-1]]
    s.trimesh(mat, V, F2, smooth=False)


def leaf_cluster(s, centre, radius, n=30, mat="leaf_dark", seed=0, elong=1.0, leaf_len=None, droop=0.5):
    """A shrub / potted plant: leaves radiating from a centre, distributed in an ellipsoid."""
    rng = np.random.default_rng(seed)
    c = np.asarray(centre, float)
    L = leaf_len or radius * 0.7
    allV, allF = [], []
    for k in range(n):
        az = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(-0.2, 1.0)
        d = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el) * elong])
        base = c + rng.normal(0, radius * 0.18, 3) * np.array([1, 1, elong])
        v, f = _leaf(L * rng.uniform(0.7, 1.2), L * rng.uniform(0.25, 0.4), droop * rng.uniform(0.5, 1.2))
        R = _frame_from_dir(d, rng.uniform(-0.6, 0.6))
        off = sum(len(x) for x in allV)
        allV.append(v @ R.T + base)
        allF += [(a + off, b + off, cc + off) for a, b, cc in f]
    _add_double(s, mat, np.concatenate(allV), allF)


def coconut_palm(s, base, height=9.0, lean=(0.6, 0.0), seed=0, fronds=14, collide=True):
    """Tall coconut palm with a curved, tapering trunk and pinnate fronds (photos 1, 5, 13)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    lean = np.asarray(lean, float)
    n = 14
    pts = []
    for i in range(n + 1):
        t = i / n
        off = lean * (t ** 1.6) * height * 0.18
        pts.append(base + np.array([off[0], off[1], t * height]))
    for i in range(n):
        r0 = 0.2 * (1 - 0.35 * i / n)
        r1 = 0.2 * (1 - 0.35 * (i + 1) / n)
        p0, p1 = pts[i], pts[i + 1]
        d = p1 - p0
        L = np.linalg.norm(d)
        z = d / L
        x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.999 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x) + 1e-12
        R = np.c_[x, np.cross(z, x), z]
        s.cylinder("palm_bark", (p0 + p1) / 2, r0, L / 2 + 0.01, R=R, r_top=r1, collide=False, seg=12, caps=False)
    if collide:
        s.collider("cylinder", base + np.array([0, 0, 1.2]), (0.22, 1.2))
    top = pts[-1]
    s.sphere("palm_bark", top, 0.28, scale=(1, 1, 0.8))
    for k in range(5):                      # coconuts
        a = 2 * math.pi * k / 5
        s.sphere("leaf_dark", top + np.array([0.22 * math.cos(a), 0.22 * math.sin(a), -0.25]), 0.11)
    allV, allF = [], []
    cnt = 0
    for k in range(fronds):
        az = 2 * math.pi * k / fronds + rng.uniform(-0.15, 0.15)
        el = rng.uniform(-0.25, 0.8)
        Lf = rng.uniform(3.4, 4.6)
        d0 = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        # rachis: arching under its own weight
        m = 28
        rach = [top + d0 * Lf * (i / m) - np.array([0, 0, 0.55 * Lf * (i / m) ** 2]) for i in range(m + 1)]
        mats_leaflets = []
        for i in range(2, m):
            a, b = rach[i], rach[i + 1]
            d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
            side = np.cross(d, [0, 0, 1])
            side /= np.linalg.norm(side) + 1e-12
            t = i / m
            ll = 0.95 * math.sin(math.pi * min(1.0, t * 1.08) ** 0.75) + 0.12      # leaflet length along the frond
            for sg in (-1, 1):
                # narrow drooping leaflet: root on the rachis, pointing sideways/forward/down
                dirv = side * sg * 0.85 + d * 0.45 + np.array([0, 0, -0.55 + 0.25 * rng.uniform(-1, 1)])
                dirv /= np.linalg.norm(dirv)
                nrm = np.cross(dirv, d)
                nrm /= np.linalg.norm(nrm) + 1e-12
                w = 0.045
                p0 = a + rng.normal(0, 0.01, 3)
                pm = p0 + dirv * ll * 0.5 - np.array([0, 0, 0.05 * ll])
                p1 = p0 + dirv * ll - np.array([0, 0, 0.18 * ll])
                allV.append(np.array([p0 - d * w * 0.5, p0 + d * w * 0.5, pm + d * w, pm - d * w, p1]))
                allF += [(cnt, cnt + 1, cnt + 2), (cnt, cnt + 2, cnt + 3), (cnt + 3, cnt + 2, cnt + 4)]
                cnt += 5
        for i in range(m):                     # rachis stem
            a, b = rach[i], rach[i + 1]
            r0 = 0.035 * (1 - i / m) + 0.008
            allV.append(np.array([a + [0, 0, r0], b + [0, 0, r0 * 0.8], b - [0, 0, r0 * 0.8], a - [0, 0, r0]]))
            allF += [(cnt, cnt + 1, cnt + 2), (cnt, cnt + 2, cnt + 3)]
            cnt += 4
    _add_double(s, "leaf_palm", np.concatenate(allV), allF)


def areca_clump(s, base, height=3.5, stems=7, seed=0):
    """Clumping areca palm (photo 6 background, garden planters)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    for k in range(stems):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.05, 0.25)
        h = height * rng.uniform(0.6, 1.0)
        p0 = base + np.array([0.15 * math.cos(a), 0.15 * math.sin(a), 0])
        p1 = p0 + np.array([lean * h * math.cos(a), lean * h * math.sin(a), h])
        s.tube("bamboo", p0, p1, 0.035, seg=8)
        leaf_cluster(s, p1, 1.2, n=8, mat="leaf_light", seed=seed * 31 + k, leaf_len=1.4, droop=0.35)


def frangipani(s, base, height=4.0, seed=0, flowers=True):
    """Frangipani (plumeria) with forked grey branches, leaf rosettes and pink flowers (photo 11)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    s.collider("cylinder", base + np.array([0, 0, 0.8]), (0.12, 0.8))

    def branch(p, d, L, r, depth):
        q = p + d * L
        s.tube("tree_bark", p, q, r, seg=8)
        if depth == 0:
            leaf_cluster(s, q, 0.55, n=10, mat="leaf_dark", seed=int(rng.integers(1e6)), leaf_len=0.45, droop=0.2)
            if flowers:
                for _ in range(4):
                    s.sphere("flower_pink" if rng.random() < 0.7 else "flower_white",
                             q + rng.normal(0, 0.12, 3) + [0, 0, 0.1], 0.05)
            return
        for _ in range(2 if depth > 1 else 3):
            nd = d + rng.normal(0, 0.45, 3)
            nd[2] = abs(nd[2]) + 0.4
            nd /= np.linalg.norm(nd)
            branch(q, nd, L * 0.75, r * 0.7, depth - 1)

    branch(base, np.array([0, 0, 1.0]), height * 0.4, 0.11, 3)


def broadleaf_tree(s, base, height=6.0, crown=2.5, seed=0, staked=False):
    """Generic tropical tree with a rounded crown of leaf clusters; young trees can have timber stakes (photo 8)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    top = base + np.array([rng.normal(0, 0.2), rng.normal(0, 0.2), height * 0.55])
    s.tube("tree_bark", base, top, 0.09 if staked else 0.18, seg=10)
    s.collider("cylinder", base + np.array([0, 0, 1.0]), (0.12 if staked else 0.2, 1.0))
    nb = 7 if staked else 11
    for k in range(nb):
        a = 2 * math.pi * k / nb + rng.uniform(-0.3, 0.3)
        rr = crown * rng.uniform(0.35, 0.85)
        c = top + np.array([rr * math.cos(a), rr * math.sin(a), rng.uniform(0.2, height * 0.4)])
        s.tube("tree_bark", top, c, 0.05 if not staked else 0.03, seg=6)
        low = LOD["level"] == "low"
        leaf_cluster(s, c, crown * 0.6, n=(18 if low else 34) if not staked else 16,
                     mat="leaf_dark" if k % 3 else "leaf_light", seed=int(rng.integers(1e6)),
                     leaf_len=(0.75 if low else 0.55) if not staked else 0.4, droop=0.35, elong=0.7)
    leaf_cluster(s, top + np.array([0, 0, height * 0.42]), crown * 0.7, n=30 if not staked else 16,
                 mat="leaf_dark", seed=int(rng.integers(1e6)), leaf_len=0.55 if not staked else 0.35, droop=0.3,
                 elong=0.7)
    if staked:
        for k in range(3):
            a = 2 * math.pi * k / 3
            foot = base + np.array([0.9 * math.cos(a), 0.9 * math.sin(a), 0])
            s.tube("wood_light", foot, base + np.array([0, 0, 1.8]), 0.03, seg=6)


def ginger_clump(s, base, radius=0.8, n=9, height=1.6, seed=0, mat="leaf_variegated"):
    """Variegated shell ginger / heliconia: upright stems with big drooping leaves (photos 6, 7, 15)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    allV, allF = [], []
    for k in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(rng.uniform(0, 1)) * 0.6
        p0 = base + np.array([r * math.cos(a), r * math.sin(a), 0])
        h = height * rng.uniform(0.6, 1.0)
        top = p0 + np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), h])
        s.tube("leaf_dark", p0, top, 0.02, seg=5)
        for j in range(5):
            t = 0.35 + 0.65 * j / 5
            p = p0 + (top - p0) * t
            az = rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(az), math.sin(az), rng.uniform(0.2, 0.9)])
            v, f = _leaf(rng.uniform(0.45, 0.75), rng.uniform(0.16, 0.24), rng.uniform(0.3, 0.8), n=6)
            R = _frame_from_dir(d, rng.uniform(-0.5, 0.5))
            off = sum(len(x) for x in allV)
            allV.append(v @ R.T + p)
            allF += [(i + off, j_ + off, l + off) for i, j_, l in f]
    _add_double(s, mat, np.concatenate(allV), allF)


def grass_tuft(s, base, radius=0.35, blades=26, height=0.7, seed=0, mat="leaf_light"):
    """Ornamental grass (photo 7 planters)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    V, F = [], []
    for k in range(blades):
        a = rng.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(rng.uniform(0, 1)) * 0.4
        p = base + np.array([r * math.cos(a), r * math.sin(a), 0])
        tilt = rng.uniform(0.15, 0.6)
        h = height * rng.uniform(0.6, 1.0)
        tip = p + np.array([tilt * h * math.cos(a), tilt * h * math.sin(a), h])
        mid = (p + tip) / 2 + np.array([0, 0, 0.05])
        side = np.array([-math.sin(a), math.cos(a), 0]) * 0.012
        o = len(V)
        V += [p - side, p + side, mid + side * 0.8, mid - side * 0.8, tip]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3), (o + 3, o + 2, o + 4)]
    _add_double(s, mat, np.array(V), F)


def bamboo_clump(s, base, height=6.0, culms=14, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    for k in range(culms):
        p0 = base + np.array([rng.normal(0, 0.25), rng.normal(0, 0.25), 0])
        lean = rng.normal(0, 0.06, 2)
        h = height * rng.uniform(0.7, 1.0)
        p1 = p0 + np.array([lean[0] * h, lean[1] * h, h])
        s.tube("bamboo", p0, p1, 0.03, seg=6)
        for j in range(3):
            leaf_cluster(s, p0 + (p1 - p0) * (0.6 + 0.15 * j), 0.6, n=8, mat="leaf_light",
                         seed=int(rng.integers(1e6)), leaf_len=0.3, droop=0.3)
    s.collider("cylinder", base + np.array([0, 0, 1.0]), (0.4, 1.0))


def boulder(s, centre, r, seed=0, flat=0.6, collide=True):
    """Irregular granite boulder: displaced sphere (shore rocks of the Angra coast)."""
    from geom import sphere_mesh
    rng = np.random.default_rng(seed)
    v, n, t, f = sphere_mesh(1.0, 14, 9)
    k = rng.normal(0, 1, (6, 3))
    disp = 1 + 0.18 * np.sin(v @ k[0] * 2.1 + k[1, 0]) + 0.1 * np.sin(v @ k[2] * 4.3 + k[3, 0]) + \
        0.05 * np.sin(v @ k[4] * 8.0)
    sc = np.array([rng.uniform(0.8, 1.3), rng.uniform(0.8, 1.2), flat])
    vv = v * disp[:, None] * sc * r
    ang = rng.uniform(0, 2 * math.pi)
    c, sn = math.cos(ang), math.sin(ang)
    R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    s.trimesh("rock", vv @ R.T + np.asarray(centre, float), f, smooth=True)
    if collide:
        s.collider("sphere", np.asarray(centre, float) - [0, 0, r * flat * 0.3], (r * flat * 1.0,),
                   friction=(1.0, 0.05, 0.005))


def forest_blob(s, base, size=3.0, seed=0):
    """Cheap understorey/canopy mass for distant slopes: a few large leaf clusters on short stems."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    for k in range(2 if LOD["level"] == "low" else 3):
        c = base + np.array([rng.normal(0, size * 0.3), rng.normal(0, size * 0.3), size * rng.uniform(0.3, 0.7)])
        leaf_cluster(s, c, size * 0.5, n=14 if LOD["level"] == "low" else 18,
                     mat=["leaf_dark", "leaf_light", "leaf_dark"][k],
                     seed=int(rng.integers(1e6)), leaf_len=size * 0.32, droop=0.3, elong=0.6)


def hedge(s, a, b, height=1.2, depth=0.7, seed=0, mat="leaf_dark"):
    """Dense planted hedge between floor points a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b[:2] - a[:2])
    n = max(1, int(L / 0.6))
    for i in range(n):
        p = a + (b - a) * (i + 0.5) / n
        leaf_cluster(s, p + np.array([0, 0, height * 0.55]), max(height, depth) * 0.55, n=26, mat=mat,
                     seed=seed * 101 + i, leaf_len=0.25, droop=0.2, elong=height / depth)
