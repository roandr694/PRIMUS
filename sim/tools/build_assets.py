"""Gera malhas (visual + colisao) e propriedades de massa por corpo, no frame de cada corpo, em metros."""
import json, os, glob, collections, sys
import numpy as np, trimesh, fast_simplification, coacd
sys.path.insert(0, os.path.dirname(__file__))
from robot_def import link_origin, MASS_OVERRIDES

RAW = os.environ.get('RAW', 'robo_sim_export/raw')   # pasta com os STL + bodies.json exportados do Fusion
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(f'{OUT}/meshes/visual', exist_ok=True)
os.makedirs(f'{OUT}/meshes/collision', exist_ok=True); os.makedirs(f'{OUT}/cad', exist_ok=True)

# ---------------- massa ----------------
B = [b for b in json.load(open(f'{RAW}/bodies.json')) if 'err' not in b]
def match(b, f): return all(b.get(k) == v for k, v in f.items())
fac = np.ones(len(B)); notes = []
for f, mode, val, why in MASS_OVERRIDES:
    idx = [i for i, b in enumerate(B) if match(b, f)]
    m0 = sum(B[i]['mass'] for i in idx)
    s = val if mode == 'scale' else val / m0
    for i in idx: fac[i] *= s
    notes.append(dict(filtro=f, corpos=len(idx), massa_cad=round(m0, 4), massa_usada=round(m0 * s, 4), motivo=why))
links = collections.defaultdict(lambda: dict(m=0.0, mc=np.zeros(3), I=np.zeros((3, 3))))
for b, s in zip(B, fac):
    m = b['mass'] * s
    c = np.array(b['com']) / 100.0                          # cm -> m
    xx, yy, zz, xy, yz, xz = [v * s * 1e-4 for v in b['I']]   # kg cm^2 -> kg m^2 (sobre a origem do CAD)
    I0 = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    L = links[b['link']]
    L['m'] += m; L['mc'] += m * c; L['I'] += I0
inertial = {}
for name, L in links.items():
    m = L['m']; com = L['mc'] / m
    Ic = L['I'] - m * (com @ com * np.eye(3) - np.outer(com, com))   # sobre o CoM, eixos do mundo
    o = np.array(link_origin(name)) / 1000.0
    w, v = np.linalg.eigh(Ic)
    inertial[name] = dict(mass=m, com_local=(com - o).tolist(), fullinertia=[Ic[0, 0], Ic[1, 1], Ic[2, 2], Ic[0, 1], Ic[0, 2], Ic[1, 2]], principal=w.tolist())
json.dump(dict(inertial=inertial, mass_overrides=notes, total_mass=sum(v['mass'] for v in inertial.values())), open(f'{OUT}/cad/mass_properties.json', 'w'), indent=1)
print('massa total %.3f kg' % sum(v['mass'] for v in inertial.values()))
for k in sorted(inertial): print('  %-11s %.3f kg' % (k, inertial[k]['mass']))

# ---------------- malhas ----------------
BUDGET = dict(base=160000, shin=24000, wheel=14000, linkYm=10000, armYp=8000, armYm=8000, gear=4000)
def budget(link): return BUDGET.get(link.split('_')[0], 8000)
visual = collections.defaultdict(list)
per_link = collections.defaultdict(list)
files = sorted(glob.glob(f'{RAW}/*.stl'))
tot = {l: sum(trimesh.load(f, force='mesh').faces.shape[0] for f in files if os.path.basename(f).split('__')[0] == l) for l in set(os.path.basename(f).split('__')[0] for f in files)}
for f in files:
    link, col = os.path.basename(f)[:-4].split('__')
    rgb = [float(x.replace('p', '.')) for x in col.split('_')]
    m = trimesh.load(f, force='mesh', process=False)
    m.merge_vertices()
    m.vertices = (m.vertices - np.array(link_origin(link))) / 1000.0
    n = m.faces.shape[0]
    target = max(200, int(budget(link) * n / tot[link]))
    if n > target:
        v, fc = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32), target_reduction=1 - target / n)
        m = trimesh.Trimesh(v, fc, process=True)
    name = f'{link}__{col}'
    m.export(f'{OUT}/meshes/visual/{name}.stl')
    visual[link].append(dict(mesh=name, rgba=rgb + [1.0], faces=int(m.faces.shape[0])))
    per_link[link].append(m)
json.dump(visual, open(f'{OUT}/cad/visual_meshes.json', 'w'), indent=1)
print('visual faces:', {k: sum(x['faces'] for x in v) for k, v in visual.items()})

# ---------------- colisao ----------------
coll = {}
for link, ms in per_link.items():
    whole = trimesh.util.concatenate(ms)
    if link.startswith('wheel'):
        # pneu: cilindro coaxial com a junta da roda (eixo X)
        r = np.sqrt(whole.vertices[:, 1] ** 2 + whole.vertices[:, 2] ** 2).max()
        x0, x1 = whole.vertices[:, 0].min(), whole.vertices[:, 0].max()
        coll[link] = dict(type='cylinder', radius=float(r), x0=float(x0), x1=float(x1))
        continue
    if whole.faces.shape[0] > 30000:
        v, fc = fast_simplification.simplify(whole.vertices.astype(np.float32), whole.faces.astype(np.int32), target_reduction=1 - 30000 / whole.faces.shape[0])
        whole = trimesh.Trimesh(v, fc, process=True)
    thr = 0.06 if link == 'base' else 0.08
    parts = coacd.run_coacd(coacd.Mesh(whole.vertices, whole.faces), threshold=thr, max_convex_hull=24 if link == 'base' else 6, preprocess_resolution=40)
    names = []
    for i, (v, fc) in enumerate(parts):
        h = trimesh.Trimesh(v, fc).convex_hull
        nm = f'{link}__hull{i:02d}'
        h.export(f'{OUT}/meshes/collision/{nm}.stl'); names.append(nm)
    coll[link] = dict(type='hulls', meshes=names)
    print(link, 'hulls', len(names))
json.dump(coll, open(f'{OUT}/cad/collision.json', 'w'), indent=1)
