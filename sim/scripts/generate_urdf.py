"""Gera o URDF (para Isaac Sim) com as mesmas malhas/massas do MJCF.
URDF nao representa cadeias fechadas: o laco de cada perna (link Y- <-> canela, no eixo da roda)
fica descrito em loop_closures.json e e criado no USD pelo script isaac/add_loop_closures.py."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from robot_def import LEGS, GEAR, link_origin
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'isaac'))
from actuator_parameters import GEAR_RATIO, JOINT_TORQUE_LIMIT, JOINT_NO_LOAD_SPEED
CAD = os.path.join(ROOT, 'models', 'cad')
OUT = os.path.join(ROOT, 'isaac', 'urdf')
MESH = '../../models/meshes'   # relativo ao URDF
I = json.load(open(f'{CAD}/mass_properties.json'))['inertial']
V = json.load(open(f'{CAD}/visual_meshes.json'))
C = json.load(open(f'{CAD}/collision.json'))
HIP_EFFORT = JOINT_TORQUE_LIMIT   # Unmeasured hypothesis, joint-side N.m
WHEEL_EFFORT = JOINT_TORQUE_LIMIT
VEL = JOINT_NO_LOAD_SPEED        # Ideal 24 V / 130 kV / 9:1 reference [rad/s]
def f(v): return ' '.join('%.6g' % x for x in v)
def rel(c, p): return (np.array(link_origin(c)) - np.array(link_origin(p))) / 1000.0
mats = {}
def link(name):
    d = I[name]; xx, yy, zz, xy, xz, yz = d['fullinertia']
    s = [f'  <link name="{name}">',
         f'    <inertial><origin xyz="{f(d["com_local"])}" rpy="0 0 0"/><mass value="{d["mass"]:.6g}"/>'
         f'<inertia ixx="{xx:.6g}" iyy="{yy:.6g}" izz="{zz:.6g}" ixy="{xy:.6g}" ixz="{xz:.6g}" iyz="{yz:.6g}"/></inertial>']
    for v in V[name]:
        key = 'c_' + '_'.join('%02d' % round(x * 100) for x in v['rgba'][:3])
        mats[key] = v['rgba']
        s.append(f'    <visual><geometry><mesh filename="{MESH}/visual/{v["mesh"]}.stl"/></geometry><material name="{key}"/></visual>')
    c = C.get(name)
    if c and c['type'] == 'cylinder':
        xc = (c['x0'] + c['x1']) / 2; L = c['x1'] - c['x0']
        s.append(f'    <collision name="{name}_tire"><origin xyz="{xc:.5f} 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="{c["radius"]:.5f}" length="{L:.5f}"/></geometry></collision>')
    elif c:
        for m in c['meshes']:
            s.append(f'    <collision><geometry><mesh filename="{MESH}/collision/{m}.stl"/></geometry></collision>')
    s.append('  </link>')
    return '\n'.join(s)
def joint(name, typ, parent, child, extra=''):
    return (f'  <joint name="{name}" type="{typ}"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{f(rel(child, parent))}" rpy="0 0 0"/><axis xyz="1 0 0"/>{extra}</joint>')
links = ['base']; joints = []
for s in LEGS:
    links += [f'armYp_{s}', f'shin_{s}', f'wheel_{s}', f'armYm_{s}', f'linkYm_{s}']
    hip_lim = f'<limit lower="-3.1416" upper="3.1416" effort="{HIP_EFFORT}" velocity="{VEL}"/><dynamics damping="0.05" friction="0"/>'
    joints.append(joint(f'hip_Yp_{s}', 'revolute', 'base', f'armYp_{s}', hip_lim))
    joints.append(joint(f'knee_Yp_{s}', 'continuous', f'armYp_{s}', f'shin_{s}', '<dynamics damping="0.01"/>'))
    joints.append(joint(f'wheel_{s}', 'continuous', f'shin_{s}', f'wheel_{s}', f'<limit effort="{WHEEL_EFFORT}" velocity="{VEL}"/><dynamics damping="0.01"/>'))
    joints.append(joint(f'hip_Ym_{s}', 'revolute', 'base', f'armYm_{s}', hip_lim))
    joints.append(joint(f'knee_Ym_{s}', 'continuous', f'armYm_{s}', f'linkYm_{s}', '<dynamics damping="0.01"/>'))
for g, d in GEAR.items():
    links.append(g); lo, hi = np.radians(d['range_deg'])
    joints.append(joint(g, 'revolute', 'base', g, f'<limit lower="{lo:.4f}" upper="{hi:.4f}" effort="0.25" velocity="10"/><dynamics damping="0.02"/>'))
body = '\n'.join(link(n) for n in links)
matx = '\n'.join(f'  <material name="{k}"><color rgba="{f(v)}"/></material>' for k, v in mats.items())
urdf = f'''<?xml version="1.0"?>
<!-- Robo bipedal com rodas - gerado da montagem principal do Fusion. X lateral, Y frente, Z cima. SI.
     ATENCAO: cada perna e um 5 barras FECHADO. O URDF e uma arvore; o fechamento (linkYm_* <-> shin_* no eixo da roda)
     esta em loop_closures.json e e criado no USD por isaac/add_loop_closures.py. -->
<robot name="robo_bipedal">
{matx}
{body}
{chr(10).join(joints)}
</robot>
'''
open(f'{OUT}/robo_bipedal.urdf', 'w').write(urdf)
loops = []
for s, L in LEGS.items():
    foot = np.array(L['foot']); loops.append(dict(name=f'loop_{s}', body0=f'linkYm_{s}', body1=f'shin_{s}', type='revolute', axis='X',
        local_pos0=((foot - np.array(L['kneeYm'])) / 1000).tolist(), local_pos1=((foot - np.array(L['kneeYp'])) / 1000).tolist()))
json.dump(loops, open(f'{CAD}/loop_closures.json', 'w'), indent=1)
print('urdf ok', len(urdf), 'links', len(links), 'joints', len(joints))
