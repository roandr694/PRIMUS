"""Teste rapido no Isaac Sim: carrega o USD, solta o robo no chao por 3 s sem torque e mede
se o laco de cada perna continua fechado (distancia entre os dois lados da junta do pe).
Rodar: <ISAAC_SIM>/python.sh check_isaac.py   (NAO TESTADO - escrito sem acesso ao Isaac Sim)"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World
world = World(stage_units_in_meters=1.0, physics_dt=1 / 500)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
prim = stage.DefinePrim('/World/robo', 'Xform')
prim.GetReferences().AddReference(os.path.join(HERE, 'usd', 'robo_bipedal.usd'))
UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(0, 0, 0.52))
loops = json.load(open(os.path.join(HERE, '..', 'models', 'cad', 'loop_closures.json')))
def find(n): return [p for p in stage.Traverse() if p.GetName() == n and str(p.GetPath()).startswith('/World/robo')][0]
world.reset()
for k in range(1500):
    world.step(render=False)
    if k % 250 == 0:
        cache = UsdGeom.XformCache()
        out = []
        for L in loops:
            a = cache.GetLocalToWorldTransform(find(L['body0'])).Transform(Gf.Vec3d(*L['local_pos0']))
            b = cache.GetLocalToWorldTransform(find(L['body1'])).Transform(Gf.Vec3d(*L['local_pos1']))
            out.append('%s gap=%.2f mm' % (L['name'], (a - b).GetLength() * 1000))
        print('t=%.1fs' % (k / 500), ' | '.join(out))
app.close()
