"""Fecha o mecanismo de 5 barras de cada perna no USD gerado pelo importador de URDF do Isaac Sim.

Cria, para cada perna, uma junta de revolucao (eixo X) entre linkYm_* e shin_* no eixo da roda,
marcada com excludeFromArticulation=True (o PhysX resolve a articulacao em arvore + esta junta extra
como restricao de laco).

Uso (Python do Isaac Sim, ou qualquer Python com usd-core):
    ./python.sh add_loop_closures.py usd/robo_bipedal.usd ../models/cad/loop_closures.json
O arquivo e alterado no lugar (faz backup .bak).
"""
import json, shutil, sys
from pxr import Usd, UsdPhysics, Sdf, Gf

def find_link(stage, name):
    hits = [p for p in stage.Traverse() if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not hits:  # algumas versoes aplicam o RigidBodyAPI so depois; aceita Xform com o nome exato
        hits = [p for p in stage.Traverse() if p.GetName() == name and p.IsA(Usd.Typed)]
    if len(hits) != 1:
        raise RuntimeError(f'esperava 1 prim "{name}", achei {len(hits)}: {[str(h.GetPath()) for h in hits]}')
    return hits[0]

def main(usd_path, loops_path):
    shutil.copy(usd_path, usd_path + '.bak')
    stage = Usd.Stage.Open(usd_path)
    loops = json.load(open(loops_path))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot().GetChildren()[0]
    parent = root.GetPath().AppendChild('loop_closures')
    stage.DefinePrim(parent, 'Scope')
    for L in loops:
        b0 = find_link(stage, L['body0']); b1 = find_link(stage, L['body1'])
        j = UsdPhysics.RevoluteJoint.Define(stage, parent.AppendChild(L['name']))
        j.CreateBody0Rel().SetTargets([b0.GetPath()])
        j.CreateBody1Rel().SetTargets([b1.GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*L['local_pos0']))
        j.CreateLocalPos1Attr(Gf.Vec3f(*L['local_pos1']))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr(L['axis'])
        j.CreateExcludeFromArticulationAttr(True)
        j.CreateCollisionEnabledAttr(False)
        print('criada', j.GetPath(), ':', b0.GetPath(), '<->', b1.GetPath())
    stage.GetRootLayer().Save()
    print('ok ->', usd_path)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'loop_closures.json')
