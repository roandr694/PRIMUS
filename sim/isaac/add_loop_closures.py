"""Closes the 5-bar mechanism of each leg in the USD made by Isaac Sim's URDF importer.

For each leg it creates a revolute joint (X axis) between linkYm_* and shin_* at the wheel axle, marked
excludeFromArticulation=True (PhysX solves the tree articulation + this extra joint as a loop constraint).

Usage (Isaac Sim Python, or any Python with usd-core):
    ./python.sh add_loop_closures.py usd/robo_bipedal.usd ../models/cad/loop_closures.json
The file is edited in place (a .bak backup is made).
"""
import json, shutil, sys
from pxr import Usd, UsdPhysics, Sdf, Gf

def find_link(stage, name):
    hits = [p for p in stage.Traverse() if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not hits:  # some versions apply RigidBodyAPI later; accept an Xform with the exact name
        hits = [p for p in stage.Traverse() if p.GetName() == name and p.IsA(Usd.Typed)]
    if len(hits) != 1:
        raise RuntimeError(f'expected 1 prim "{name}", found {len(hits)}: {[str(h.GetPath()) for h in hits]}')
    return hits[0]

def main(usd_path, loops_path):
    shutil.copy(usd_path, usd_path + '.bak')
    stage = Usd.Stage.Open(usd_path)
    with open(loops_path) as stream:
        loops = json.load(stream)
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
        print('created', j.GetPath(), ':', b0.GetPath(), '<->', b1.GetPath())
    stage.GetRootLayer().Save()
    print('ok ->', usd_path)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'loop_closures.json')
