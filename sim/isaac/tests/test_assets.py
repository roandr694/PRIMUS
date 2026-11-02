"""Asset checks for the Isaac import: URDF paths/inertias, reset pose against MuJoCo, USD loop joints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'control'))
from five_bar import nominal_pose
from preflight import check_assets


class ModelTests(unittest.TestCase):
    def test_asset_paths_and_inertias(self):
        self.assertEqual(check_assets()['errors'],[])

    def test_closed_nominal_pose_against_mujoco(self):
        import mujoco
        import numpy as np
        model=mujoco.MjModel.from_xml_path(str(HERE.parent/'models/cena_plana.xml'))
        for length in (0.18,0.27,0.52):
            with self.subTest(length=length):
                q,z=nominal_pose(length); data=mujoco.MjData(model)
                data.qpos[:]=model.qpos0; data.qpos[:3]=(0,0,z); data.qpos[3:7]=(1,0,0,0)
                for name,value in q.items():
                    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                    self.assertGreaterEqual(jid,0); data.qpos[model.jnt_qposadr[jid]]=value
                mujoco.mj_forward(model,data)
                for side in ('xp','xm'):
                    point=lambda n:data.site_xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,n)]
                    foot=point('foot_'+side); wheel=point('wheel_center_'+side); hip=point('hip_mid_'+side)
                    self.assertLess(float(np.linalg.norm(foot-wheel)),1e-8)
                    self.assertAlmostEqual(float(np.linalg.norm((wheel-hip)[1:])),length,places=7)
                    self.assertAlmostEqual(float(wheel[2]),0.1095,places=7)

    def test_invalid_leg_length_rejected(self):
        for length in (0.1,0.6):
            with self.assertRaises(ValueError): nominal_pose(length)

    def test_usd_loop_joint_authoring(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        from add_loop_closures import main
        loops=json.loads((HERE.parent/'models/cad/loop_closures.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'robot.usda'; stage=Usd.Stage.CreateNew(str(path))
            root=UsdGeom.Xform.Define(stage,'/Robot').GetPrim(); stage.SetDefaultPrim(root)
            for L in loops:
                for name in (L['body0'],L['body1']):
                    prim=UsdGeom.Xform.Define(stage,'/Robot/'+name).GetPrim(); UsdPhysics.RigidBodyAPI.Apply(prim)
            stage.GetRootLayer().Save(); del stage
            main(str(path),str(HERE.parent/'models/cad/loop_closures.json'))
            stage=Usd.Stage.Open(str(path))
            for L in loops:
                joint=UsdPhysics.RevoluteJoint.Get(stage,'/Robot/loop_closures/'+L['name'])
                self.assertTrue(joint.GetExcludeFromArticulationAttr().Get())
                self.assertFalse(joint.GetCollisionEnabledAttr().Get())
                self.assertEqual(joint.GetAxisAttr().Get(),'X')
                self.assertEqual(len(joint.GetBody0Rel().GetTargets()),1)
            self.assertTrue(Path(str(path)+'.bak').is_file())



if __name__ == '__main__':
    unittest.main()
