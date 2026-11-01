"""Independent MuJoCo forward-kinematics checks for the Isaac reset pose."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from kinematics import nominal_pose
from preflight import check_assets


class ModelTests(unittest.TestCase):
    def test_asset_paths_and_inertias(self):
        self.assertEqual(check_assets()['errors'],[])

    def test_closed_nominal_pose_against_mujoco(self):
        import mujoco
        import numpy as np
        model=mujoco.MjModel.from_xml_path(str(HERE.parent/'models/cena_plana.xml'))
        for length in (0.18,0.27,0.36):
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
        for length in (0.1,0.4):
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


class TerrainTests(unittest.TestCase):
    def test_course_geometry_and_spawn_clearance(self):
        import numpy as np
        from primus_task.terrain import course_mesh, course_grid, SPAWN_Y
        x,y,z = course_grid(); vertices,faces = course_mesh()
        self.assertTrue(np.isfinite(vertices).all())
        self.assertGreaterEqual(faces.min(),0)
        self.assertLess(faces.max(),len(vertices))
        normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                           vertices[faces[:,2]]-vertices[faces[:,0]])
        self.assertTrue((normals[:,2]>0).all())
        self.assertAlmostEqual(z[:,(y>=1)&(y<=3)].max(),0.08)
        self.assertAlmostEqual(z[:,(y>=4)&(y<=6.5)].max(),0.075)
        self.assertLessEqual(abs(z[:,(y>=7)&(y<=10)]).max(),0.012)
        for start in SPAWN_Y:
            # Entire axle contact region must spawn on a flat pad.
            self.assertLess(abs(z[:,abs(y-start)<0.15]).max(),1e-8)

    def test_height_observation_matches_collision_triangles(self):
        import numpy as np
        import torch
        from primus_task.terrain import course_grid, course_mesh, sample_height
        vertices,faces = course_mesh()
        rng = np.random.default_rng(42)
        triangles = vertices[faces[rng.integers(len(faces),size=1000)]]
        weights = rng.dirichlet([1,1,1],size=1000)
        points = (triangles*weights[:,:,None]).sum(1)
        heights = sample_height(torch.tensor(points[:,0]),torch.tensor(points[:,1]),
                                torch.tensor(course_grid()[2])).numpy()
        np.testing.assert_allclose(heights,points[:,2],atol=2e-6)


if __name__=='__main__': unittest.main()
