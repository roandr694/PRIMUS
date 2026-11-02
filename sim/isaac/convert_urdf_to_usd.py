"""Isaac Sim 6.1 conversion via Isaac Lab v3.0.0-EA; preserve imported layers."""
from pathlib import Path
from workspace import CACHE, STATE
import sys
import traceback


def main():
    from isaaclab.app import AppLauncher
    here = Path(__file__).resolve().parent
    cache = CACHE/'conversion-kit'
    cache.mkdir(parents=True, exist_ok=True)
    # Asset conversion does not require CUDA. Keep Kit's writable data outside the shared Sim.
    launcher = AppLauncher(headless=True, device='cpu', kit_args=f'--portable-root {cache}')
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdGeom
        from add_loop_closures import main as close_loops
        from preflight import check_assets
        errors = check_assets()['errors']
        if errors: raise RuntimeError('; '.join(errors))
        cfg=UrdfConverterCfg(asset_path=str(here/'urdf/robo_bipedal.urdf'),
            usd_dir=str(here/'usd/imported'),fix_base=False,merge_fixed_joints=False,
            force_usd_conversion=True,make_instanceable=False,
            collision_from_visuals=False,self_collision=False,collision_type='Convex Hull',
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0,damping=0.0)),
            run_multi_physics_conversion=False)
        converted=UrdfConverter(cfg)
        for receipt in STATE.glob('smoke-*.json'): receipt.unlink()  # assets changed: physics checks are stale
        output=here/'usd/robo_bipedal.usd'
        if output.exists():
            import shutil
            shutil.copy2(output,str(output)+'.previous.bak')
            output.unlink()
        stage=Usd.Stage.CreateNew(str(output))
        root=UsdGeom.Xform.Define(stage,'/robo_bipedal').GetPrim(); stage.SetDefaultPrim(root)
        import os
        root.GetReferences().AddReference(os.path.relpath(converted.usd_path,output.parent))
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1.0)
        stage.GetRootLayer().Save(); del stage
        close_loops(str(output),str(here.parent/'models/cad/loop_closures.json'))
        print('Converted with closed loops:',output)
    finally:
        failed = sys.exc_info()[0] is not None
        if failed: traceback.print_exc()
        launcher.app.close(exit_code=int(failed))


if __name__=='__main__': main()
