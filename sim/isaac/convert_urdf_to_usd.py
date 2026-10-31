"""Converte urdf/robo_bipedal.urdf -> usd/robo_bipedal.usd no Isaac Sim e fecha os lacos das pernas.

Rodar com o Python do Isaac Sim (4.5 ou 5.x), a partir da pasta isaac/:
    <ISAAC_SIM>/python.sh convert_urdf_to_usd.py
(no Windows: python.bat)

NAO TESTADO numa instalacao do Isaac Sim (foi escrito sem acesso a ela). Se algum nome de opcao do
importador mudar na sua versao, o script avisa e segue; o caminho pela interface esta no README.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, 'urdf', 'robo_bipedal.urdf')
OUTD = os.path.join(HERE, 'usd'); os.makedirs(OUTD, exist_ok=True)
USD = os.path.join(OUTD, 'robo_bipedal.usd')

from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
for ext in ('isaacsim.asset.importer.urdf', 'omni.importer.urdf'):
    try:
        enable_extension(ext); break
    except Exception:
        pass
try:
    from isaacsim.asset.importer.urdf import _urdf
except ImportError:
    from omni.importer.urdf import _urdf

ok, cfg = omni.kit.commands.execute('URDFCreateImportConfig')
def setcfg(name, value):
    try:
        setattr(cfg, name, value)
    except Exception as e:
        print(f'[aviso] opcao {name} nao existe nesta versao ({e})')
setcfg('merge_fixed_joints', False)
setcfg('fix_base', False)               # base flutuante
setcfg('import_inertia_tensor', True)   # usa massas/inercias do CAD
setcfg('distance_scale', 1.0)           # URDF ja esta em metros
setcfg('self_collision', False)
setcfg('collision_from_visuals', False) # usa as cascas convexas de colisao
setcfg('convex_decomp', False)
setcfg('make_default_prim', True)
setcfg('create_physics_scene', True)
try:
    setcfg('default_drive_type', _urdf.UrdfJointTargetType.JOINT_DRIVE_NONE)  # torque direto (RL define os atuadores)
except AttributeError:
    pass

omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=URDF, import_config=cfg, dest_path=USD)
print('USD gerado:', USD)

sys.path.insert(0, HERE)
from add_loop_closures import main as close_loops
close_loops(USD, os.path.join(HERE, '..', 'models', 'cad', 'loop_closures.json'))
app.close()
