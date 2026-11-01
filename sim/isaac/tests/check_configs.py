"""Runtime configuration regression checks. Run through launch.py after accepting the EULA."""
from pathlib import Path
import sys
import traceback
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ConfigurationTests(unittest.TestCase):
    def test_terrain_configs_construct_without_changing_balance_defaults(self):
        from primus_task.env_cfg import PrimusBalanceEnvCfg
        from primus_task.ppo_cfg import PrimusPPORunnerCfg
        from primus_task.terrain_cfg import PrimusTerrainEnvCfg, PrimusTerrainPPORunnerCfg
        terrain = PrimusTerrainEnvCfg()
        agent = PrimusTerrainPPORunnerCfg()
        terrain.validate()
        # RSL RL's deprecated optional fields intentionally remain MISSING;
        # its entrypoint migrates/serializes these rather than calling validate().
        self.assertEqual(agent.to_dict()['experiment_name'], 'primus_terrain')
        self.assertEqual(terrain.scene.num_envs, 128)
        self.assertEqual(terrain.scene.env_spacing, 16.0)
        self.assertEqual(terrain.observation_space, 39)
        self.assertEqual(agent.actor.hidden_dims, [256, 128, 128])
        self.assertEqual(agent.critic.hidden_dims, [256, 128, 128])
        self.assertEqual(PrimusBalanceEnvCfg().scene.num_envs, 32)
        self.assertEqual(PrimusPPORunnerCfg().actor.hidden_dims, [128, 128])
        terrain.scene.num_envs = 1
        agent.actor.hidden_dims[0] = 8
        self.assertEqual(PrimusTerrainEnvCfg().scene.num_envs, 128)
        self.assertEqual(PrimusTerrainPPORunnerCfg().actor.hidden_dims[0], 256)
        self.assertEqual(terrain.video_recorders, [])


if __name__ == '__main__':
    from isaaclab.app import AppLauncher
    from workspace import CACHE
    cache = CACHE/'config-check-kit'
    cache.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(headless=True, device='cpu', kit_args=f'--portable-root {cache}')
    failed = True
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ConfigurationTests))
        failed = not result.wasSuccessful()
    except Exception:
        traceback.print_exc()
    finally:
        launcher.app.close(exit_code=int(failed))
