"""Small flat-ground standing task; deliberately selects Isaac Sim PhysX."""
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_robo_cfg import ROBO_BIPEDAL_CFG, BASE_HEIGHT


@configclass
class PrimusBalanceEnvCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 10.0
    action_space = 6
    observation_space = 22
    state_space = 25
    sim = SimulationCfg(dt=0.002, render_interval=5, physics=PhysxCfg(), use_newton_actuators=False)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=3.0,
                                replicate_physics=False, clone_in_fabric=False)
    robot_cfg = ROBO_BIPEDAL_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    base_height = BASE_HEIGHT
    max_loop_gap = 0.002
