"""Mixed-course locomotion task, separate from the 22-observation balance task."""
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg
from .env_cfg import PrimusBalanceEnvCfg
from .ppo_cfg import PrimusPPORunnerCfg


@configclass
class PrimusTerrainEnvCfg(PrimusBalanceEnvCfg):
    observation_space = 39  # 22 proprioception + 2 commands + 15 ideal height samples
    state_space = 42
    episode_length_s = 20.0
    curriculum = True
    evaluation_stage = 0
    scene = PrimusBalanceEnvCfg().scene.replace(num_envs=128, env_spacing=16.0)
    sim = PrimusBalanceEnvCfg().sim.replace(default_visualizer_cfg=KitVisualizerCfg(
        headless=True, eye=(2.8,-3.5,2.0), lookat=(0,0.5,0.25),
        origin_type='asset', origin_track_path='robot', origin_env_index=0,
        window_width=960, window_height=540))
    # The official CLI enables recording only when --video is passed.
    # Smoke tests and non-rendering evaluations must not create recorders.
    video_recorders = []


@configclass
class PrimusTerrainPPORunnerCfg(PrimusPPORunnerCfg):
    experiment_name = 'primus_terrain'
    max_iterations = 3000
    save_interval = 100
    num_steps_per_env = 64
    actor = PrimusPPORunnerCfg().actor.replace(hidden_dims=[256,128,128])
    critic = PrimusPPORunnerCfg().critic.replace(hidden_dims=[256,128,128])
