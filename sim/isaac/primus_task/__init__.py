"""Lazy Gym registration; safe to import before starting Kit."""
import gymnasium as gym

gym.register(
    id='Primus-Balance-Direct', entry_point=__name__+'.env:PrimusBalanceEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point':__name__+'.env_cfg:PrimusBalanceEnvCfg',
            'rsl_rl_cfg_entry_point':__name__+'.ppo_cfg:PrimusPPORunnerCfg',
            'default_agent':'rsl_rl'},
)

gym.register(
    id='Primus-Terrain-Direct', entry_point=__name__+'.terrain_env:PrimusTerrainEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point':__name__+'.terrain_cfg:PrimusTerrainEnvCfg',
            'rsl_rl_cfg_entry_point':__name__+'.terrain_cfg:PrimusTerrainPPORunnerCfg',
            'default_agent':'rsl_rl'},
)
