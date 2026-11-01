"""Conservative small PPO baseline for RSL RL 5.4.1."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PrimusPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 20
    save_interval = 10
    experiment_name = 'primus_balance'
    obs_groups = {'actor':['policy'],'critic':['critic']}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(hidden_dims=[128,128],activation='elu',obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3))
    critic = RslRlMLPModelCfg(hidden_dims=[128,128],activation='elu',obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0,use_clipped_value_loss=True,
        clip_param=0.2,entropy_coef=0.005,num_learning_epochs=4,num_mini_batches=4,
        learning_rate=3e-4,schedule='adaptive',gamma=0.99,lam=0.95,desired_kl=0.01,max_grad_norm=1.0)
