"""First standing experiment. Actor uses IMU-like and motor encoder signals.

The critic also receives simulated body velocity. Sensor noise, delays and
encoder-derived odometry must be validated before any sim-to-real transfer.
"""
import json
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply
from actuator_parameters import JOINT_TORQUE_LIMIT

MOTOR_NAMES = ['hip_Yp_xp','hip_Ym_xp','hip_Yp_xm','hip_Ym_xm','wheel_xp','wheel_xm']


class PrimusBalanceEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.motor_ids, _ = self.robot.find_joints(MOTOR_NAMES, preserve_order=True)
        self.actions = torch.zeros((self.num_envs,6),device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.loops = json.loads((Path(__file__).resolve().parents[2]/'models/cad/loop_closures.json').read_text())
        self.loop_bodies = [(self.robot.body_names.index(L['body0']),self.robot.body_names.index(L['body1'])) for L in self.loops]
        self.loop_anchors = [(torch.tensor(L['local_pos0'],device=self.device).expand(self.num_envs,-1),
                              torch.tensor(L['local_pos1'],device=self.device).expand(self.num_envs,-1)) for L in self.loops]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        global_paths = self._spawn_terrain()
        positions = cloner.grid_transforms(self.scene.num_envs,self.scene.cfg.env_spacing,device=self.device)[0]
        plan = cloner.clone_plan_from_env_0('/World/envs/env_0','/World/envs/env_{}',
            self.scene.num_envs,self.device,positions,global_paths=tuple(global_paths))
        cloner.replicate(plan,stage=self.scene.stage)
        self.scene.filter_collisions(global_prim_paths=global_paths)
        self.scene.articulations['robot'] = self.robot
        light = sim_utils.DistantLightCfg(intensity=2000.0)
        light.func('/World/light',light)

    def _spawn_terrain(self):
        ground = sim_utils.CuboidCfg(size=(200.0,200.0,0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0,dynamic_friction=1.0,restitution=0.0))
        ground.func('/World/ground',ground,translation=(0.0,0.0,-0.05))
        return ['/World/ground']

    def _pre_physics_step(self, actions):
        self.last_actions.copy_(self.actions)
        # Keep a normal tensor: evaluation may reset outside inference_mode.
        self.actions.copy_(actions.clamp(-1,1))

    def _apply_action(self):
        self.robot.set_joint_effort_target_index(target=JOINT_TORQUE_LIMIT*self.actions,joint_ids=self.motor_ids)

    def loop_gaps(self):
        pose = self.robot.data.body_link_pose_w.torch
        gaps = []
        for (a,b),(p,q) in zip(self.loop_bodies,self.loop_anchors):
            x = pose[:,a,:3]+quat_apply(pose[:,a,3:7],p)
            y = pose[:,b,:3]+quat_apply(pose[:,b,3:7],q)
            gaps.append(torch.linalg.vector_norm(x-y,dim=-1))
        return torch.stack(gaps,dim=-1)

    def _get_observations(self):
        d = self.robot.data
        q = d.joint_pos.torch[:,self.motor_ids[:4]]-d.default_joint_pos.torch[:,self.motor_ids[:4]]
        dq = d.joint_vel.torch[:,self.motor_ids]*0.05
        actor = torch.cat((d.projected_gravity_b.torch,d.root_link_ang_vel_b.torch*0.25,q,dq,self.actions),dim=-1)
        return {'policy':actor, 'critic':torch.cat((actor,d.root_link_lin_vel_b.torch),dim=-1)}

    def _get_rewards(self):
        d = self.robot.data
        upright = (-d.projected_gravity_b.torch[:,2]).clamp(0,1)
        height = d.root_link_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        xy = d.root_link_pos_w.torch[:,:2]-self.scene.env_origins[:,:2]
        reward = (1.0 + upright + torch.exp(-((height-self.cfg.base_height)/0.06)**2)
                  - 0.5*d.root_link_lin_vel_b.torch.square().sum(-1)
                  - 0.2*xy.square().sum(-1) - 0.1*d.root_link_ang_vel_b.torch.square().sum(-1)
                  - 0.01*self.actions.square().sum(-1) - 0.05*(self.actions-self.last_actions).square().sum(-1))
        return (reward-5.0*self.reset_terminated.float())*self.step_dt

    def _get_dones(self):
        d = self.robot.data
        height = d.root_link_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        gaps = self.loop_gaps()
        finite = torch.isfinite(d.joint_pos.torch).all(-1)&torch.isfinite(d.root_link_state_w.torch).all(-1)&torch.isfinite(gaps).all(-1)
        self.extras.setdefault('log',{})['Loop/max_gap_m'] = gaps.max().detach()
        self.extras['log']['Physics/finite'] = finite.all().detach()
        fallen = (height<0.20)|(d.projected_gravity_b.torch[:,2]>-0.5)|(~finite)|(gaps.max(-1).values>self.cfg.max_loop_gap)
        return fallen, self.episode_length_buf>=self.max_episode_length

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.extras.setdefault('log',{})['Metrics/survived'] = self.reset_time_outs[env_ids].float().mean()
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0; self.last_actions[env_ids] = 0
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(),env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self.robot.data.default_joint_pos.torch[env_ids].clone(),env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch[env_ids].clone(),env_ids=env_ids)
