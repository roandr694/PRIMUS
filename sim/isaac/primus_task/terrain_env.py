"""Torque-controlled mixed-course PPO environment with success-based start progression."""
import torch
from pxr import UsdGeom, UsdPhysics, Gf, Vt
from isaaclab.utils.math import quat_apply
from .env import PrimusBalanceEnv
from .terrain import course_grid, course_mesh, sample_height, SPAWN_Y


class PrimusTerrainEnv(PrimusBalanceEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.height_grid = torch.tensor(course_grid()[2], dtype=torch.float32, device=self.device)
        self.commands = torch.zeros((self.num_envs,2), device=self.device)
        self.stages = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_y = torch.zeros(self.num_envs, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.scan_offsets = torch.tensor([(x,y,0) for x in (-0.3,0,0.3)
                           for y in (-0.2,0,0.3,0.6,0.9)],device=self.device)

    def _spawn_terrain(self):
        points, faces = course_mesh()
        mesh = UsdGeom.Mesh.Define(self.scene.stage, '/World/envs/env_0/Terrain')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.32,0.42,0.36)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        return []

    def terrain_height(self, xy):
        return sample_height(xy[...,0], xy[...,1], self.height_grid)

    def _local_position(self):
        return self.robot.data.root_link_pos_w.torch-self.scene.env_origins

    def _get_observations(self):
        base = super()._get_observations()['policy']
        pose = self.robot.data.root_link_pose_w.torch
        # Yaw-only scan frame; roll/pitch must not rotate height samples into the ground.
        q = pose[:,3:7].clone(); q[:,:2] = 0
        q = q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-6)
        offsets = self.scan_offsets.expand(self.num_envs,-1,-1)
        world = quat_apply(q[:,None,:].expand(-1,15,-1).reshape(-1,4),offsets.reshape(-1,3)).reshape(-1,15,3)
        p = self._local_position()
        heights = self.terrain_height((p[:,None,:]+world)[...,:2])
        scan = (p[:,None,2]-self.cfg.base_height-heights).clamp(-0.3,0.3)*3
        actor = torch.cat((base,self.commands,scan),-1)
        return {'policy':actor,'critic':torch.cat((actor,self.robot.data.root_link_lin_vel_b.torch),-1)}

    def _get_rewards(self):
        d = self.robot.data
        p = self._local_position()
        height = p[:,2]-self.terrain_height(p[:,:2])
        speed_error = (d.root_link_lin_vel_b.torch[:,1]-self.commands[:,0]).square()
        yaw_error = (d.root_link_ang_vel_b.torch[:,2]-self.commands[:,1]).square()
        upright = (-d.projected_gravity_b.torch[:,2]).clamp(0,1)
        reward = (1.0+1.5*upright+2*torch.exp(-speed_error/0.08)
                  +0.5*torch.exp(-yaw_error/0.15)+torch.exp(-((height-self.cfg.base_height)/0.06)**2)
                  -0.3*d.root_link_lin_vel_b.torch[:,0].square()
                  -0.1*d.root_link_ang_vel_b.torch[:,:2].square().sum(-1)
                  -0.01*self.actions.square().sum(-1)
                  -0.05*(self.actions-self.last_actions).square().sum(-1))
        self.extras.setdefault('log',{}).update({
            'Terrain/stage':self.stages.float().mean().detach(),
            'Tracking/forward_error_mps':speed_error.sqrt().mean().detach(),
            'Terrain/progress_m':(p[:,1]-self.start_y).mean().detach()})
        return reward*self.step_dt-2.0*self.reset_terminated.float()+2.0*self.completed.float()

    def _get_dones(self):
        d = self.robot.data
        p = self._local_position()
        gap = self.loop_gaps().max(-1).values
        finite = torch.isfinite(d.root_link_state_w.torch).all(-1)&torch.isfinite(d.joint_pos.torch).all(-1)&torch.isfinite(gap)
        relative_height = p[:,2]-self.terrain_height(p[:,:2])
        fallen = ((relative_height<0.20)|(d.projected_gravity_b.torch[:,2]>-0.5)
                  |(~finite)|(gap>self.cfg.max_loop_gap)|(p[:,0].abs()>1.15)|(p[:,1]<-1.5))
        self.completed.copy_(p[:,1]>10.5)
        timeout = (self.episode_length_buf>=self.max_episode_length)|self.completed
        self.extras.setdefault('log',{})['Loop/max_gap_m'] = gap.max().detach()
        self.extras['log']['Physics/finite'] = finite.all().detach()
        return fallen, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None: env_ids = self.robot._ALL_INDICES
        ids = env_ids.long()
        p = self._local_position()
        if self.cfg.curriculum:
            promote = self.reset_time_outs[ids] & (~self.reset_terminated[ids]) & (p[ids,1]-self.start_y[ids]>0.5)
            demote = self.reset_terminated[ids] & (self.episode_length_buf[ids]<200)
            self.stages[ids] = (self.stages[ids]+promote.long()-demote.long()).clamp(0,3)
        else:
            self.stages[ids] = self.cfg.evaluation_stage
        super()._reset_idx(env_ids)
        starts = torch.tensor(SPAWN_Y,device=self.device)[self.stages[ids]]
        self.start_y[ids] = starts
        pose = self.robot.data.default_root_pose.torch[ids].clone()
        pose[:,0] += 0.05*(2*torch.rand(len(ids),device=self.device)-1)
        pose[:,1] = starts
        pose[:,:3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.completed[ids] = False
        # 20% standing, otherwise slow forward motion; yaw stays zero on narrow course.
        speed = 0.15+0.30*torch.rand(len(ids),device=self.device)
        standing = torch.rand(len(ids),device=self.device)<0.2
        self.commands[ids,0] = torch.where(standing,0.0,speed)
        self.commands[ids,1] = 0.0
