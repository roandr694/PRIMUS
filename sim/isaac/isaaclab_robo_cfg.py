"""PRIMUS asset for Isaac Sim 6.1 + Isaac Lab v3.0.0-EA (PhysX).

Efforts are joint torques [N.m], already including the 9:1 reduction.
Motor ratings and rotor inertia remain provisional until measured.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from kinematics import nominal_pose
from actuator_parameters import JOINT_NO_LOAD_SPEED, JOINT_TORQUE_LIMIT, JOINT_STALL_TORQUE, ROTOR_ARMATURE

JOINT_POS, BASE_HEIGHT = nominal_pose()
USD_PATH = Path(__file__).resolve().parent / 'usd' / 'robo_bipedal.usd'
ROBO_BIPEDAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASE_HEIGHT), joint_pos=JOINT_POS,
    ),
    actuators={
        'hips': DCMotorCfg(joint_names_expr=['hip_.*'], saturation_effort=JOINT_STALL_TORQUE,
            actuator_effort_limit=JOINT_TORQUE_LIMIT, actuator_velocity_limit=JOINT_NO_LOAD_SPEED,
            joint_effort_limit=JOINT_TORQUE_LIMIT, joint_velocity_limit=JOINT_NO_LOAD_SPEED,
            stiffness=0.0, damping=0.0, armature=ROTOR_ARMATURE),
        'wheels': DCMotorCfg(joint_names_expr=['wheel_.*'], saturation_effort=JOINT_STALL_TORQUE,
            actuator_effort_limit=JOINT_TORQUE_LIMIT, actuator_velocity_limit=JOINT_NO_LOAD_SPEED,
            joint_effort_limit=JOINT_TORQUE_LIMIT, joint_velocity_limit=JOINT_NO_LOAD_SPEED,
            stiffness=0.0, damping=0.0, armature=ROTOR_ARMATURE),
        'landing_gear': ImplicitActuatorCfg(joint_names_expr=['gear_.*'],
            joint_effort_limit=0.25, joint_velocity_limit=10.0, stiffness=3.0, damping=0.05),
        'passive_knees': ImplicitActuatorCfg(joint_names_expr=['knee_.*'],
            joint_effort_limit=0.1, stiffness=0.0, damping=0.01),
    },
)
