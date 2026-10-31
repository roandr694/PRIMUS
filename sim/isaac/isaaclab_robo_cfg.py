"""Exemplo de ArticulationCfg para Isaac Lab (ponto de partida para RL; NAO TESTADO).
Os limites de torque/velocidade sao PLACEHOLDERS ate voce passar os numeros reais dos motores."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

GEAR = 9.0
MOTOR_PEAK_TAU = 2.0      # N.m no motor - PLACEHOLDER
MOTOR_MAX_SPEED = 30.0    # rad/s na junta - PLACEHOLDER

ROBO_BIPEDAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="isaac/usd/robo_bipedal.usd",   # gerado por isaac/convert_urdf_to_usd.py (com os lacos fechados)
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=2),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.52), joint_pos={".*": 0.0}),
    actuators={
        "quadril": DCMotorCfg(joint_names_expr=["hip_.*"], saturation_effort=MOTOR_PEAK_TAU * GEAR,
                              effort_limit=MOTOR_PEAK_TAU * GEAR, velocity_limit=MOTOR_MAX_SPEED,
                              stiffness=0.0, damping=0.0, armature=0.006),
        "rodas": DCMotorCfg(joint_names_expr=["wheel_.*"], saturation_effort=MOTOR_PEAK_TAU * GEAR,
                            effort_limit=MOTOR_PEAK_TAU * GEAR, velocity_limit=2 * MOTOR_MAX_SPEED,
                            stiffness=0.0, damping=0.0, armature=0.006),
        "trem": ImplicitActuatorCfg(joint_names_expr=["gear_.*"], effort_limit=0.25, velocity_limit=10.0,
                                    stiffness=3.0, damping=0.05),
        # knee_* sao passivas (sem atuador) - o laco do 5 barras e fechado pela junta extra no USD
        "passivas": ImplicitActuatorCfg(joint_names_expr=["knee_.*"], effort_limit=0.0, stiffness=0.0, damping=0.01),
    },
)
