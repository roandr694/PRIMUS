"""Owner-reported actuator data and explicit, unmeasured simulation hypotheses."""
import json
import math
from pathlib import Path

PROFILE = json.loads((Path(__file__).resolve().parents[2] / 'hardware/actuators.json').read_text())
REPORTED = PROFILE['reported_by_owner']
HYPOTHESES = PROFILE['simulation_hypotheses_not_measured']
GEAR_RATIO = PROFILE['existing_project_geometry']['gear_ratio_motor_to_joint']
SUPPLY_VOLTAGE = REPORTED['supply_voltage_v']
MOTOR_KV = REPORTED['motor_kv_rpm_per_v']
# Ideal voltage-limited reference, without field weakening or voltage drop.
JOINT_NO_LOAD_SPEED = SUPPLY_VOLTAGE * MOTOR_KV * 2.0 * math.pi / (60.0 * GEAR_RATIO)
JOINT_TORQUE_LIMIT = HYPOTHESES['joint_torque_limit_nm']
JOINT_STALL_TORQUE = HYPOTHESES['joint_stall_torque_nm']
ROTOR_ARMATURE = HYPOTHESES['reflected_rotor_inertia_kg_m2']
