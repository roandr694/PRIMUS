"""Generates the MuJoCo models of the robot from the CAD export.

Outputs (models/):
    robo_bipedal.xml   the robot only (included by the scenes)
    cena_arena.xml     "arena" scene: terrain (hfield), ramp, stairs, floor patches, bumps, rocks
    cena_plana.xml     flat scene without obstacles (to measure the controller)
Inputs:
    models/cad/mass_properties.json, visual_meshes.json, collision.json   (tools/build_assets.py)
    models/meshes/visual/*.stl, models/meshes/collision/*.stl
    tools/robot_def.py   joint points (from the CAD)

    python3 scripts/generate_mjcf.py

Conventions: X right (X+ = leg "xp"), Y forward, Z up. All joints rotate about +X. Angle 0 of every joint is
the CAD pose (legs almost straight).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT.parent / "control"))
from robot_def import LEGS, GEAR, link_origin  # noqa: E402
import five_bar  # noqa: E402
from repo_control_policy import PARK_LEG_LENGTH  # noqa: E402

MODELS = ROOT / "models"
CAD = MODELS / "cad"
I = json.load(open(CAD / "mass_properties.json"))["inertial"]
V = json.load(open(CAD / "visual_meshes.json"))
C = json.load(open(CAD / "collision.json"))

# ===================== PARAMETROS DOS ATUADORES (editar aqui) =====================
GEAR_RATIO = 9.0          # actuator reduction (hip and wheel)
HIP_MOTOR_TAU = 2.0       # [N.m] peak MOTOR torque -> 18 N.m at the joint   (PROVISIONAL)
WHEEL_MOTOR_TAU = 2.0     # [N.m] same for the wheel                         (PROVISIONAL)
HIP_ARMATURE = 0.006      # [kg.m^2] rotor inertia reflected to the joint (J_rotor*N^2) (PROVISIONAL)
WHEEL_ARMATURE = 0.006    # (PROVISORIO)
JOINT_DAMPING = 0.05      # [N.m.s/rad] atrito viscoso
SERVO_TAU = 0.25          # [N.m] MG90S landing-gear servo
SERVO_KP = 3.0
TIRE_FRICTION = "1.0 0.02 0.002"
# ==================================================================================

SIDES = ("xp", "xm")   # xp = direita (X+), xm = esquerda (X-)


def f(v):
    return " ".join("%.6g" % x for x in v)


def mm(p):
    return np.array(p, float) / 1000.0


def rel(child, parent):
    return mm(link_origin(child)) - mm(link_origin(parent))


def inertial(name):
    d = I[name]
    return '<inertial pos="%s" mass="%.6g" fullinertia="%s"/>' % (f(d["com_local"]), d["mass"], f(d["fullinertia"]))


def geoms(name, ind):
    s = [f'{ind}<geom class="visual" mesh="{v["mesh"]}" rgba="{f(v["rgba"])}"/>' for v in V[name]]
    c = C.get(name)
    if c and c["type"] == "cylinder":
        s.append(f'{ind}<geom class="collision" name="{name}_tire" type="cylinder" '
                 f'fromto="{c["x0"]:.5f} 0 0 {c["x1"]:.5f} 0 0" size="{c["radius"]:.5f}" friction="{TIRE_FRICTION}"/>')
    elif c:
        s += [f'{ind}<geom class="collision" mesh="{m}"/>' for m in c["meshes"]]
    return "\n".join(s)


def body(name, parent, joint, ind, children="", extra=""):
    p = rel(name, parent)
    j = ""
    if joint:
        jn, attrs = joint
        j = f'{ind}  <joint name="{jn}" {attrs}/>\n'
    return (f'{ind}<body name="{name}" pos="{f(p)}">\n{ind}  {inertial(name)}\n{j}'
            f'{geoms(name, ind + "  ")}\n{extra}{children}{ind}</body>\n')


def leg(side, ind):
    L = LEGS[side]
    wheel = body(f"wheel_{side}", f"shin_{side}", (f"wheel_{side}", 'class="wheel"'), ind + "    ")
    # wheel-centre site on the shin (used by the VMC / leg state)
    wc = mm(L["foot"]) - mm(L["kneeYp"])
    shin = body(f"shin_{side}", f"armYp_{side}", (f"knee_Yp_{side}", 'class="passive"'), ind + "  ", wheel,
                extra=f'{ind}    <site name="wheel_center_{side}" pos="{f(wc)}" size="0.006" rgba="1 0.5 0 1"/>\n')
    armp = body(f"armYp_{side}", "base", (f"hip_Yp_{side}", 'class="hip"'), ind, shin)
    foot_in_link = mm(L["foot"]) - mm(L["kneeYm"])
    link = body(f"linkYm_{side}", f"armYm_{side}", (f"knee_Ym_{side}", 'class="passive"'), ind + "  ",
                extra=f'{ind}    <site name="foot_{side}" pos="{f(foot_in_link)}" size="0.004"/>\n')
    armm = body(f"armYm_{side}", "base", (f"hip_Ym_{side}", 'class="hip"'), ind, link)
    return armp + armm


def hip_sites(ind):
    out = []
    for s in SIDES:
        L = LEGS[s]
        hp, hm = mm(L["hipYp"]), mm(L["hipYm"])
        out.append(f'{ind}<site name="hip_Yp_{s}" pos="{f(hp)}" size="0.005" rgba="0 1 0 1"/>')
        out.append(f'{ind}<site name="hip_Ym_{s}" pos="{f(hm)}" size="0.005" rgba="0 1 0 1"/>')
        # "virtual hip" of the equivalent leg (midpoint of the two hips): origin of the leg vector
        out.append(f'{ind}<site name="hip_mid_{s}" pos="{f(0.5 * (hp + hm))}" size="0.006" rgba="1 1 0 1"/>')
    return "\n".join(out) + "\n"


def robot_xml():
    gears = ""
    for g, d in GEAR.items():
        lo, hi = d["range_deg"]
        gears += body(g, "base", (g, f'class="gear" range="{lo} {hi}"'), "      ")

    ztire = C["wheel_xp"]["radius"] + (-LEGS["xp"]["foot"][2]) / 1000.0
    imu = mm((-11.6, 25.7, -38.7))   # posicao do WitMotion no CAD

    assets = []
    for vs in V.values():
        assets += [f'    <mesh name="{v["mesh"]}" file="visual/{v["mesh"]}.stl"/>' for v in vs]
    for c in C.values():
        if c["type"] == "hulls":
            assets += [f'    <mesh name="{m}" file="collision/{m}.stl"/>' for m in c["meshes"]]

    eq = "\n".join(
        f'    <connect name="loop_{s}" body1="linkYm_{s}" body2="shin_{s}" '
        f'anchor="{f(mm(LEGS[s]["foot"]) - mm(LEGS[s]["kneeYm"]))}" solref="0.004 1" solimp="0.95 0.99 0.001"/>'
        for s in SIDES)

    act = []
    for s in SIDES:
        for h in ("Yp", "Ym"):
            act.append(f'    <motor name="hip_{h}_{s}" joint="hip_{h}_{s}" gear="{GEAR_RATIO}" '
                       f'ctrlrange="{-HIP_MOTOR_TAU} {HIP_MOTOR_TAU}"/>')
    for s in SIDES:
        act.append(f'    <motor name="wheel_{s}" joint="wheel_{s}" gear="{GEAR_RATIO}" '
                   f'ctrlrange="{-WHEEL_MOTOR_TAU} {WHEEL_MOTOR_TAU}"/>')
    for g, d in GEAR.items():
        lo, hi = d["range_deg"]
        act.append(f'    <position name="servo_{g}" joint="{g}" kp="{SERVO_KP}" '
                   f'ctrlrange="{np.radians(lo):.4f} {np.radians(hi):.4f}" forcerange="{-SERVO_TAU} {SERVO_TAU}"/>')

    joints = [f"hip_{h}_{s}" for s in SIDES for h in ("Yp", "Ym")] + [f"wheel_{s}" for s in SIDES] + list(GEAR)
    sens = ['    <framequat name="imu_quat" objtype="site" objname="imu"/>',
            '    <gyro name="imu_gyro" site="imu"/>',
            '    <accelerometer name="imu_acc" site="imu"/>']
    sens += [f'    <jointpos name="{j}_pos" joint="{j}"/>' for j in joints]
    sens += [f'    <jointvel name="{j}_vel" joint="{j}"/>' for j in joints]

    xml = f'''<mujoco model="robo_bipedal">
  <!-- GENERATED by scripts/generate_mjcf.py from the Fusion assembly "Robo bipedal - montagem principal".
       Frames: X right (X+ = leg xp), Y forward, Z up. SI units.
       Each leg is a CLOSED 5-bar: the loop closes at the wheel axle (equality/connect "loop_*").
       All joints rotate about the body +X. Angle 0 = CAD pose.
       Actuators: ctrl = MOTOR torque [N.m]; gear = 9:1 reduction (joint torque = 9*ctrl). -->
  <compiler angle="degree" meshdir="meshes" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default>
    <joint damping="{JOINT_DAMPING}" axis="1 0 0"/>
    <default class="hip"><joint armature="{HIP_ARMATURE}"/></default>
    <default class="wheel"><joint armature="{WHEEL_ARMATURE}" damping="0.01"/></default>
    <default class="passive"><joint damping="0.01"/></default>
    <default class="gear"><joint damping="0.02" armature="0.0005"/></default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>
    <default class="collision"><geom type="mesh" contype="2" conaffinity="1" group="3" density="0" rgba="0.8 0.3 0.3 0.4" friction="0.8 0.02 0.002"/></default>
  </default>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <body name="base" pos="0 0 {ztire + 0.002:.4f}">
      <freejoint name="root"/>
      {inertial("base")}
{geoms("base", "      ")}
      <site name="imu" pos="{f(imu)}" size="0.006" rgba="0 0.6 1 1"/>
{hip_sites("      ")}{leg("xp", "      ")}{leg("xm", "      ")}{gears}    </body>
  </worldbody>
  <equality>
{eq}
  </equality>
  <actuator>
{chr(10).join(act)}
  </actuator>
  <sensor>
{chr(10).join(sens)}
  </sensor>
</mujoco>
'''
    return xml, ztire


COMMON_VISUAL = '''  <visual>
    <global azimuth="135" elevation="-20" offwidth="1920" offheight="1080"/>
    <headlight ambient="0.4 0.4 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <statistic center="0 0 0.3" extent="1.2"/>'''

# The robot faces +Y in its own frame. In the scenes it spawns rotated -90 deg about Z, facing world +X
# (where the ramp and bumps of the arena are).
SPAWN_QUAT = "0.70710678 0 0 -0.70710678"


def parked_qpos():
    """Parked: legs at PARK_LEG_LENGTH, landing gear down (0 rad), tires on the ground.
    qpos joint order follows the body tree: per leg hip_Yp, knee_Yp, wheel, hip_Ym, knee_Ym; then the gears."""
    q, z = five_bar.nominal_pose(PARK_LEG_LENGTH, wheel_radius=C["wheel_xp"]["radius"])
    order = [f"{j}_{s}" for s in SIDES for j in ("hip_Yp", "knee_Yp", "wheel", "hip_Ym", "knee_Ym")]
    return z, [q[n] for n in order] + [0.0, 0.0]


def key(z):
    """Keyframe 0 = "parked" (default start: the controller stands up and retracts the gear by itself);
    keyframe 1 = "home" (CAD pose, legs almost straight, gear down)."""
    zp, jp = parked_qpos()
    return (f'  <keyframe>\n'
            f'    <key name="parked" qpos="0 0 {zp:.4f} {SPAWN_QUAT} {" ".join(f"{q:.6f}" for q in jp)}"/>\n'
            f'    <key name="home" qpos="0 0 {z:.4f} {SPAWN_QUAT} {" ".join(["0"] * 12)}"/>\n'
            f'  </keyframe>')


def arena_xml(z):
    geoms_txt = (CAD / "arena_geoms.xml.part").read_text()
    return f'''<mujoco model="robo_bipedal_arena">
  <!-- GENERATED by scripts/generate_mjcf.py. Robot (robo_bipedal.xml) + arena from the earlier project. -->
  <include file="robo_bipedal.xml"/>
{COMMON_VISUAL}
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.62 0.82" rgb2="0.86 0.90 0.95" width="256" height="256"/>
    <texture name="ground_grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.16 0.20 0.24" rgb2="0.22 0.27 0.32"/>
    <material name="ground" texture="ground_grid" texrepeat="24 24" reflectance="0.15"/>
    <hfield name="terrain" file="../assets/terrain_heightmap.png" size="10 10 0.6 0.05"/>
    <material name="ramp_wood" rgba="0.66 0.46 0.27 1" reflectance="0.05"/>
    <material name="concrete" rgba="0.62 0.63 0.66 1" reflectance="0.08"/>
    <material name="ice" rgba="0.66 0.83 0.94 0.96" reflectance="0.6"/>
    <material name="grass" rgba="0.24 0.55 0.24 1" reflectance="0.0"/>
    <material name="sand" rgba="0.82 0.71 0.46 1" reflectance="0.02"/>
    <material name="metal_plate" rgba="0.55 0.58 0.62 1" reflectance="0.5"/>
    <material name="rock" rgba="0.40 0.40 0.43 1" reflectance="0.05"/>
    <material name="hump_rubber" rgba="0.10 0.10 0.12 1" reflectance="0.02"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" material="ground" contype="1" conaffinity="1" friction="1.2 0.02 0.002"/>
{geoms_txt}  </worldbody>
{key(z)}
</mujoco>
'''


def flat_xml(z):
    return f'''<mujoco model="robo_bipedal_plano">
  <!-- GENERATED by scripts/generate_mjcf.py. Flat ground, no obstacles (to measure the controller). -->
  <include file="robo_bipedal.xml"/>
{COMMON_VISUAL}
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.62 0.82" rgb2="0.86 0.90 0.95" width="256" height="256"/>
    <texture name="ground_grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.16 0.20 0.24" rgb2="0.22 0.27 0.32"/>
    <material name="ground" texture="ground_grid" texrepeat="24 24" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light name="key" pos="3 -4 6" dir="-0.3 0.5 -1" directional="true" diffuse="0.7 0.7 0.68" specular="0.2 0.2 0.2" castshadow="true"/>
    <geom name="ground" type="plane" size="0 0 0.05" material="ground" contype="1" conaffinity="1" friction="1.2 0.02 0.002"/>
  </worldbody>
{key(z)}
</mujoco>
'''


def main():
    xml, ztire = robot_xml()
    (MODELS / "robo_bipedal.xml").write_text(xml)
    z = ztire + 0.002
    (MODELS / "cena_arena.xml").write_text(arena_xml(z))
    (MODELS / "cena_plana.xml").write_text(flat_xml(z))
    print("ok:", MODELS / "robo_bipedal.xml", "+ 2 scenes")


if __name__ == "__main__":
    main()
