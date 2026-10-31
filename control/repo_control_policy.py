"""Controlador LQR + VMC + PIDs em cascata (estrutura do Skythinker616/foc-wheel-legged-robot),
portado para o robo NOVO (modelo gerado do CAD, pernas 5 barras com motores 9:1).

O que mudou em relacao a versao antiga deste projeto
----------------------------------------------------
* Estado do robo: por padrao (state_source="robot") so o que o robo REAL mede:
    - atitude e velocidades angulares: IMU simulada (sensores `imu_quat` / `imu_gyro` / `imu_acc` num
      site no lugar do WitMotion; ruido/bias opcionais; modo "mahony" estima a atitude so com giro +
      acelerometro);
    - posicao/velocidade para frente: encoders das rodas, CORRIGIDOS pelo giro da canela (giro da base
      no IMU + cinematica dos quadris). O projeto antigo usava so a media dos encoders, o que mistura o
      balanco da perna com a velocidade. Nao enxerga escorregamento, igual ao robo de verdade;
    - contato com o chao: forca normal ESTIMADA pelos torques dos motores do quadril (como no firmware
      de referencia), filtrada, com limiar em N.
  Com state_source="sim" usa o estado perfeito do simulador (velocidade real do centro das rodas,
  forca real no pneu) -- para comparar, ou como observacao privilegiada do critic em RL.
* Convencoes do modelo planar sao as do sys_calc.m e estao escritas abaixo (antes eram compensadas
  com "botoes de sinal" ajustados na tentativa).
* Torques: o LQR/VMC calcula torque NA JUNTA; o comando do atuador e torque NO MOTOR = junta / 9.
* Ganhos do LQR re-derivados com as massas/inercias do CAD (control/lqr_design.py).

Convencoes (frame do robo: X lateral p/ a direita, Y frente, Z cima)
--------------------------------------------------------------------
    x, dx   posicao/velocidade da roda para frente [m, m/s]
    theta   inclinacao da perna (roda -> quadril virtual) em relacao a vertical; + = quadril a frente da roda
    phi     pitch do corpo; + = nariz para cima (rotacao em +X)
    T       torque de UMA roda; + = rola a roda para frente (no motor isso e torque NEGATIVO da junta)
    Tp      torque virtual de UMA perna no quadril; + = aumenta theta (e empurra o corpo nariz-para-cima)
    roll    rotacao em +Y (frente); + = lado direito (xp) desce
    yaw     rotacao em +Z; + = vira para a esquerda
"""
import math
from dataclasses import dataclass, field

import numpy as np

from lqr_gains import LQR_COEFFS, L0_VALID_RANGE


NOMINAL_LEG_LENGTH = 0.27      # [m] quadril virtual -> eixo da roda (perna toda esticada = 0.394 m)
MIN_LEG_LENGTH = 0.18          # abaixo disso o trem de pouso abaixado encosta no chao
MAX_LEG_LENGTH = 0.36
SIDES = ("xp", "xm")           # xp = direita, xm = esquerda


@dataclass
class RobotCommand:
    speed_cmd: float = 0.0          # [m/s] + = para frente
    yaw_speed_cmd: float = 0.0      # [rad/s] + = vira para a esquerda
    leg_length: float = NOMINAL_LEG_LENGTH
    roll_angle: float = 0.0         # [rad]
    brake: bool = False             # freio forte: zera o alvo de velocidade na hora
    gear_down: bool = True          # trem de pouso abaixado (pose do CAD) ou recolhido


@dataclass
class LegState:
    length: float = NOMINAL_LEG_LENGTH
    body_angle: float = 0.0     # angulo da perna no frame da base (0 = vertical), + = quadril a frente
    theta: float = 0.0          # body_angle - phi  (em relacao a vertical do mundo)
    d_length: float = 0.0
    d_body_angle: float = 0.0
    d_theta: float = 0.0
    fwd: float = 0.0            # vetor quadril -> roda no frame da base (frente, cima)
    up: float = -NOMINAL_LEG_LENGTH


class PID:
    def __init__(self, kp, ki, kd, max_integral, max_output, err_lpf_ratio=1.0, deadzone=0.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.max_integral = float(max_integral)
        self.max_output = float(max_output)
        self.err_lpf_ratio = float(err_lpf_ratio)
        self.deadzone = float(deadzone)
        self.reset()

    def reset(self):
        self.error = self.last_error = self.integral = self.output = 0.0

    def calc(self, reference, feedback):
        self.last_error = self.error
        raw_error = 0.0 if abs(reference - feedback) < self.deadzone else reference - feedback
        self.error = raw_error * self.err_lpf_ratio + self.last_error * (1.0 - self.err_lpf_ratio)
        self.output = (self.error - self.last_error) * self.kd + self.error * self.kp
        self.integral = clamp(self.integral + self.error * self.ki, -self.max_integral, self.max_integral)
        self.output = clamp(self.output + self.integral, -self.max_output, self.max_output)
        return self.output


class CascadePID:
    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner
        self.output = 0.0

    def reset(self):
        self.outer.reset()
        self.inner.reset()

    def calc(self, angle_ref, angle_feedback, speed_feedback):
        self.output = self.inner.calc(self.outer.calc(angle_ref, angle_feedback), speed_feedback)
        return self.output


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lqr_k(leg_length):
    """K(L0) 2x6 (linhas T, Tp; colunas [theta, dTheta, x, dx, phi, dPhi]); u = -K x."""
    l0 = clamp(float(leg_length), L0_VALID_RANGE[0], L0_VALID_RANGE[1])
    p = (l0 ** 3, l0 ** 2, l0, 1.0)
    return [[c[0] * p[0] + c[1] * p[1] + c[2] * p[2] + c[3] for c in row] for row in LQR_COEFFS]


def rot_to_rpy(R):
    """Matriz de rotacao da base (mundo <- base) -> (roll, pitch, yaw) na convencao do robo.

    pitch = rotacao em +X (nariz para cima +), roll = rotacao em +Y (direita desce +), yaw em +Z.
    Ordem: yaw (Z), depois pitch (X), depois roll (Y)."""
    R = np.asarray(R).reshape(3, 3)
    pitch = math.asin(clamp(R[2, 1], -1.0, 1.0))
    roll = math.atan2(-R[2, 0], R[2, 2])
    yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_rpy(data, base_body=1):
    """(roll, pitch, yaw) VERDADEIROS da base, na convencao do robo (para os scripts de analise)."""
    return rot_to_rpy(data.xmat[base_body])


def quat_to_euler_wxyz(q):
    """Compatibilidade com os scripts antigos: devolve (roll, pitch, yaw) na convencao do robo."""
    return rot_to_rpy(quat_to_rot(q))


# --------------------------------------------------------------------------------------------
# IMU simulada
# --------------------------------------------------------------------------------------------
@dataclass
class ImuConfig:
    mode: str = "sensor"          # "sensor": usa a orientacao fundida do IMU (como o WitMotion entrega)
                                  # "mahony": estima a atitude so com giro + acelerometro
    gyro_noise: float = 0.0       # [rad/s] desvio padrao do ruido branco do giro
    gyro_bias: float = 0.0        # [rad/s] desvio padrao do bias constante sorteado por eixo
    angle_noise: float = 0.0      # [rad] ruido na orientacao entregue (modo "sensor")
    acc_noise: float = 0.0        # [m/s^2]
    mahony_kp: float = 0.5
    mahony_ki: float = 0.01
    mahony_acc_gate: float = 1.0  # [m/s^2] ignora o acelerometro se |a| fugir de g mais que isso
    seed: int = 0


class SimulatedIMU:
    """Le os sensores do site `imu` (mesma orientacao da base)."""

    def __init__(self, mujoco, model, config=None):
        self.cfg = config or ImuConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        adr = lambda n: int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)])
        self.adr_q, self.adr_g, self.adr_a = adr("imu_quat"), adr("imu_gyro"), adr("imu_acc")
        self.bias = self.rng.normal(0.0, self.cfg.gyro_bias, 3) if self.cfg.gyro_bias > 0 else np.zeros(3)
        self.q_est = None
        self.e_int = np.zeros(3)

    def reset(self, data):
        self.q_est = np.array(data.sensordata[self.adr_q:self.adr_q + 4], float)
        self.e_int[:] = 0.0

    def read(self, data, dt):
        """-> (R mundo<-base 3x3, giro [rad/s] no frame da base, acelerometro [m/s^2] no frame da base)."""
        c = self.cfg
        gyro = np.array(data.sensordata[self.adr_g:self.adr_g + 3], float) + self.bias
        acc = np.array(data.sensordata[self.adr_a:self.adr_a + 3], float)
        if c.gyro_noise > 0:
            gyro = gyro + self.rng.normal(0.0, c.gyro_noise, 3)
        if c.acc_noise > 0:
            acc = acc + self.rng.normal(0.0, c.acc_noise, 3)
        if c.mode == "mahony":
            R = self._mahony(gyro, acc, dt)
        else:
            q = np.array(data.sensordata[self.adr_q:self.adr_q + 4], float)
            R = quat_to_rot(q)
            if c.angle_noise > 0:
                R = R @ _small_rot(self.rng.normal(0.0, c.angle_noise, 3))
        return R, gyro, acc

    def _mahony(self, gyro, acc, dt):
        if self.q_est is None:
            self.q_est = np.array([1.0, 0, 0, 0])
        q = self.q_est
        R = quat_to_rot(q)
        n = np.linalg.norm(acc)
        w = gyro.copy()
        if abs(n - 9.81) < self.cfg.mahony_acc_gate:   # so corrige quando o acelerometro ~ so gravidade
            v_meas = acc / n                       # parado, o acelerometro aponta para cima (frame da base)
            v_est = R.T @ np.array([0.0, 0.0, 1.0])
            e = np.cross(v_meas, v_est)
            self.e_int += self.cfg.mahony_ki * e * dt
            w = w + self.cfg.mahony_kp * e + self.e_int
        dq = 0.5 * np.array([
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0],
        ])
        q = q + dq * dt
        self.q_est = q / np.linalg.norm(q)
        return quat_to_rot(self.q_est)


def _small_rot(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# --------------------------------------------------------------------------------------------
# Controlador
# --------------------------------------------------------------------------------------------
class RepoControlPolicy:
    def __init__(self, mujoco, model, imu_config=None, state_source="robot"):
        """state_source:
            "robot" (padrao): so o que o robo real mede -> encoders das rodas + IMU para a odometria,
                     torque dos motores do quadril para estimar o contato com o chao.
            "sim":   estado perfeito do simulador (velocidade real do centro das rodas, forca real no pneu).
                     Util como referencia / "critic privilegiado" em RL; o robo real NAO tem isso."""
        assert state_source in ("robot", "sim")
        self.state_source = state_source
        self.mujoco = mujoco
        self.model = model
        mid = lambda t, n: _id(mujoco, model, t, n)
        O = mujoco.mjtObj
        self.base_body = mid(O.mjOBJ_BODY, "base")
        self.imu = SimulatedIMU(mujoco, model, imu_config)
        self.dt = float(model.opt.timestep)

        # atuadores: ctrl = torque no motor; torque na junta = gear * ctrl
        self.act = {}
        for s in SIDES:
            for h in ("Yp", "Ym"):
                self.act[f"hip_{h}_{s}"] = mid(O.mjOBJ_ACTUATOR, f"hip_{h}_{s}")
            self.act[f"wheel_{s}"] = mid(O.mjOBJ_ACTUATOR, f"wheel_{s}")
        self.gear = {k: float(model.actuator_gear[a][0]) for k, a in self.act.items()}
        self.joint_limit = {k: float(model.actuator_ctrlrange[a][1]) * self.gear[k] for k, a in self.act.items()}
        self.servo = {g: mid(O.mjOBJ_ACTUATOR, f"servo_{g}") for g in ("gear_front", "gear_rear")}

        self.site = {s: {
            "hip": mid(O.mjOBJ_SITE, f"hip_mid_{s}"),
            "wheel": mid(O.mjOBJ_SITE, f"wheel_center_{s}"),
            "foot": mid(O.mjOBJ_SITE, f"foot_{s}"),
        } for s in SIDES}
        self.hip_dofs = {s: [_dof(mujoco, model, f"hip_Yp_{s}"), _dof(mujoco, model, f"hip_Ym_{s}")] for s in SIDES}
        self.knee_dofs = {s: [_dof(mujoco, model, f"knee_Yp_{s}"), _dof(mujoco, model, f"knee_Ym_{s}")] for s in SIDES}
        self.shin_body = {s: mid(O.mjOBJ_BODY, f"shin_{s}") for s in SIDES}
        self.link_body = {s: mid(O.mjOBJ_BODY, f"linkYm_{s}") for s in SIDES}
        self.tire_geom = {s: mid(O.mjOBJ_GEOM, f"wheel_{s}_tire") for s in SIDES}
        # encoders: roda (relativa a canela) e cadeia base -> braco Y+ -> canela (o angulo do joelho passivo
        # sai da cinematica do 5 barras a partir dos 2 encoders do quadril; aqui lemos direto da junta)
        jd = lambda n: _dof(mujoco, model, n)
        self.wheel_dof = {s: jd(f"wheel_{s}") for s in SIDES}
        self.shin_chain_dofs = {s: [jd(f"hip_Yp_{s}"), jd(f"knee_Yp_{s}")] for s in SIDES}
        self.wheel_radius = float(model.geom_size[self.tire_geom["xp"]][0])
        self.wheel_mass = float(model.body_mass[mid(O.mjOBJ_BODY, "wheel_xp")])
        # massa pendurada abaixo dos motores do quadril que o torque deles nao "ve": roda + canela + link Y-
        self.below_hip_mass = self.wheel_mass + sum(float(model.body_mass[mid(O.mjOBJ_BODY, f"{n}_xp")])
                                                    for n in ("shin", "linkYm"))

        # massa sustentada pelas pernas (tudo menos as rodas) -> peso por perna
        wheels = sum(float(model.body_mass[mid(O.mjOBJ_BODY, f"wheel_{s}")]) for s in SIDES)
        self.robot_mass = float(model.body_subtreemass[self.base_body])
        self.force_bias_n = (self.robot_mass - wheels) * 9.81 / 2.0

        self.target = RobotCommand()
        self.last_leg = {s: LegState() for s in SIDES}

        # ---- PIDs em cascata (estrutura da referencia; ganhos para este robo e dt = 1 ms) ----
        # yaw: comando de taxa vai direto para o laco interno; o externo so corrige deriva de heading
        self.yaw_pid = CascadePID(outer=PID(1.8, 0.0, 0.0, 0.0, 2.0),
                                  inner=PID(3.0, 0.004, 0.0, 2.0, 6.0))
        # roll: P-D nivelador (forca diferencial entre as pernas) [N]
        self.roll_pid = CascadePID(outer=PID(12.0, 0.0, 0.0, 0.0, 3.0),
                                   inner=PID(25.0, 0.0, 0.0, 0.0, 50.0))
        # comprimento da perna -> forca ao longo da perna [N]
        self.leg_length_pid = CascadePID(outer=PID(12.0, 0.0, 0.0, 0.0, 1.0),
                                         inner=PID(250.0, 0.2, 0.0, 30.0, 140.0))
        # mantem as duas pernas com o mesmo angulo (o diferencial de yaw abre as pernas) [N.m]
        self.leg_angle_pid = CascadePID(outer=PID(40.0, 0.0, 0.0, 0.0, 8.0),
                                        inner=PID(4.0, 0.0, 0.0, 0.0, 25.0))

        self.accel_step = 0.006      # [m/s por 4 ms] rampa do comando de velocidade
        self.pos_lead = 0.15         # [m]   quanto o alvo de posicao pode adiantar o estado
        self.speed_lead = 0.6        # [m/s]
        self.brake_speed_lead = 3.0
        self.max_yaw_lead = 1.0
        self.max_leg_length = MAX_LEG_LENGTH
        self.min_leg_length = MIN_LEG_LENGTH
        self.air_retract_length = NOMINAL_LEG_LENGTH
        # suspensao ativa (usa a altura/velocidade vertical da base do simulador). 0 desliga.
        self.susp_height_gain = 0.0
        self.susp_vel_gain = 0.0
        self.susp_ref_alpha = 0.004
        self.ref_z = 0.0
        # contato: pneu "no chao" se a forca normal (real no modo "sim", estimada pelos torques do quadril
        # no modo "robot") passar do limiar; tolera hops curtos
        self.contact_min_force_n = 5.0 if state_source == "sim" else 15.0
        self.contact_lpf = 0.1
        self.air_grace_s = 0.03
        self._air_time = {s: 0.0 for s in SIDES}
        self.tire_force = {s: 0.0 for s in SIDES}
        self._zrel_hist, self._ddz_f = {}, {}
        # odometria
        self.x = 0.0
        self.dx = 0.0
        self.target_position = 0.0
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.theta_zero = 0.0
        self.theta_zero_ki = -0.1  # [rad/(m.s)]
        self.theta_zero_max = 0.15      # [rad]
        self.gear_down_angle = {"gear_front": 0.0, "gear_rear": 0.0}
        self.gear_up_angle = {"gear_front": math.radians(-100.0), "gear_rear": math.radians(100.0)}

    # ---------------------------------------------------------------------------------------
    def reset(self, data):
        for p in (self.yaw_pid, self.roll_pid, self.leg_length_pid, self.leg_angle_pid):
            p.reset()
        self.imu.reset(data)
        self._air_time = {s: 0.0 for s in SIDES}
        # comeca assumindo "no chao" (a estimativa pelo torque do quadril depende da propria forca comandada;
        # comecar em 0 travaria o robo em "no ar")
        self.tire_force = {s: self.force_bias_n for s in SIDES}
        self._zrel_hist, self._ddz_f = {}, {}
        self.ref_z = float(data.qpos[2])
        self.x = 0.0
        self.dx = 0.0
        self.target_position = 0.0
        self.target_speed = 0.0
        R, gyro, _ = self.imu.read(data, self.dt)
        _, pitch, yaw = rot_to_rpy(R)
        self.target_yaw = yaw
        self.last_leg = {s: self._leg_state(data, s, None, pitch, gyro[0]) for s in SIDES}
        self.theta_zero = 0.0

    def apply(self, data, command=None):
        command = command or RobotCommand()
        dt = self.dt

        # ---------- estado: IMU + simulador ----------
        R, gyro, acc = self.imu.read(data, dt)
        roll, pitch, yaw = rot_to_rpy(R)
        pitch_rate, roll_rate, yaw_rate = float(gyro[0]), float(gyro[1]), float(gyro[2])
        legs = {s: self._leg_state(data, s, self.last_leg[s], pitch, pitch_rate) for s in SIDES}
        self.last_leg = legs
        self._forward_state(data, R, gyro)
        grounded = self._touching_ground(data, R, acc)

        avg_len = 0.5 * (legs["xp"].length + legs["xm"].length)
        avg_dlen = 0.5 * (legs["xp"].d_length + legs["xm"].d_length)
        self._update_target(command, yaw, avg_len, dt)

        # integral lento do erro de posicao -> offset de equilibrio da perna. Compensa o CoM fora do eixo
        # (o CoM do corpo fica ~1 cm atras do quadril) e rampas, sem o robo ficar rolando devagar.
        if grounded and not command.brake:
            self.theta_zero = clamp(self.theta_zero + self.theta_zero_ki * (self.x - self.target_position) * dt,
                                    -self.theta_zero_max, self.theta_zero_max)
        theta = 0.5 * (legs["xp"].theta + legs["xm"].theta) - self.theta_zero
        d_theta = 0.5 * (legs["xp"].d_theta + legs["xm"].d_theta)
        state = [theta, d_theta, self.x - self.target_position, self.dx - self.target_speed, pitch, pitch_rate]
        K = lqr_k(avg_len)
        T = -sum(K[0][i] * state[i] for i in range(6))     # u = -K x
        Tp = -sum(K[1][i] * state[i] for i in range(6))

        # ---------- yaw (diferencial das rodas) ----------
        yaw_rate_sp = command.yaw_speed_cmd + self.yaw_pid.outer.calc(0.0, -wrap(self.target_yaw - yaw))
        yaw_out = self.yaw_pid.inner.calc(yaw_rate_sp, yaw_rate)   # + = vira para a esquerda
        if grounded:
            T_side = {"xp": T + yaw_out, "xm": T - yaw_out}           # roda direita mais para frente -> esquerda
        else:
            T_side = {"xp": 0.0, "xm": 0.0}

        # ---------- forca ao longo da perna: comprimento + peso + roll ----------
        body_z, body_vz = float(data.qpos[2]), float(data.qvel[2])
        self.ref_z += self.susp_ref_alpha * (body_z - self.ref_z)
        susp = clamp(-self.susp_height_gain * (body_z - self.ref_z) - self.susp_vel_gain * body_vz, -0.06, 0.06)
        length_target = clamp(command.leg_length + susp, self.min_leg_length, self.max_leg_length) \
            if grounded else self.air_retract_length
        f_len = self.leg_length_pid.calc(length_target, avg_len, avg_dlen)
        roll_out = self.roll_pid.calc(command.roll_angle, roll, roll_rate)   # roll > ref -> negativo
        bias = self.force_bias_n if grounded else 0.0
        F = {"xp": f_len + bias - roll_out, "xm": f_len + bias + roll_out}  # roll + (direita desce) -> empurra a direita
        for s in SIDES:
            if legs[s].length > self.max_leg_length:
                F[s] -= (legs[s].length - self.max_leg_length) * 2000.0

        # ---------- torque virtual da perna: LQR + sincronismo das pernas ----------
        la = self.leg_angle_pid.calc(0.0, legs["xm"].theta - legs["xp"].theta,
                                     legs["xm"].d_theta - legs["xp"].d_theta)
        Tp_side = {"xm": Tp + la, "xp": Tp - la}

        tau = {}
        for s in SIDES:
            tau[f"hip_Yp_{s}"], tau[f"hip_Ym_{s}"] = self._vmc(data, s, legs[s], F[s], Tp_side[s])
            tau[f"wheel_{s}"] = -T_side[s]          # rolar para frente = torque negativo na junta (+X)

        if self._protect_triggered(theta, pitch):
            tau = {k: 0.0 for k in tau}

        controls = {}
        for k, v in tau.items():
            lim = self.joint_limit[k]
            controls[k] = clamp(v, -lim, lim)
            data.ctrl[self.act[k]] = controls[k] / self.gear[k]
        for g, a in self.servo.items():
            data.ctrl[a] = (self.gear_down_angle if command.gear_down else self.gear_up_angle)[g]

        # chaves com os nomes antigos (left = xm, right = xp) para os scripts de analise/joystick
        return {
            "theta": theta, "d_theta": d_theta, "pitch": pitch, "roll": roll, "yaw": yaw,
            "x": self.x, "dx": self.dx,
            "target_position": self.target_position, "target_speed": self.target_speed,
            "left_leg_length": legs["xm"].length, "right_leg_length": legs["xp"].length,
            "lqr_out_t": T, "lqr_out_tp": Tp,
            "touching_ground": grounded, "tire_force": dict(self.tire_force),
            "controls": controls,   # torques NA JUNTA [N.m]
            "left_wheel": controls["wheel_xm"], "right_wheel": controls["wheel_xp"],
        }

    # ---------------------------------------------------------------------------------------
    def _update_target(self, command, yaw, leg_length, dt):
        step_ratio = dt / 0.004
        slope = max(0.003 * step_ratio, (-(leg_length - NOMINAL_LEG_LENGTH) * 0.02 + self.accel_step) * step_ratio)
        if command.brake:
            self.target_speed = 0.0
        else:
            err = command.speed_cmd - self.target_speed
            self.target_speed = command.speed_cmd if abs(err) < slope else self.target_speed + math.copysign(slope, err)
        lead = self.brake_speed_lead if command.brake else self.speed_lead
        self.target_position += self.target_speed * dt
        if command.brake:
            self.target_position = self.x
        else:
            self.target_position = clamp(self.target_position, self.x - self.pos_lead, self.x + self.pos_lead)
        self.target_speed = clamp(self.target_speed, self.dx - lead, self.dx + lead)
        if abs(command.yaw_speed_cmd) > 1e-3:
            self.target_yaw = yaw
        else:
            self.target_yaw = yaw + clamp(wrap(self.target_yaw - yaw), -self.max_yaw_lead, self.max_yaw_lead)

    def _forward_state(self, data, R, gyro=None):
        if self.state_source == "robot":
            self._forward_state_encoders(data, gyro)
        else:
            self._forward_state_sim(data, R)

    def _forward_state_encoders(self, data, gyro):
        """Odometria do robo real: encoders das rodas + giro do IMU + encoders do quadril.

        O encoder da roda mede o giro RELATIVO a canela (onde fica o estator). O giro absoluto da roda e
        esse + o giro da canela no mundo = giro da base (IMU) + giro da canela em relacao a base (cinematica
        dos quadris). Rolando sem escorregar: v = -R * omega_abs (+X da junta = roda andando para tras).
        Nao enxerga escorregamento nem roda no ar -- igual ao robo de verdade."""
        v = 0.0
        for s in SIDES:
            w_shin = float(gyro[0]) + sum(float(data.qvel[d]) for d in self.shin_chain_dofs[s])
            v += -self.wheel_radius * (float(data.qvel[self.wheel_dof[s]]) + w_shin)
        self.dx = v / len(SIDES)
        self.x += self.dx * self.dt

    def _forward_state_sim(self, data, R):
        """Velocidade REAL do centro das rodas (simulador), projetada no heading do robo; x integrado."""
        h = np.array(R)[:, 1].copy()
        h[2] = 0.0
        n = np.linalg.norm(h)
        h = h / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        v = np.zeros(6)
        vel = np.zeros(3)
        for s in SIDES:
            self.mujoco.mj_objectVelocity(self.model, data, self.mujoco.mjtObj.mjOBJ_SITE, self.site[s]["wheel"], v, 0)
            vel += v[3:6]
        self.dx = float(np.dot(vel / len(SIDES), h))
        self.x += self.dx * self.dt

    def _leg_state(self, data, side, prev, pitch, pitch_rate):
        """Geometria da perna a partir da cinematica (= encoders dos quadris) no frame da base."""
        Rb = data.xmat[self.base_body].reshape(3, 3)
        v = Rb.T @ (data.site_xpos[self.site[side]["wheel"]] - data.site_xpos[self.site[side]["hip"]])
        fwd, up = float(v[1]), float(v[2])
        length = max(1e-6, math.hypot(fwd, up))
        body_angle = math.atan2(-fwd, -up)          # quadril em relacao a roda; + = quadril a frente
        st = LegState(length=length, body_angle=body_angle, theta=body_angle - pitch, fwd=fwd, up=up)
        if prev is not None:
            st.d_length = (length - prev.length) / self.dt
            st.d_body_angle = wrap(body_angle - prev.body_angle) / self.dt
            st.d_theta = st.d_body_angle - pitch_rate
        return st

    def _vmc(self, data, side, leg, force_n, tp):
        """VMC: forca F ao longo da perna (+ = estica, empurra o chao) e torque Tp -> torques dos 2 quadris.

        Jacobiano do centro da roda em relacao aos 2 quadris, com os joelhos passivos se movendo como o
        laco fechado exige (projecao pela restricao). tau = J^T * F_mundo."""
        L = max(leg.length, 1e-6)
        e = np.array([leg.fwd, leg.up]) / L                 # unitario quadril -> roda
        perp = np.array([e[1], -e[0]])                      # forca na roda que gera Tp (+theta) no quadril
        f_fu = force_n * e + (tp / L) * perp
        Rb = data.xmat[self.base_body].reshape(3, 3)
        f_world = Rb @ np.array([0.0, f_fu[0], f_fu[1]])

        t = self._leg_jacobian(data, side).T @ f_world
        return float(t[0]), float(t[1])

    def _leg_jacobian(self, data, side):
        """Jacobiano 3x2 (mundo) do centro da roda em relacao aos 2 quadris, com o laco fechado."""
        m, mj, nv = self.model, self.mujoco, self.model.nv
        Js = np.zeros((3, nv)); scratch = np.zeros((3, nv))
        mj.mj_jacSite(m, data, Js, scratch, self.site[side]["wheel"])
        anchor = data.site_xpos[self.site[side]["foot"]]
        J1 = np.zeros((3, nv)); J2 = np.zeros((3, nv))
        mj.mj_jac(m, data, J1, scratch, anchor, self.link_body[side])
        mj.mj_jac(m, data, J2, scratch, anchor, self.shin_body[side])
        C = J1 - J2
        hips, knees = self.hip_dofs[side], self.knee_dofs[side]
        dk_dh = -np.linalg.lstsq(C[:, knees], C[:, hips], rcond=None)[0]
        return Js[:, hips] + Js[:, knees] @ dk_dh

    def _touching_ground(self, data, R=None, acc=None):
        if self.state_source == "robot":
            az = float((np.asarray(R) @ acc)[2]) - 9.81 if acc is not None else 0.0   # acel. vertical da base
            force = self._estimated_normal_force(data, az)
        else:
            force = self._sim_normal_force(data)
        self.tire_force = force
        ok = True
        for s in SIDES:
            self._air_time[s] = 0.0 if force[s] > self.contact_min_force_n else self._air_time[s] + self.dt
            ok = ok and self._air_time[s] <= self.air_grace_s
        return ok

    def _estimated_normal_force(self, data, acc_world_z=0.0):
        """Como no robo real (mesma ideia do firmware de referencia): torque medido dos 2 motores do quadril
        -> forca que a perna faz na roda (inverte tau = J^T f no plano da perna) -> componente vertical
        + m_baixo * (g + aceleracao vertical da roda). A aceleracao da roda = aceleracao da base (IMU)
        + 2a derivada da altura da roda em relacao ao quadril (cinematica). Sem esse termo, com a roda no
        ar a forca comandada apareceria como apoio. Filtrado."""
        Rb = data.xmat[self.base_body].reshape(3, 3)
        out = {}
        for s in SIDES:
            tau = np.array([float(data.qfrc_actuator[d]) for d in self.hip_dofs[s]])
            A = self._leg_jacobian(data, s).T @ Rb[:, 1:3]      # tau = A @ f_(frente, cima) no frame da base
            try:
                f_fu = np.linalg.solve(A, tau)
            except np.linalg.LinAlgError:
                f_fu = np.zeros(2)
            fz = float((Rb[:, 1:3] @ f_fu)[2])                  # forca da perna na roda, eixo Z do mundo
            leg = self.last_leg[s]
            zrel = float((Rb[:, 1:3] @ np.array([leg.fwd, leg.up]))[2])
            h = self._zrel_hist.setdefault(s, [zrel, zrel])
            ddz = (zrel - 2.0 * h[0] + h[1]) / (self.dt * self.dt)
            h[1], h[0] = h[0], zrel
            self._ddz_f[s] = self._ddz_f.get(s, 0.0) + 0.05 * (ddz - self._ddz_f.get(s, 0.0))
            n = -fz + self.below_hip_mass * (9.81 + acc_world_z + self._ddz_f[s])
            prev = self.tire_force.get(s, n)
            out[s] = prev + self.contact_lpf * (n - prev)
        return out

    def _sim_normal_force(self, data):
        """Contato direto do simulador: soma da forca normal de cada pneu."""
        f6 = np.zeros(6)
        force = {s: 0.0 for s in SIDES}
        for i in range(data.ncon):
            c = data.contact[i]
            for s in SIDES:
                if c.geom1 == self.tire_geom[s] or c.geom2 == self.tire_geom[s]:
                    self.mujoco.mj_contactForce(self.model, data, i, f6)
                    force[s] += abs(float(f6[0]))
        return force

    def _protect_triggered(self, theta, pitch):
        return abs(theta) > math.pi / 3.0 or abs(pitch) > math.pi / 3.0


def _id(mujoco, model, objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"nao achei '{name}' no modelo")
    return i


def _dof(mujoco, model, name):
    return int(model.jnt_dofadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
