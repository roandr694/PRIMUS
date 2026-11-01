"""Cinematica analitica da perna 5 barras (plano YZ da base: Y frente, Z cima), sem simulador.

Geometria de sim/tools/robot_def.py (pose do CAD = todos os angulos zero). Cada perna:
quadril Y+ e quadril Y- na base (140 mm), bracos de 220 mm, barras inferiores de 350 mm que se
encontram no eixo da roda. hip_Yp gira o braco da frente, hip_Ym o de tras, ambos em +X.

Usado por:
  * sim/scripts/robot_model.py: pose de cada L0 para o lqr_design.py (sem o Newton do MuJoCo, que
    pula para o outro ramo do mecanismo com a perna mais esticada que a pose do CAD);
  * o controlador vetorizado do treino em GPU (mesmas equacoes, em torch).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "tools"))
from robot_def import LEGS  # noqa: E402

# Faixa geometrica medida com as malhas de colisao (perna simetrica, trem recolhido):
# abaixo de ~0.17 m a roda bate no chassi; 0.565 m e a perna toda esticada (singular).
GEOMETRIC_LENGTH_RANGE = (0.17, 0.565)


def leg_points(side):
    """Pontos do CAD da perna `side` no plano (y, z) [m]."""
    return {k: (v[1] / 1000.0, v[2] / 1000.0) for k, v in LEGS[side].items()}


def _rot(v, a):
    y, z = v
    return (math.cos(a) * y - math.sin(a) * z, math.sin(a) * y + math.cos(a) * z)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _ang(v):
    return math.atan2(v[1], v[0])


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def closure(side, hip_p, hip_m):
    """Fecha o 5 barras para os angulos dos quadris [rad].

    Retorna (knee_Yp, knee_Ym, foot) com os angulos dos joelhos passivos [rad] e o eixo da roda (y, z)
    no frame do CAD, ou None se o laco nao fecha. Escolhe o ramo com a roda para baixo."""
    P = leg_points(side)
    hp, hm = P["hipYp"], P["hipYm"]
    kp = tuple(x + y for x, y in zip(hp, _rot(_sub(P["kneeYp"], hp), hip_p)))
    km = tuple(x + y for x, y in zip(hm, _rot(_sub(P["kneeYm"], hm), hip_m)))
    vp, vm = _sub(P["foot"], P["kneeYp"]), _sub(P["foot"], P["kneeYm"])
    rp, rm = math.hypot(*vp), math.hypot(*vm)
    dy, dz = _sub(km, kp)
    d = math.hypot(dy, dz)
    if d < 1e-9 or d > rp + rm or d < abs(rp - rm):
        return None
    along = (rp * rp - rm * rm + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, rp * rp - along * along))
    cy, cz = kp[0] + along * dy / d, kp[1] + along * dz / d
    foot = min([(cy - h * dz / d, cz + h * dy / d), (cy + h * dz / d, cz - h * dy / d)], key=lambda p: p[1])
    knee_p = _wrap(_ang(_sub(foot, kp)) - _ang(vp) - hip_p)
    knee_m = _wrap(_ang(_sub(foot, km)) - _ang(vm) - hip_m)
    return knee_p, knee_m, foot


def hip_mid(side):
    P = leg_points(side)
    return ((P["hipYp"][0] + P["hipYm"][0]) / 2.0, (P["hipYp"][1] + P["hipYm"][1]) / 2.0)


def leg_length(side, hip_p, hip_m):
    c = closure(side, hip_p, hip_m)
    if c is None:
        return None
    v = _sub(c[2], hip_mid(side))
    return math.hypot(*v)


def symmetric_angle_for_length(side, length):
    """Angulo `a` (hip_Yp = +a, hip_Ym = -a) que da o comprimento `length`. L0 cai monotonamente com `a`
    em [-55, 90] graus; a = 0 e a pose do CAD (0.394 m)."""
    lo, hi = math.radians(-55.0), math.radians(90.0)
    if not leg_length(side, hi, -hi) <= length <= leg_length(side, lo, -lo):
        raise ValueError(f"L0 = {length:.3f} m fora da faixa do mecanismo")
    for _ in range(60):
        a = 0.5 * (lo + hi)
        if leg_length(side, a, -a) > length:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)
