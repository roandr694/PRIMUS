"""Closed five-bar reset pose in SI units; no simulator dependency."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from robot_def import LEGS


def rotate(v, a):
    y, z = v
    return (math.cos(a)*y - math.sin(a)*z, math.sin(a)*y + math.cos(a)*z)


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def leg_pose(side, angle):
    """Return closed joint angles [rad] and axle position [m] in the CAD frame."""
    leg = {k: tuple(x/1000 for x in v[1:]) for k, v in LEGS[side].items()}
    hp, hm = leg['hipYp'], leg['hipYm']
    ap, am = rotate(sub(leg['kneeYp'], hp), angle), rotate(sub(leg['kneeYm'], hm), -angle)
    kp, km = tuple(x+y for x,y in zip(hp,ap)), tuple(x+y for x,y in zip(hm,am))
    vp, vm = sub(leg['foot'],leg['kneeYp']), sub(leg['foot'],leg['kneeYm'])
    rp, rm = math.hypot(*vp), math.hypot(*vm)
    dy, dz = sub(km,kp); d = math.hypot(dy,dz)
    along = (rp*rp-rm*rm+d*d)/(2*d)
    h2 = rp*rp-along*along
    if h2 < -1e-12:
        raise ValueError('Hip pose has no closed five-bar solution')
    h = math.sqrt(max(0,h2)); cy,cz = kp[0]+along*dy/d,kp[1]+along*dz/d
    candidates = [(cy-h*dz/d,cz+h*dy/d),(cy+h*dz/d,cz-h*dy/d)]
    foot = min(candidates,key=lambda p:p[1])
    def knee(k,v,hip):
        w = sub(foot,k)
        return math.atan2(math.sin(math.atan2(w[1],w[0])-math.atan2(v[1],v[0])-hip),
                          math.cos(math.atan2(w[1],w[0])-math.atan2(v[1],v[0])-hip))
    q = {f'hip_Yp_{side}':angle, f'hip_Ym_{side}':-angle,
         f'knee_Yp_{side}':knee(kp,vp,angle), f'knee_Ym_{side}':knee(km,vm,-angle), f'wheel_{side}':0.0}
    return q,foot


def nominal_pose(length=0.27):
    """Return closed nominal joints [rad] and base height [m] for both wheels on flat ground."""
    if not 0.18 <= length <= 0.36:
        raise ValueError('Leg length must be in [0.18, 0.36] m')
    q = {}; heights = []
    for side,L in LEGS.items():
        hy,hz = ((L['hipYp'][i]+L['hipYm'][i])/2000 for i in (1,2))
        lo,hi = 0.0,math.radians(85)
        for _ in range(55):
            a = (lo+hi)/2; _,f = leg_pose(side,a)
            if math.hypot(f[0]-hy,f[1]-hz)>length: lo=a
            else: hi=a
        joints,foot = leg_pose(side,(lo+hi)/2); q.update(joints)
        heights.append(0.1075-foot[1]+0.002)
    q.update(gear_front=-math.radians(100),gear_rear=math.radians(100))
    return q,sum(heights)/len(heights)
