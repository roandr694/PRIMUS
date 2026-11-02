"""Derives the LQR gain table K(L0) of the wheel-legged robot from the CAD-generated MJCF.

Python port of `sys_calc.m` from the reference project (Skythinker616/foc-wheel-legged-robot): for each leg
length L0 it linearises the planar wheel + variable-length pendulum + body model, discretises it, solves the
infinite-horizon discrete LQR and fits each gain with a cubic polynomial in L0.

The physical parameters come from models/robo_bipedal.xml at each L0, with the closed 5-bar leg actually posed
(robot_model.leg_lqr_params): leg mass and CoM (the wheel motor sits on the shin, so the leg CoM is near the
wheel), leg inertia about its CoM, wheel inertia + rotor armature, body CoM (below the hip on this robot).
The model is HALF a robot (1 leg + 1 wheel + half body), so T is the torque of ONE wheel and Tp the virtual
torque of ONE leg, exactly what the controller applies per side. dt = 0.001 s (the MJCF step).

    python3 control/lqr_design.py --validate   # checks the pipeline against the small reference robot
    python3 control/lqr_design.py              # prints the robot parameters and gains
    python3 control/lqr_design.py --write      # writes control/lqr_gains.py
"""

import argparse

import numpy as np
from scipy.linalg import expm, solve_discrete_are


# State order as in sys_calc.m:
#   x = [theta, dTheta, x, dx, phi, dPhi]     theta = leg angle, phi = body pitch
#   u = [T (wheel torque), Tp (virtual hip torque)]
STATE_LABELS = ["theta", "dTheta", "x", "dx", "phi", "dPhi"]

# Small reference robot (sys_calc.m), only to validate the pipeline.
SMALL_ROBOT = {
    "R": 0.02,
    "mw": 0.251327,
    "mp": 0.0248,
    "M": 0.325,
    "Iw": 5.02655e-05,
    "Ip": (9.66667e-07 + 7.9975e-06) * 2,
    "Im": 0.000338542,
    "l": 0.0,
    "g": 9.8,
    "dt": 0.005,
    "Q": [1.0, 10.0, 100.0, 20.0, 1000.0, 1.0],
    "Ruu": [1.0, 1.0],
    "L0_sweep": np.arange(0.04, 0.14 + 1e-9, 0.01),
}

# This robot. The physical parameters come from the MJCF (see robot_params()).
# Q = [theta, dTheta, x, dx, phi, dPhi], Ruu = [T, Tp].
ROBOT = {
    "g": 9.81,
    "dt": 0.001,
    "Q": [10.0, 4.0, 40.0, 80.0, 600.0, 10.0],
    "Ruu": [4.0, 4.0],
    # L0 range (virtual hip -> wheel axle) of the fit; covers the controller range (0.18-0.52 m) with margin.
    # Geometry (collision meshes): wheel hits the chassis below ~0.17 m; fully extended (singular) 0.565 m.
    "L0_sweep": np.arange(0.17, 0.54 + 1e-9, 0.01),
}


def robot_params(L0, _cache={}):
    """Half-robot parameters at leg length L0, from the MJCF (cached)."""
    key = round(float(L0), 5)
    if key not in _cache:
        import os, sys
        import mujoco
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "scripts"))
        from robot_model import ROBOT_PATH, leg_lqr_params
        if "model" not in _cache:
            _cache["model"] = mujoco.MjModel.from_xml_path(str(ROBOT_PATH))
        _cache[key] = leg_lqr_params(mujoco, _cache["model"], L0)
    return _cache[key]

def accelerations(state, inp, p):
    """Solve the 3 coupled dynamics equations for [x2, theta2, phi2].

    These are equ1/equ2/equ3 from sys_calc.m. They are affine in the three
    accelerations, so we recover the linear system by sampling residuals at the
    origin and three unit accelerations, then solve exactly.
    """
    theta, th1, _x, _x1, phi, ph1 = state
    T, Tp = inp
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    L, Lm, l = p["L"], p["Lm"], p["l"]
    M, mp, mw = p["M"], p["mp"], p["mw"]
    Iw, Ip, Im, R, g = p["Iw"], p["Ip"], p["Im"], p["R"], p["g"]

    def residual(a, b, c):  # a=x2, b=theta2, c=phi2
        Nm = M * (a + (L + Lm) * (b * ct - th1**2 * st) - l * (c * cp - ph1**2 * sp))
        Pm = M * g + M * ((L + Lm) * (-th1**2 * ct - b * st) - l * (ph1**2 * cp + c * sp))
        N = Nm + mp * (a + L * (b * ct - th1**2 * st))
        P = Pm + mp * g + mp * L * (-th1**2 * ct - b * st)
        e1 = a - (T - N * R) / (Iw / R + mw * R)
        e2 = (P * L + Pm * Lm) * st - (N * L + Nm * Lm) * ct - T + Tp - Ip * b
        e3 = Tp + Nm * l * cp + Pm * l * sp - Im * c
        return np.array([e1, e2, e3])

    r0 = residual(0.0, 0.0, 0.0)
    jac = np.column_stack([
        residual(1.0, 0.0, 0.0) - r0,
        residual(0.0, 1.0, 0.0) - r0,
        residual(0.0, 0.0, 1.0) - r0,
    ])
    return np.linalg.solve(jac, -r0)  # [x2, theta2, phi2]


def state_deriv(state, inp, p):
    a, b, c = accelerations(state, inp, p)
    return np.array([state[1], b, state[3], a, state[5], c])


def linearize(p, eps=1e-6):
    """Numerical Jacobians A, B at the upright equilibrium (all states zero)."""
    x0 = np.zeros(6)
    u0 = np.zeros(2)
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = eps
        A[:, j] = (state_deriv(x0 + dx, u0, p) - state_deriv(x0 - dx, u0, p)) / (2 * eps)
    for k in range(2):
        du = np.zeros(2)
        du[k] = eps
        B[:, k] = (state_deriv(x0, u0 + du, p) - state_deriv(x0, u0 - du, p)) / (2 * eps)
    return A, B


def c2d(A, B, dt):
    """Zero-order-hold discretization via the block matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def dlqr(G, H, Q, R):
    P = solve_discrete_are(G, H, Q, R)
    return np.linalg.solve(R + H.T @ P @ H, H.T @ P @ G)  # u = -K x


def gain_schedule(params):
    """Cubic coefficients [c3, c2, c1, c0] of each of the 12 gains, shape (2, 6, 4).

    With params["per_L0"] (function L0 -> dict) the physical parameters are updated at each L0; otherwise the
    leg CoM is at mid-length (L = Lm = L0/2), as in the original sys_calc.m.
    """
    p = dict(params)
    Q = np.diag(params["Q"])
    R = np.diag(params["Ruu"])
    L0s = params["L0_sweep"]
    Ks = np.zeros((2, 6, len(L0s)))
    for idx, L0 in enumerate(L0s):
        if "per_L0" in params:
            p.update({k: v for k, v in params["per_L0"](L0).items() if k in ("L", "Lm", "mp", "Ip", "M", "Im", "l", "mw", "Iw", "R")})
        else:
            p["L"] = L0 / 2.0
            p["Lm"] = L0 / 2.0
        A, B = linearize(p)
        G, H = c2d(A, B, params["dt"])
        Ks[:, :, idx] = dlqr(G, H, Q, R)
    coeffs = np.zeros((2, 6, 4))
    for r in range(2):
        for c in range(6):
            coeffs[r, c] = np.polyfit(L0s, Ks[r, c, :], 3)
    return coeffs, L0s, Ks


def eval_gain(coeffs, L0):
    powers = np.array([L0**3, L0**2, L0, 1.0])
    return np.tensordot(coeffs, powers, axes=([2], [0]))  # (2,6)


# Reference published coefficients (lqr_k.m), as [c3, c2, c1, c0] per entry,
# ordered [K00, K10, K01, K11, ... column-major like matlabFunction]. Used only
# to validate the re-derivation pipeline against the small robot.
REFERENCE_MT = [
    [7.860447805125654e1, -3.991388867291207e1, -5.126991380196162, 2.480998544325396e-2],
    [-3.621574295474152e-1, 6.125917721992718e-2, 1.641856077686523e-2, -2.012885459933266e-3],
    [9.914464208320304, -7.752864657531594, -7.138184021122035e-1, 6.339958822075569e-3],
    [8.096460807167376e-3, -5.330480346213592e-3, 2.598467030390104e-3, -1.809641657435409e-5],
    [3.900283589932653e1, -1.391236665767639e1, -2.538932550843302, 2.720156782574435e-2],
    [-5.654841252418429e-1, 2.173999389707158e-1, -3.220257944972706e-2, 3.976782027053733e-3],
    [2.793115239202604e1, -1.092142163310129e1, -2.330144966025284, 1.581329403874648e-2],
    [1.089192101634851e-1, -4.800661687059873e-2, 8.784508831905395e-3, 3.841540294269537e-4],
    [-3.228861698470031e2, 1.286165933939939e2, -2.01008434091536e1, 1.639137670391309],
    [4.405391731571748, -1.61672432957469, 2.129654558241931e-1, 1.9640865609757],
    [-1.160473605985421e1, 4.591316876933686, -7.070696115116452e-1, 6.021656215297832e-2],
    [7.127268389081669e-2, -2.625587214283516e-2, 3.493240214123524e-3, 7.219094918422932e-2],
]


def validate():
    coeffs, _, _ = gain_schedule(SMALL_ROBOT)
    # Map our (row, col) coeffs to matlabFunction's column-major mt1..mt12.
    mine = []
    for col in range(6):
        for row in range(2):
            mine.append(coeffs[row, col])
    print("Validating re-derivation against reference lqr_k.m (small robot)\n")
    print(f"{'entry':>6} {'coeff':>6} {'reference':>16} {'re-derived':>16} {'rel.err':>10}")
    max_rel = 0.0
    for i, (ref, got) in enumerate(zip(REFERENCE_MT, mine)):
        for power, (a, b) in enumerate(zip(ref, got)):
            denom = abs(a) if abs(a) > 1e-9 else 1.0
            rel = abs(a - b) / denom
            max_rel = max(max_rel, rel)
            label = ["L0^3", "L0^2", "L0^1", "L0^0"][power]
            print(f"mt{i+1:<4} {label:>6} {a:16.6e} {b:16.6e} {rel:10.2e}")
    print(f"\nMax relative coefficient error: {max_rel:.2e}")
    print("PASS" if max_rel < 0.05 else "FAIL (pipeline does not match reference)")




def robot_design():
    return dict(ROBOT, per_L0=robot_params)


def emit():
    print("Physical parameters (half robot) from the MJCF:")
    for L0 in (ROBOT["L0_sweep"][0], 0.27, ROBOT["L0_sweep"][-1]):
        p = robot_params(L0)
        print(f"  L0={L0:.2f}: " + ", ".join(f"{k}={v:.4g}" for k, v in p.items()))
    coeffs, L0s, Ks = gain_schedule(robot_design())
    fit_err = max(abs(eval_gain(coeffs, L0) - Ks[:, :, i]).max() for i, L0 in enumerate(L0s))
    print(f"\nCubic fit: max error {fit_err:.3g}")
    for L0 in (L0s[0], 0.27, L0s[-1]):
        K = eval_gain(coeffs, L0)
        print(f"K(L0={L0:.2f})  T : {np.array2string(K[0], precision=3)}")
        print(f"               Tp: {np.array2string(K[1], precision=3)}")


def write_gains(path):
    coeffs, L0s, _ = gain_schedule(robot_design())
    lines = [
        '"""LQR gain table of the robot -- GENERATED by control/lqr_design.py --write, do not edit by hand.',
        "",
        "Each entry is the cubic [c3, c2, c1, c0] of one gain as a function of the leg length L0:",
        "    K[row, col](L0) = c3*L0**3 + c2*L0**2 + c1*L0 + c0",
        "row 0 -> torque of ONE wheel T [N.m], row 1 -> virtual torque of ONE leg Tp [N.m];",
        "columns = state [theta, dTheta, x, dx, phi, dPhi].  u = -K x.",
        f"Q = {ROBOT['Q']}, R = {ROBOT['Ruu']}, dt = {ROBOT['dt']} s.",
        f"Valid for L0 in [{L0s[0]:.2f}, {L0s[-1]:.2f}] m.",
        '"""',
        "",
        f"L0_VALID_RANGE = ({L0s[0]:.4f}, {L0s[-1]:.4f})",
        "",
        "LQR_COEFFS = [",
    ]
    for r in range(2):
        lines.append(f"    [  # row {r}: {'wheel torque T' if r == 0 else 'hip torque Tp'}")
        for c in range(6):
            c3, c2, c1, c0 = coeffs[r, c]
            lines.append(f"        ({c3: .10e}, {c2: .10e}, {c1: .10e}, {c0: .10e}),  # d/d({STATE_LABELS[c]})")
        lines.append("    ],")
    lines.append("]")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="Reproduce the small reference robot coefficients.")
    parser.add_argument("--write", action="store_true", help="Write control/lqr_gains.py.")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.write:
        import os
        write_gains(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lqr_gains.py"))
    else:
        emit()


if __name__ == "__main__":
    main()
