# control

Controller shared by the simulation and the real robot.

| file | what it does |
|---|---|
| `repo_control_policy.py` | LQR (balance) + VMC (leg forces → hip torques through the closed-chain Jacobian) + cascaded PIDs (yaw, roll, leg length, leg sync). Includes the simulated IMU and the two state sources: `robot` (sensors only) and `sim` (ground truth). |
| `lqr_design.py` | derives the LQR gain schedule K(L0) from the MJCF model (port of `sys_calc.m` from Skythinker616/foc-wheel-legged-robot). `--validate` checks the pipeline against the reference; `--write` regenerates the gains. |
| `lqr_gains.py` | generated gain table. Do not edit by hand. |

Design notes, conventions and results (in Portuguese): [`../docs/notas_modelo_controle.md`](../docs/notas_modelo_controle.md).
