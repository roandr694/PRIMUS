# control

Controller shared by the simulation and the real robot.

| file | what it does |
|---|---|
| `repo_control_policy.py` | LQR (balance) + VMC (leg forces → hip torques through the closed-chain Jacobian) + cascaded PIDs (yaw, roll, leg length, leg sync). Includes the simulated IMU and the two state sources: `robot` (sensors only) and `sim` (ground truth). |
| `lqr_design.py` | derives the LQR gain schedule K(L0) from the MJCF model (port of `sys_calc.m` from Skythinker616/foc-wheel-legged-robot). `--validate` checks the pipeline against the reference; `--write` regenerates the gains. |
| `lqr_gains.py` | generated gain table (L0 from 0.17 to 0.54 m). Do not edit by hand. |
| `five_bar.py` | analytic closed 5-bar leg kinematics (pose for a leg length, knee angles, axle position); used by the LQR design and by the GPU controller in `sim/warp_rl/`. |

The controller runs the landing gear automatically: the robot starts parked (legs at 0.18 m, gear down),
stands up, retracts the gear once still, operates at any leg length from 0.18 to 0.52 m, and lands
again on `RobotCommand(park=True)`. See `Mode` in `repo_control_policy.py`.

Design notes, conventions and results (in Portuguese): [`../docs/notas_modelo_controle.md`](../docs/notas_modelo_controle.md).
