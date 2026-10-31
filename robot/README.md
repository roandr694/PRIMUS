# robot

Software for the physical robot. Planned contents:

- motor drivers (hip and wheel actuators, 9:1);
- IMU driver (WitMotion);
- a state estimator that mirrors `state_source="robot"` in [`../control/repo_control_policy.py`](../control/repo_control_policy.py): encoder odometry corrected for leg rotation, contact from hip torques;
- the main control loop running the shared controller from [`../control/`](../control/);
- logging, and tools to compare real runs with simulation.

The controller already separates *what it computes* from *where the measurements come from*. The real robot only needs to provide the same signals (IMU, encoders, motor torques) that the simulation provides in `robot` mode.
