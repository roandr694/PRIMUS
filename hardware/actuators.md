# PRIMUS actuators

Data from the owner (2026-10-31): 24 V battery, mjbots moteus-r4, a motor equivalent to the OpenQDD one with a
130 kV winding. The printed gearbox was modified and is mechanically stronger; its backlash is similar to the
reference design but unmeasured. The 9:1 reduction is the one documented in this repository.

The [original project](https://www.aaedmusa.com/projects/openqdd) uses a 90 kV Eagle Power motor and an ODrive S1.
Those performance figures do not describe PRIMUS's modified actuator. The exact motor model is still to be
confirmed; no model number is inferred from the verbal reference alone.

## Assumptions for simulation

Data and assumptions are in [actuators.json](actuators.json); Isaac and the URDF generator read this profile.

- Ideal no-load speed: 24 × 130 = 3120 rpm at the rotor, ~346.7 rpm or 36.30 rad/s at the joint with 9:1. No
  voltage drop, losses or field weakening; an approximation of the torque–speed curve, not a measurement.
- The provisional 18 N·m joint limit and 0.006 kg·m² reflected inertia are kept as assumptions. Higher mechanical
  strength does not justify more electrical torque.
- The controller peak limit is not the configured or continuous motor current. The
  [moteus documentation](https://mjbots.github.io/moteus/troubleshooting/limited-torque/) lists the current,
  voltage/back-EMF, power, temperature and torque-command limits.
- Gearbox backlash is not simulated yet: its amplitude is needed to choose a model and randomisation range.
- The CAD battery mass still uses the old 0.68 kg assumption; 24 V alone does not give its mass.

With access to the real controller, a read-only dump of its configuration/calibration can give the maximum
current, calibrated torque constant, resistance, speed limits and rotor/output ratio, improving the model
without the original motor datasheet. No commands were sent and no moteus firmware/configuration was changed.
