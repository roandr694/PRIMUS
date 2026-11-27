# hardware

Mechanical and electrical design of PRIMUS. Planned contents:

- `cad/`: STEP exports of the main assembly and printable parts. The source of truth is the Fusion 360 project "robo bipedal".
- `bom.md`: bill of materials (actuators, bearings, fasteners, electronics, battery).
- `electrical/`: wiring diagram, power distribution (M18 battery → XL4016 buck → logic), CAN/UART buses.
- `assembly.md`: assembly notes and lessons learned.

The step-by-step build guide, generated from the Fusion 360 assembly with an animation for every step and the full
bill of materials, is at [roandr694.github.io/portfolio/primus-assembly](https://roandr694.github.io/portfolio/primus-assembly/).

The simulation reads joint locations and mass corrections from [`../sim/tools/robot_def.py`](../sim/tools/robot_def.py). Keep it in sync when the hardware changes.
