# Design parameters and units

A design YAML contains `design`, `simulation`, `visualization` and
`physical_limits`. Complete examples are under `examples/designs/`.

| Design field | Meaning | Unit |
| --- | --- | --- |
| `profile` | `matlab`: reconciled original model; `legacy`: earlier port | — |
| `damping_ratio` | Structural damping ratio | dimensionless |
| `adhesive_thickness_um` | Adhesive thickness | µm |
| `voltage_v` | Piezoelectric drive voltage | V |
| `front_angle_deg`, `hind_angle_deg` | Leg rest angles | degrees |
| `front_leg_offset_mm`, `hind_leg_offset_mm` | Leg attachment offsets | mm |
| `drive_frequency_hz` | Explicit frequency, or `null` for the original calculated rule | Hz |

The material map recomputes masses, inertias, stiffness, damping and actuator
torque. It represents the existing laminate and geometry; arbitrary materials
and leg lengths are not currently YAML inputs. Some stiffnesses are rounded.

For explicit `drive_frequency_hz`, use `tools/run_expanded_design.py` for a
single design. The sweep runner supports this field directly. Original design
files can use `examples/run_design.py`.

Duration and timestep are in seconds. `initial_clearance_um: 0` starts supported;
`501` reproduces the original MATLAB drop. The drop can increase transient
penetration. A result with one start does not establish the result with another.

Mean speed refers to the center of mass (COM). `late_mean_com_speed_m_s` uses
the last half of the simulation: seconds 0.5–1.0 for a one-second run. Changing
duration changes that interval and can change the ranking.

Movie playback and frame cap are independent of integration resolution.
Physical checks use all integration samples, even when saved/animated states
are reduced. This avoids hiding short contact violations in a coarse movie.
