# Tutorial: simulate and compare a design grid

This workflow simulates a grid of physical designs with the analytical model,
checks each trajectory and compares their speeds. All commands assume the
repository root.

## Choose the experiment

`supported_25_cases.yaml` uses five damping ratios and five adhesive thicknesses,
with feet initially supported. `matlab_25_cases.yaml` uses the same designs and
the original MATLAB 501 µm initial drop. Both retain 200 V and the original
design-dependent frequency rule. This is distinct from the fixed-363-Hz neural
optimization study.

The parameter lists define a Cartesian product:

```yaml
parameters:
  damping_ratio: [0.01, 0.03, 0.05, 0.07, 0.09]
  adhesive_thickness_um: [1.0, 5.0, 10.0, 30.0, 50.0]
```

Add values or other numeric fields listed in [design parameters](design_parameters.md)
to build another grid. Add a `voltage_v` axis only when changing actuation is
part of that experiment. `drive_frequency_hz: null` recomputes frequency from
the design; `363.0` fixes it. The applied frequency is recorded for every case.

The base-design path is relative to the sweep YAML. Copy configurations into
`examples/sweeps/` to keep those relative paths working, or use an absolute path.

## Execute and read the results

```bash
python examples/run_sweep.py \
  --config examples/sweeps/supported_25_cases.yaml \
  --output outputs/my_sweep --backend cpu --render-best --export-mat
```

CPU is the measured faster backend for this workload. `--backend cuda` requests
CUDA; the numerical code also runs there. See [performance](performance.md).

For planning, two full-duration cases of the original MATLAB workflow imply
about 32 minutes for 25 cases, excluding graphics/video. Most measured MATLAB
time was per-case numerical-function generation. This is an estimate from two
designs; the JAX 25-case measurements are actual complete sweeps. Numerical
methods and original pose-solver conventions differ.

Open `results.csv` in a spreadsheet. `objective_m_s` is the screening speed;
`physical_passed` and `violations` explain eligibility. `refinement_status`
distinguishes `passed`, `failed` and `not_checked`; `verified_objective_m_s`
contains the finer result. `selected_best` identifies the exported finalist.

`late_mean_com_speed_m_s` uses the final half of the simulation, seconds 0.5–1.0
in this example. `mean_com_speed_m_s` instead includes startup. Pick the same
objective and horizon when comparing configurations.

The speed map marks physical failures with ×, and the penetration map explains
one common failure. A finite result is not necessarily physically admissible.
Failed cases remain in the report/CSV and never enter the eligible ranking.

The original-drop example produced 25 finite trajectories but all exceeded the
provisional 0.250 mm foot-penetration limit (observed maxima 0.275–0.519 mm).
Consequently it exported no feasible winner. Starting supported produced 25
passing cases under the same limits. This changes the initial condition rather
than silently relaxing a constraint.

## Check finalists and visualize

The runner refines the fastest passing screening cases at both 5 and 2.5 µs.
They must pass all physical checks and the configured speed/penetration-change
tolerances. `verification.top_cases: 3` checks three finalists; use `25` to
refine every passing case. The best verified finalist is not a global optimum.
Unrefined cases have only screening evidence.

For the supported example, case 4 was selected: damping 0.01, adhesive 50 µm,
applied frequency 498 Hz and verified late-half speed 0.803766 m/s. This is a
different experiment from the 1.005 m/s geometry candidate at fixed 363 Hz.

The runner saves `best_design.yaml` and optional `best_case.mat`, plots and
animations. If no finalist qualifies, those best-case files are absent. Full
screening/refinement checks use every integration sample; saved trajectories
use the reduced interval specified by `saved_timestep_s`.

Use `--resume` only with identical settings, sources and backend. Cache
fingerprints reject incompatible results; use a fresh output directory after
changes. A resumed runtime measures the remaining work, not the original run.

## Independently check a selected design in MATLAB

The `.mat` exchange file contains named design inputs, the applied frequency,
parameter vector, initial state and reduced JAX trajectory. `dt_jax` is the
integration step; `output_timestep_s` is the saved interval. Export alone does
not perform a MATLAB verification.

```bash
python tools/check_sweep_matlab.py \
  --design outputs/my_sweep/best_design.yaml \
  --matlab /path/to/MATLAB/bin/matlab \
  --timestep 0.00000125 \
  --output outputs/my_sweep/matlab_check.json
```

MATLAB with Symbolic Math Toolbox is optional. This helper independently derives
the original symbolic balances after substituting the selected scalar design
settings in memory. The archived source is preserved. It checks ten equation
samples and a 0.02-second trajectory with matched initial state, RK4 and trace
regularization. `--duration` and `--timestep` change this short comparison.

The short check does not validate the entire one-second ranking or reproduce
all `ode45` behavior. To validate a selected design, independently run its YAML
parameters in the original full MATLAB script and compare the same objective,
initial conditions, frequency and numerical resolution. Failure diagnostics
remain saved when the strict comparison fails.

For the recorded selected 50 µm design, equation samples agreed to about
1e-17. The strict trajectory comparison used `rtol=1e-7`, `atol=1e-9`:

| Matched RK4 step | Maximum absolute state difference | Strict check |
| ---: | ---: | --- |
| 10 µs | 3.98e-8 | Failed |
| 2.5 µs | 2.45e-9 | Failed |
| 1.25 µs | 6.11e-10 | Passed |

These maxima mix state units; the largest fine-step difference was in angular
velocity. Position differences were below 5e-15 m. The original failed reports
were retained and tolerances were unchanged. This supports the short matched
comparison, not a full-horizon MATLAB certification of the whole grid.
