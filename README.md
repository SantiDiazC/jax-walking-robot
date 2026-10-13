# JAX Walking Robot

Simulate a planar piezoelectric walking robot, compare physical designs and
create plots and animations from editable YAML files. The analytical dynamics
are ported from the MATLAB thesis model included in this repository.

**No AI knowledge, neural network, MATLAB or GPU is required for simulation or
design sweeps.** An optional NeRD research component proposes designs for
verification with the analytical solver.

## Run one design

Use Python 3.11 in a virtual or Conda environment. From the repository root:

```bash
python -m pip install -e ".[analysis,test]"
JAX_PLATFORMS=cpu python examples/run_design.py \
  --design examples/designs/matlab_supported.yaml \
  --output outputs/my_robot
```

Open `diagnostics.png`, `motion.png` and `robot.html` under `outputs/my_robot/`.
The run saves the trajectory, design and `validation.json`, which explains
whether provisional physical limits passed. Edit a YAML value and use a new
output directory for your next design.

See [installation](docs/installation.md) and
[design parameters and units](docs/design_parameters.md).

## Compare 25 designs

This example combines five damping ratios with five adhesive thicknesses:

```bash
python examples/run_sweep.py \
  --config examples/sweeps/supported_25_cases.yaml \
  --output outputs/my_sweep --render-best --export-mat
```

Every design is simulated and screened. Passing cases are ranked by mean COM
speed, and three finalists are rerun at two finer timesteps. Outputs include:

- `results.csv`: all designs, speeds, failures and refinement status.
- `sweep.png`: speed and foot-penetration maps.
- `best_design.yaml`: the fastest finalist that passed refinement.
- `best/`: plots, MP4 and HTML animation with `--render-best`.
- `best_case.mat`: MATLAB exchange data with `--export-mat`.
- `report.json`: detailed evidence, hardware, timing and source fingerprints.

The recorded supported-start run screened 25 designs and refined three
finalists in **117 seconds**, excluding final plotting and animation. All 25
passed screening; the selected finalist reached **0.803766 m/s** during seconds
0.5–1.0. This is the best verified finalist of this grid, not a global optimum.
The frequency follows the original MATLAB rule and varies with thickness.

`matlab_25_cases.yaml` reproduces the original 501 µm initial drop. That sweep
took **52 seconds**, but all 25 cases exceeded the provisional 0.250 mm foot
penetration limit during startup. No feasible best design was exported. The
supported-start example changes initial clearance, not that limit.

Read the [sweep tutorial](docs/sweeps.md) to edit the grid, interpret failures,
refine more cases and independently check a selected design in MATLAB.
The selected design passed a short matched MATLAB/JAX trajectory comparison
at a finer timestep; this check is separate from the full sweep.

Two full one-second MATLAB cases, including original per-design function
generation and `ode45`, took 75 and 79 seconds. Extrapolating gives **about
32 minutes for 25 cases**, excluding plotting and video. The JAX original-drop
sweep took 52 seconds; methods and initialization conventions differ, so this
is a workflow estimate rather than a controlled solver speedup.
See [timing methodology](docs/performance.md).

## Tutorials and documentation

Start with the [tutorial notebooks](tutorials/README.md): one design, a physical
sweep and then the optional neural experiment.

- [Physical model](docs/physical_model.md).
- [Validation and its limits](docs/validation.md).
- [CPU/GPU measurements](docs/performance.md).
- [NeRD optimization](docs/optimization.md).
- [Expanded fixed-actuation study](docs/expanded_design.md): a separate 1.00476 m/s
  physical candidate at fixed 200 V / 363 Hz, checked at two resolutions and four starts.

The expanded NeRD model currently fails held-out accuracy criteria. Its
predictions are experimental and must be checked with the analytical solver.
The physical 1 m/s candidate was found by solver search, not by NeRD.

## Repository layout

```text
src/walking_robot/      Analytical solver, geometry, validation and neural adapters
examples/designs/      Editable single-design YAML files
examples/sweeps/       Editable parameter grids
examples/learning/     Advanced experimental protocols
tutorials/             Beginner-to-advanced notebooks
docs/                  User guides, model explanation and research history
benchmarks/            Reproducible timing tools
tools/                 Verification and research entry points
reference/matlab/      Original MATLAB, helpers and generated reference equations
tests/                 Numerical and workflow checks
outputs/               Generated artifacts, excluded from Git
```

[Source provenance](reference/provenance.json) records the original files and
migration choices. Generated reports, plots, datasets and checkpoints stay in
ignored `outputs/`; reproduce them using the documented commands. NeRD uses a separate checkout
of the [official implementation](https://github.com/NVlabs/neural-robot-dynamics).

Run `python -m pytest -q` for the test suite. Core tests run on CPU; optional
neural tests require the `learning` extra. Software checks do not certify
hardware performance or global optimality.
