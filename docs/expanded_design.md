# Fixed-actuation design search

This search changes adhesive thickness and both leg rest angles while holding
the drive at **200 V and 363 Hz**. Explicit frequency matters: the original map
otherwise recomputes a rounded frequency for each design. `ExpandedRobotDesign`
and its YAML runner preserve the original interface and checkpoint fingerprints.

## Run the accepted design

From the repository root, with the package and its `examples` extra installed:

```bash
JAX_PLATFORMS=cpu python tools/run_expanded_design.py \
  examples/designs/fixed_actuation_1m_s.yaml \
  --output outputs/fixed_actuation_example
```

Edit the YAML to change geometry, duration, resolution, provisional limits or
movie settings. Outputs include `trajectory.npz`, `validation.json`, `run.json`,
PNG figures, `robot.mp4` and self-contained `robot.html`. MP4 requires FFmpeg
with H.264 support. Add `--render-only` to reuse an unchanged saved trajectory.

The original hard-contact simulation gives **1.0047588 m/s during seconds 1–2**,
compared with **0.7568656 m/s** for the baseline: a **32.75%** increase.

| Parameter | Value |
| --- | ---: |
| Adhesive thickness | 2.651043 µm |
| Front rest angle | 109.081771° |
| Hind rest angle | 99.568254° |
| Voltage | 200 V |
| Drive frequency | 363 Hz |

Of 96 one-second screening cases, 83 passed provisional physical limits. Five
finalists and the baseline were checked for two seconds with RK4 timesteps of
2.5 and 1.25 µs. The best passing candidate was frozen before three additional
paired starts with altered clearance, tilt, velocity and actuator phase.
All four starts passed at both resolutions and retained gains above the
empirical timestep uncertainty screen. Candidate speeds span
**1.0047588–1.0048316 m/s**. Maximum foot penetration was **0.242239 mm**, close
to the provisional **0.250 mm** limit.

All samples at both verification resolutions were screened; downsampled plotting
files do not replace those checks. This is evidence under the existing model,
provisional limits and four starts. It does not establish manufacturing validity,
hardware speed, arbitrary-start robustness or a global optimum. The search
neither established 1.2 m/s nor proved that target impossible.

## Repeat the physical search

```bash
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage screen
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage collect-screen
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage verify
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage collect-verify
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage robust
JAX_PLATFORMS=cpu python tools/expanded_design_search.py --stage collect-final --render
python tools/collect_expanded_results.py --output outputs/expanded_design_summary.json
```

The protocol is `examples/learning/expanded_design_search.yaml`. Simulation
stages support independent worker processes, e.g. `--worker 0 --workers 3` and
the other two indices. Finish all workers before collecting each stage.
Source/protocol fingerprints reject incompatible caches; use a new `--output`
directory after changes. Generated evidence, trajectories and movies stay under
ignored `outputs/`.

## Experimental three-variable NeRD adapter

`ExpandedNeRD` uses the unchanged official NeRD GPT backbone for design/time
queries. It predicts COM position, posture and two actuator memory states.
An auxiliary slow-time MLP predicts horizontal COM drift. Analytical geometry
reconstructs the original positions; exact time differentiation supplies
velocities. This avoids amplifying periodic posture error into COM velocity
error. No trajectory, contact state or physical neural output is clipped.
The loss supervises coordinates, COM velocity, signed foot heights and mean
speed. Mixed design/velocity derivatives use the tested elementary LayerNorm
adapter. This remains a supported, zero-phase trajectory surrogate rather than
a general state-to-state dynamics model or an external Newton simulation.

```bash
JAX_PLATFORMS=cpu python tools/generate_expanded_data.py
JAX_PLATFORMS=cpu python tools/generate_expanded_data.py --collect
python tools/train_expanded_nerd.py --nerd-root ../neural-robot-dynamics
JAX_PLATFORMS=cpu python tools/verify_expanded_nerd.py
JAX_PLATFORMS=cpu python tools/verify_expanded_nerd.py --collect
python tools/collect_expanded_results.py --output outputs/expanded_design_summary.json
```

Generation and physical verification support worker partitions. The YAML
`examples/learning/expanded_nerd.yaml` defines 48 two-second teacher trajectories:
32 training, eight validation and eight test designs. Normalization uses
training only; validation selects weights; test evaluation follows selection.
The baseline and physical 1 m/s candidate are test anchors. Complete finite
trajectories retain physical pass/fail labels; failures would remain explicit
constraint examples rather than being repaired or discarded.

With frozen network weights, bounded Adam minimizes
`(predicted_mean_speed - target_speed)**2` for targets 1.0 and 1.2 m/s through
the three design inputs. Bounding an input does not modify a physical state.
Frozen proposals are rerun with original hard contact at both resolutions.
Forecast agreement, physical improvement and target attainment are separate
checks. Those canonical checks alone do not enable unattended optimization;
independent three-variable teacher gradient validation and perturbed-start
acceptance remain necessary.

The inherited material map still rounds some hinge stiffnesses. Their ordinary
AD derivatives are zero within each regime and do not describe jumps across
regime boundaries. Fixed drive frequency removes the frequency jumps from this
study, but does not remove stiffness discontinuities.

The physical candidate above came from the solver search, not from NeRD.
Checkpoints, training history, held-out predictions and proposal checks live
under `outputs/expanded_nerd/`. The collector adds available neural reports,
including failures, to a generated local summary excluded from the source
repository.

### Recorded neural experiment

All 48 teacher trajectories completed numerically and passed the provisional
physical limits. Training ran for 6,000 iterations on an RTX 3080; validation
selected iteration 1,000. On the two test anchors, mean-speed errors relative
to the smooth teacher were 0.000058 m/s for the baseline and 0.001140 m/s for
the physical candidate. Other held-out designs failed accuracy thresholds:
the largest mean-speed error was 0.429885 m/s in validation and 0.040023 m/s
in test. The candidate's COM velocity RMSE was also 0.074358 m/s, above the
0.020 m/s threshold. Both held-out trajectory gates therefore failed.

The frozen network nevertheless produced exploratory loss-minimizing proposals:

| Target (m/s) | Neural prediction (m/s) | Original solver (m/s) | Outcome |
| ---: | ---: | ---: | --- |
| 1.0 | 1.000000 | 0.977482 | Positive canonical gain, forecast and target fail |
| 1.2 | 1.069411 | 0.571520 | Worse than baseline; rejected |

Both physical proposal trajectories passed provisional limits at both
timesteps. Physical plausibility alone therefore does not establish prediction
accuracy or useful design gradients. The first proposal improved the canonical
baseline, but neither proposal is accepted as a reliable neural optimization
result. Neither received the additional perturbed-start acceptance test.

The current expanded model keeps `optimization_enabled: false`. A useful next
training step is to add independently solved counterexamples around the
high-thickness prediction failures, validate new held-out designs and compare
local gradients before expanding a solver-verified trust region. Increasing
training iterations alone did not fix the recorded generalization failure.
Changing actuation is unnecessary for the demonstrated 1 m/s result; whether
it helps reach 1.2 m/s remains a separate experiment.

Verification for this change: **103 repository tests passed**, including the
three-variable Torch/JAX parameter map, initial pose, exact time-derived
velocities and mixed design/velocity derivatives against finite differences.
The YAML runner reproduced the physical candidate speed and completed PNG,
MP4 and HTML rendering.
