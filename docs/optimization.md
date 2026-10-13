# Optional NeRD design optimization

An ordinary physical sweep enumerates designs and simulates each directly;
it requires no neural model. NeRD is an optional approximation trained on solver
trajectories using the official NeRD GPT backbone and local trajectory adapters.

With trained weights frozen, a cost such as
`(predicted_mean_speed - target_speed)**2` supplies gradients with respect to
design inputs. Changing the cost changes the search objective; it does not teach
new physics or extend the trained domain. The current expanded model assumes
a supported, zero-phase start and fixed 200 V / 363 Hz actuation.

A proposed cycle samples a broad domain, simulates training data, trains the
surrogate, proposes designs and checks them with the original solver. Large-error
proposals can become new training data, while independent designs remain reserved
for validation and test. Many time samples from a few designs do not cover a large
design domain.

The latest expanded model predicted 1.000 m/s for a design that achieved
0.977 m/s. Another prediction of 1.069 m/s achieved only 0.572 m/s, below the
0.757 m/s baseline, and was rejected. The accepted physical 1.005 m/s candidate
is a different design found by solver search.

The expanded model fails held-out accuracy criteria and remains experimental.
Its unattended optimization gate is disabled. Independent physical verification
remains necessary even when the neural forecast looks plausible.

For training commands and evidence, read [the expanded study](expanded_design.md).
These dependencies and checkpoints are unnecessary for the beginner simulation
and sweep tutorials. Generated model and verification reports stay under
ignored `outputs/`; the optional notebook also works when none are present.
