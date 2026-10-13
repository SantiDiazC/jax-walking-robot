# Reading validation results

Reports check finite states, maximum foot/body penetration, body height,
center/foot/angular speeds and relative body angles. `passed: true` means the
recorded trajectory stayed within configured provisional limits, not that
hardware performance has been established.

A finite trajectory can still violate a limit. A failed trajectory's finite-prefix
speed never enters the sweep ranking. Every integration sample is checked before
states are reduced for plots, animation or MATLAB exchange.

The sweep ranks passing screening cases and refines a configured number of
finalists. A finalist must pass all physical screens and the speed/penetration
change thresholds at the two finest timesteps. This is an empirical convergence
check, not a rigorous error bound.

`best_design.yaml` is the best passing, resolution-checked finalist. It is absent
if none qualifies. Other cases retain screening evidence unless `top_cases` is
increased. New starts, longer horizons and hardware require further checks.

MATLAB comparisons independently derive original symbolic balances and compare
equations and matched RK4 trajectories. The default sweep double-check covers
0.02 seconds; it does not replace full-horizon screening. Matching RK4 separates
equation differences from `ode45` choices, not every numerical/contact convention.

Generated reports under ignored `outputs/` retain successes and failures. Neural prediction
accuracy, physical plausibility, gradient usefulness and verified improvement
are separate criteria. Plausible neural motion can still have the wrong speed.
