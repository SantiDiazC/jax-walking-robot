# CPU, GPU and compilation

JAX runs on CPU or CUDA GPU. `jax.jit` compiles for the selected backend; it does
not mean GPU execution. An environment named `jaxgpu` can run either. NeRD
training used the RTX 3080; recent analytical sweeps deliberately used CPU.

This solver advances a small 14-state system with sequential time steps and
12 × 12 linear solves. For the measured implementation, small GPU operations
cost more than the available parallelism saves. JAX documents this effect for
[small workloads](https://docs.jax.dev/en/latest/benchmarking.html).
Batching independent designs improves throughput, but must be measured rather
than assumed to make GPU faster.

## Recorded short benchmark

Same designs, float64, 1,000 RK4 steps of 10 µs per design (0.01 simulated
seconds), five timed repeats after compilation. Each process had a separate
four-logical-CPU affinity; CPU and GPU initialization was outside warm timing.

| Batch | CPU mean execution | RTX 3080 mean execution |
| ---: | ---: | ---: |
| 1 design | 0.0402 s | 0.8091 s |
| 25 designs | 0.2127 s | 1.1477 s |

These are short workload timings, not two-second runtimes. They exclude host
transfer, plotting and MATLAB. Compilation plus first execution was measured
separately, and all states were finite. Commands below generate measurements and source fingerprints under ignored
`outputs/`; raw timing reports are not part of the published source tree.

```bash
JAX_PLATFORMS=cpu python benchmarks/benchmark_sweep.py --repeats 6
JAX_PLATFORMS=cuda python benchmarks/benchmark_sweep.py --repeats 6
```

The benchmark waits for device completion using `block_until_ready()` to avoid
mistaking asynchronous dispatch for execution. The same compiled function
accepts new parameter values at fixed shapes and integration settings. Changing
the batch shape or number of integration steps can require recompilation.

## Full sweeps

### MATLAB estimate from two measured one-second cases

Two corner cases were executed with the original MATLAB helpers, `ode45`,
`RelTol=1e-6`, `AbsTol=1e-8`, original `lsqnonlin` initialization and initial
drop. Numerical preparation includes the original geometry, contact and RHS
function exports. Plotting, movies, GUI and process startup are excluded.

| Damping | Adhesive | Numerical preparation | `ode45` integration | Per-case total |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 1 µm | 67.38 s | 7.96 s | 75.34 s |
| 0.09 | 50 µm | 65.20 s | 14.20 s | 79.40 s |

Both completed the full second with finite states. Extrapolating the mean
per-case preparation and integration to 25 cases, with shared symbolic setup
counted once, gives **1,935 seconds, about 32 minutes**. Integration alone
accounts for about **277 seconds, or 4.6 minutes**. Scaling each sampled total
separately gives 31.4–33.1 minutes; this is a two-sample illustration, not a
statistical interval or a guarantee for unsampled designs. Full plotting and
video generation would add work. The original estimate of over an hour can
depend on those steps and the machine used.

Reproduce the measurement from the repository root:

This benchmark requires MATLAB with Symbolic Math Toolbox and Optimization
Toolbox. The recorded measurement used MATLAB R2025b Update 2 on the same
workstation as the JAX studies, with four logical CPUs assigned to the process.

```bash
mkdir -p outputs/matlab_runtime_estimate
matlab -batch "addpath('benchmarks'); matlab_25_case_estimate(fullfile(pwd,'outputs','matlab_runtime_estimate','report.json'),1.0)"
```

The original unregularized linear solve is used here, while the JAX solver
retains its trace regularizer. MATLAB uses adaptive `ode45`; the JAX sweeps below
use fixed-step RK4. These measurements compare workflows and do not establish
a matched numerical-method speedup. The separate matched-RK4 check below isolates
equation/integrator agreement.

### Recorded JAX sweeps

The one-second, 25-case original-drop sweep took 51.69 s including setup,
compilation, physical checks and saved data, before plotting. Every case failed
the provisional penetration limit, so no best case was selected.

The supported-start sweep took 117.08 s including the same work plus two finer
integrations for each of three finalists and MATLAB export. Plotting and movie
rendering followed separately. This is an observed study runtime, not a matched
comparison against the original MATLAB `ode45` script.

The separate two-second fixed-actuation candidate took 32.76 s in JAX/CPU,
including compilation and integration at 1.25 µs. Different durations, steps,
batch sizes and initial states should not be compared as equivalent workloads.

## Short matched MATLAB comparison

For the selected supported 50 µm design, a 0.02-second check with 16,000 RK4
steps at 1.25 µs passed the strict trajectory comparison. Warm JAX/CPU execution
took 0.3465 s; MATLAB trajectory execution took 4.4850 s, about 13 times longer.
This excludes JAX compilation (5.6061 s), MATLAB symbolic/reference setup
(4.9628 s), MATLAB process startup and rendering. It is a short matched-RK4
measurement, not a speedup claim for the full original `ode45` sweep.

All three MATLAB comparison resolutions, including two strict failures, remain
in local generated reports. See [the sweep guide](sweeps.md).
