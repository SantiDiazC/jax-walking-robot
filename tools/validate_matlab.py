"""Compare against MATLAB source: python tools/validate_matlab.py --matlab /path/to/matlab.

Requires MATLAB on PATH (or --matlab) and scipy. Both trajectories use the
same JAX initial pose, RK4 timestep and trace-scaled regularization. This
isolates equation and integrator parity from differences in pose solvers.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, computeM, computeF, computeSdot, rollout_scan


def matlab_string(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    p = calculate_P_from_meta_jax(.01, 1e-6, 200.)
    x0 = jax.jit(solve_initial_pose)(p)
    states = np.tile(np.asarray(x0), (6, 1))
    # Flight, contact, forward/backward slip and small contact velocities.
    states[:, 1] += [0.01, -1e-4, -1e-4, -1e-4, -1e-4, -1e-4]
    states[:, 6] = [0., 0., .02, -.02, .0002, -.0002]
    times = np.linspace(0., .001, len(states))
    evaluate = jax.jit(lambda t, x, p: (computeM(t, x, p), computeF(t, x, p), computeSdot(t, x, p)))
    values = [evaluate(t, jnp.asarray(x), p) for t, x in zip(times, states)]
    mass, forces, friction = [np.stack(group) for group in zip(*values)]
    print("JAX equation samples evaluated.", flush=True)
    _, trajectory = jax.jit(lambda x, p: rollout_scan(x, p, dt=1e-5, steps=100))(x0, p)
    tools_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="walking-robot-") as folder:
        input_file, output_file = Path(folder)/"input.mat", Path(folder)/"output.mat"
        savemat(input_file, dict(P=np.asarray(p), x0=np.asarray(x0), states=states, times=times, dt=1e-5, steps=100))
        command = f"addpath({matlab_string(tools_dir)}); matlab_reference({matlab_string(input_file)}, {matlab_string(output_file)})"
        print("Running MATLAB reference...", flush=True)
        subprocess.run([args.matlab, "-batch", command], check=True)
        result = loadmat(output_file)
        for name, actual in [("M", mass), ("F", forces), ("S", friction), ("trajectory", trajectory)]:
            expected = result[name]
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)
            print(f"{name}: max absolute difference {np.max(np.abs(np.asarray(actual) - expected)):.3g}")


if __name__ == "__main__":
    main()
