"""Check a local voltage sensitivity; this does not perform design optimization.

Hard contact switches and rounded natural frequency make the model piecewise
differentiable. This short-horizon example keeps the initial state fixed and
checks the gradient against finite differences at the chosen operating point.
"""
import jax
import numpy as np
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, rollout_scan


def main():
    jax.config.update("jax_enable_x64", True)
    p = calculate_P_from_meta_jax(0.01, 1e-6, 200.)
    x0 = jax.jit(solve_initial_pose)(p)

    def displacement(voltage):
        parameters = calculate_P_from_meta_jax(0.01, 1e-6, voltage)
        _, states = rollout_scan(x0, parameters, dt=1e-5, steps=10)
        return states[-1, 0] - x0[0]

    value = jax.jit(displacement)
    derivative = jax.jit(jax.grad(displacement))(200.)
    epsilon = 0.01
    difference = (value(200. + epsilon) - value(200. - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(derivative, difference, rtol=5e-3, atol=1e-14)
    print(f"Local dx/dV: autodiff={float(derivative):.8g}, finite difference={float(difference):.8g} m/V")


if __name__ == "__main__":
    main()
