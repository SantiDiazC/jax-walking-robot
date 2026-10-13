import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

from walking_robot import (
    calculate_P_from_meta_jax, solve_initial_pose, computeM, computeF, computeSdot,
    dynamics_rhs, dynamics_step, rollout_scan, rollout_batched,
)


def feet(q, p):
    """Independent NumPy geometry from the MATLAB pose constraints."""
    x, y, a, b, c, d = q
    direction = lambda angle: np.array([np.cos(angle), np.sin(angle)])
    cg1 = np.array([x, y])
    cg4 = cg1 - p[9] / 2 * direction(a) - p[10] * direction(b) - p[11] * direction(c) - p[12] / 2 * direction(d)
    front = cg1 + (p[9] / 2 - p[17]) * direction(a) + p[13] * direction(a - p[15])
    hind = cg4 - (p[12] / 2 - p[18]) * direction(d) + p[14] * direction(d - p[16])
    return front, hind


@pytest.fixture(scope="module")
def design():
    p = calculate_P_from_meta_jax(0.01, 1e-6, 200.)
    return p, jax.jit(solve_initial_pose)(p)


@pytest.mark.parametrize("zeta,had,volt", [(0.01, 1e-6, 200.), (0.02, 2e-6, 150.), (0.005, 0.5e-6, 230.)])
def test_initial_pose_against_independent_root(zeta, had, volt):
    p = calculate_P_from_meta_jax(zeta, had, volt)
    x0 = np.asarray(jax.jit(solve_initial_pose)(p))
    p = np.asarray(p)

    def residual(q):
        front, hind = feet(q, p)
        return np.r_[front[1], hind[1], front[0], np.diff(q[2:6]) - p[23:26]]

    guess = np.r_[0., .003, 0., np.cumsum(p[23:26])]
    expected = least_squares(residual, guess, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    np.testing.assert_allclose(x0[:6], expected, rtol=1e-5, atol=2e-7)
    np.testing.assert_allclose(residual(x0[:6]), 0., atol=1e-8)
    np.testing.assert_array_equal(x0[6:], 0.)


def test_free_fall_and_internal_force_balance(design):
    p, x = design
    x = x.at[1].set(0.02)  # Both feet above ground, zero velocities.
    rhs = np.asarray(jax.jit(dynamics_rhs)(0., x, p))
    mass, forces = jax.jit(lambda t, x, p: (computeM(t, x, p), computeF(t, x, p)))(0., x, p)
    mass, forces = np.asarray(mass), np.asarray(forces)
    expected = [0., -9.81, 0., 0., 0., 0.]
    np.testing.assert_allclose(np.linalg.solve(mass, forces)[:6], expected, atol=1e-8)
    # Preserve the legacy trace-scaled solve, which slightly perturbs free fall.
    np.testing.assert_allclose(rhs[6:12], expected, atol=2e-5)
    np.testing.assert_allclose(mass @ np.linalg.solve(mass, forces), forces, atol=1e-15)


def test_detached_friction_memory_decays(design):
    p, x = design
    x = x.at[1].set(.02).at[12:].set(jnp.array([2e-5, -3e-5]))
    sdot = jax.jit(computeSdot)(0., x, p)
    np.testing.assert_allclose(sdot, -p[38] * x[12:], atol=1e-15)


def test_batch_matches_scalar_and_parameters_are_dynamic(design):
    p, x = design
    # Isolate vectorization from hard-contact boundary choices: the supported
    # pose is at y=0 to roundoff, which can select different friction branches
    # after CPU XLA fusion. Contact laws have separate dedicated checks.
    x = x.at[1].add(.001)
    ps = jnp.stack([p, p.at[21].multiply(0.5)])
    xs = jnp.stack([x, x])
    batch = jax.jit(lambda xs, ps: rollout_batched(xs, ps, dt=1e-5, steps=30))
    scalar = jax.jit(lambda x, p: rollout_scan(x, p, dt=1e-5, steps=30))
    ts, result = batch(xs, ps)
    assert np.isfinite(result).all()
    for i in range(2):
        scalar_t, expected = scalar(xs[i], ps[i])
        np.testing.assert_array_equal(ts, scalar_t)
        np.testing.assert_allclose(result[i], expected, rtol=1e-7, atol=1e-9)
    assert not np.allclose(result[0, -1, 8:12], result[1, -1, 8:12])


def test_voltage_gradient_matches_finite_difference(design):
    _, x = design

    def objective(v):
        p = calculate_P_from_meta_jax(.01, 1e-6, v)
        return rollout_scan(x, p, dt=1e-5, steps=10)[1][-1, 0]

    value = jax.jit(objective)
    actual = jax.jit(jax.grad(objective))(200.)
    expected = (value(200.01) - value(199.99)) / .02
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-14)


def test_zero_steps_and_unknown_method(design):
    p, x = design
    ts, xs = rollout_scan(x, p, steps=0)
    np.testing.assert_array_equal(xs[0], x)
    assert ts.shape == (1,)
    with pytest.raises(ValueError, match="Unknown method"):
        dynamics_step(x, 0., p, method="invalid")
