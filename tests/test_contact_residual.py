import jax
import jax.numpy as jnp
import numpy as np
import pytest
from walking_robot import solve_initial_pose,dynamics_rhs
from walking_robot.design import RobotDesign
from walking_robot.geometry import point_velocities
from walking_robot.contact import contact_forces,computeM,independent_force
from walking_robot.contact_residual import (positive_part_c1,ContactSmoothing,smooth_contact_forces,
    initialize_contact_network,contact_multipliers,contact_rhs,contact_rollout)


@pytest.fixture(scope='module')
def setup():
    p = RobotDesign(profile='matlab').parameters()
    x = jax.jit(solve_initial_pose)(p)
    return x,p


def test_c1_positive_part_has_exact_detachment_and_matching_join_derivatives():
    width = 1e-6
    values = jnp.array([-width,0.,width,2*width])
    np.testing.assert_allclose(jax.vmap(lambda v:positive_part_c1(v,width))(values),[0,0,width,2*width],atol=1e-15)
    derivative = jax.vmap(jax.grad(lambda v:positive_part_c1(v,width)))(values)
    np.testing.assert_allclose(derivative,[0,0,1,1],atol=1e-12)


def test_identity_network_preserves_reference_rhs_in_contact_and_flight(setup):
    x,p = setup
    weights = initialize_contact_network()
    reference = jax.jit(dynamics_rhs)
    residual = jax.jit(lambda t,x,p:contact_rhs(t,x,p,weights))
    for height in [-1e-4,.001]:
        state = x.at[1].add(height).at[6].set(.02).at[8:12].set(jnp.array([2.,-3.,1.,4.]))
        np.testing.assert_allclose(residual(.001,state,p),reference(.001,state,p),rtol=1e-12,atol=1e-10)


@pytest.mark.parametrize('mode',['hard','smooth'])
def test_bounded_positive_corrections_detach_and_preserve_external_force_balance(setup,mode):
    x,p = setup
    weights = initialize_contact_network()
    weights = dict(weights,b3=jnp.array([1e5,-1e5]))
    state = x.at[1].add(.002).at[12:].set(jnp.array([1e-7,-1e-7]))
    scale = contact_multipliers(weights,.001,state,p)
    assert bool(jnp.all(scale>=jnp.exp(-.25))) and bool(jnp.all(scale<=jnp.exp(.25)))
    law = contact_forces if mode=='hard' else smooth_contact_forces
    force,memory = law(state,p,normal_scale=scale[:,0],friction_scale=scale[:,1])
    np.testing.assert_array_equal(force,0.)
    np.testing.assert_allclose(memory,-p[38]*state[12:])
    def com_velocity(xx):
        velocity = point_velocities(xx,p,xp=jnp)['centers']
        return (velocity*p[:4,None]).sum(0)/p[:4].sum()
    for height in [.002,-1e-4]:
        state = x.at[1].add(height).at[6].set(.02)
        rhs = contact_rhs(.001,state,p,weights,mode=mode)
        _,actual = jax.jvp(com_velocity,(state,),(rhs,))
        scale = contact_multipliers(weights,.001,state,p)
        force,_ = law(state,p,normal_scale=scale[:,0],friction_scale=scale[:,1])
        expected = force.sum(0)/p[:4].sum()+jnp.array([0.,-p[8]])
        # The inherited trace regularizer slightly perturbs force balance. Check
        # its exact contribution instead of hiding it in a broad tolerance.
        matrix = computeM(.001,state,p)
        lam = 1e-9*jnp.trace(matrix)/12
        solution = jnp.linalg.solve(matrix+lam*jnp.eye(12),independent_force(.001,state,p,forces=force))
        correction = lam*jnp.array([solution[jnp.array([0,3,6,9])].sum(),
                                    solution[jnp.array([1,4,7,10])].sum()])/p[:4].sum()
        assert float(jnp.linalg.norm(correction))<2e-4
        np.testing.assert_allclose(actual,expected-correction,rtol=1e-8,atol=1e-8)
        assert bool(jnp.all(force[:,1]>=0))


def test_smooth_sliding_friction_is_dissipative_and_has_finite_zero_speed_derivative(setup):
    x,p = setup
    for vx in [-.02,0.,.02]:
        state = x.at[1].add(-1e-4).at[6].set(vx)
        force,_ = smooth_contact_forces(state,p)
        assert bool(jnp.all(force[:,0]*vx<=1e-15))
    derivative = jax.jacfwd(lambda v:smooth_contact_forces(x.at[1].add(-1e-4).at[6].set(v),p)[0])(0.)
    assert bool(jnp.isfinite(derivative).all())


def test_contact_parameter_autodiff_matches_short_rollout_finite_difference(setup):
    x,p = setup
    x = x.at[1].add(-1e-4).at[6].set(.02)
    weights = initialize_contact_network()
    def objective(value):
        w = dict(weights,b3=weights['b3'].at[0].set(value))
        return contact_rollout(x,p,5e-5,4,w,mode='smooth')[1][-1,0]
    value = jax.jit(objective)
    actual = jax.jit(jax.grad(objective))(0.)
    expected = (value(1e-4)-value(-1e-4))/(2e-4)
    assert np.isfinite(actual) and abs(actual)>1e-10
    np.testing.assert_allclose(actual,expected,rtol=.005,atol=1e-10)


def test_smoothing_settings_reject_zero_width():
    with pytest.raises(ValueError):
        ContactSmoothing(gap_width_m=0.)


def test_all_substeps_preserve_macro_states_and_absolute_actuator_phase(setup):
    x,p = setup
    x = x.at[1].add(-1e-4).at[6].set(.02)
    options = dict(start_time=.0007,substeps=2,mode='smooth')
    macro_t,macro = jax.jit(lambda:contact_rollout(x,p,1e-5,3,**options))()
    micro_t,micro = jax.jit(lambda:contact_rollout(x,p,1e-5,3,return_substeps=True,**options))()
    np.testing.assert_allclose(micro[::2],macro,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(micro_t[::2],macro_t,atol=1e-15)
    assert micro.shape==(7,14)
