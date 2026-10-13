"""Contact independence, force balance and compatibility with archived exports."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, computeF, computeSdot
from walking_robot import _generated
from walking_robot.contact import contact_forces, independent_force
from walking_robot.design import RobotDesign, DesignRun, SimulationSettings
from walking_robot.geometry import points
from walking_robot.workflow import run_design


def test_reconciled_parameters_match_independent_matlab_audit():
    p = np.asarray(RobotDesign(profile="matlab").parameters())
    np.testing.assert_allclose(p[[17,18,21,26,41]],
                               [.0028,.0025,8.305324281715295e-6,14.,19.],rtol=1e-12,atol=1e-15)


def test_independent_foot_stiffness_and_detachment():
    p = RobotDesign(profile="matlab").parameters()
    x = jax.jit(solve_initial_pose)(p).at[1].add(-1e-4).at[6].set(.02)
    evaluate = jax.jit(contact_forces)
    force,_ = evaluate(x,p)
    modified,_ = evaluate(x,p.at[41].multiply(2))
    np.testing.assert_allclose(force[0],modified[0],atol=1e-14)
    np.testing.assert_allclose(modified[1,1],2*force[1,1],rtol=1e-10)
    separating = x.at[7].set(100.).at[12:].set(jnp.array([1e-6,-1e-6]))
    force, memory = evaluate(separating,p)
    np.testing.assert_array_equal(force,0.)
    np.testing.assert_allclose(memory,-p[38]*separating[12:])


def test_explicit_force_balance_recovers_legacy_algebra():
    p = calculate_P_from_meta_jax(.01,1e-6)
    pose = jax.jit(solve_initial_pose)(p)
    explicit = jax.jit(independent_force)
    exported = jax.jit(_generated.computeF)
    selected = jax.jit(computeF)
    memory = jax.jit(computeSdot)
    original_memory = jax.jit(_generated.computeSdot)
    # Nonzero rates exercise centripetal forces, torque arms and stick/slip.
    for vx in [-.02,-.0002,0.,.0002,.02]:
        for height in [-.0001,.01]:
            x = pose.at[1].add(height).at[6].set(vx).at[8:12].set(jnp.array([2.,-3.,1.,4.]))
            expected = exported(.001,x,p)
            np.testing.assert_allclose(explicit(.001,x,p),expected,rtol=1e-10,atol=1e-14)
            np.testing.assert_array_equal(selected(.001,x,p),expected)
            np.testing.assert_array_equal(memory(.001,x,p),original_memory(.001,x,p))


def test_render_only_rejects_a_different_initial_drop(tmp_path):
    configuration = DesignRun(simulation=SimulationSettings(duration_s=.0002,timestep_s=.0001))
    p = configuration.design.parameters()
    state = np.asarray(jax.jit(solve_initial_pose)(p))
    np.savez_compressed(tmp_path/'trajectory.npz',t=np.arange(3)*1e-4,x=np.tile(state,(3,1)),P=p)
    configuration = replace(configuration,simulation=replace(configuration.simulation,initial_clearance_um=501.))
    with pytest.raises(ValueError,match='different design or simulation'):
        run_design(configuration,tmp_path,render_only=True)
