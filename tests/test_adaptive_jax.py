from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from walking_robot import solve_initial_pose
from walking_robot.adaptive_jax import JaxAdaptiveDynamics,dopri_step
from walking_robot.adaptive import AdaptiveDynamics
from walking_robot.design import RobotDesign
from walking_robot.geometry import points
from walking_robot.validation import PhysicalLimits,inspect_trajectory


def test_dormand_prince_integrates_quartic_time_forcing_exactly():
    rhs = lambda t,y:jnp.full_like(y,t**4)
    x = jnp.array([.3,.5])
    value,error,last = dopri_step(rhs,.2,x,.1,rhs(.2,x))
    np.testing.assert_allclose(value,x+(.3**5-.2**5)/5,rtol=1e-13,atol=1e-15)
    assert np.isfinite(error).all() and float(jnp.linalg.norm(error))>0
    np.testing.assert_allclose(last,rhs(.3,value),atol=1e-15)


def test_compiled_adaptive_budget_failure_has_no_partial_trajectory():
    with jax.default_device(jax.devices('cpu')[0]):
        p = RobotDesign(profile='matlab').parameters()
        x = jax.jit(solve_initial_pose)(p)
        info,result = JaxAdaptiveDynamics().integrate(x,p,.0002,max_rhs_calls=7)
        assert not info['success'] and result is None
        assert info['rhs_calls']<=7
        assert not info['accepted_step_physical_validation']['passed']


def test_compiled_adaptive_free_fall_and_custom_physical_limits():
    with jax.default_device(jax.devices('cpu')[0]):
        p = RobotDesign(profile='matlab',voltage_v=0.).parameters()
        x = jax.jit(solve_initial_pose)(p).at[1].add(.005).at[6].set(.02)
        limits = replace(PhysicalLimits(),max_body_height_m=.004)
        info,result = JaxAdaptiveDynamics().integrate(x,p,.0002,physical_limits=limits)
        assert info['success'] and not info['accepted_step_physical_validation']['passed']
        assert info['accepted_step_physical_validation']['first_violation_time_s']==0.
        weights = np.asarray(p[:4]/p[:4].sum())
        com = (points(result['states'],np.asarray(p))['centers']*weights[None,:,None]).sum(1)
        np.testing.assert_allclose(com[:,0]-com[0,0],.02*result['times'],atol=1e-10)
        np.testing.assert_allclose(com[:,1]-com[0,1],-.5*9.81*result['times']**2,atol=1e-10)


def test_compiled_forward_sensitivities_match_independent_scipy_integration():
    pytest.importorskip('scipy')
    with jax.default_device(jax.devices('cpu')[0]):
        base = RobotDesign(profile='matlab')
        def prepare(u):
            p = base.parameters(adhesive_thickness_um=1.+u[0],front_angle_deg=95.+u[1])
            return p,solve_initial_pose(p).at[1].add(-1e-4).at[6].set(.02)
        p,x = jax.jit(prepare)(jnp.zeros(2))
        dp,dx = jax.jit(jax.jacfwd(prepare))(jnp.zeros(2))
        kwargs = dict(rtol=1e-9,sample_dt=1e-5,initial_sensitivity=dx,parameter_sensitivity=dp)
        info,a = JaxAdaptiveDynamics().integrate(x,p,.0002,**kwargs)
        other,b = AdaptiveDynamics('smooth').integrate(x,p,.0002,**kwargs)
        assert info['success'] and other['success']
        np.testing.assert_allclose(a['states'],b['states'],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(a['sensitivities'],b['sensitivities'],rtol=1e-5,atol=1e-8)
        sampled = inspect_trajectory(a['times'],a['states'],p)
        for name,value in sampled['metrics'].items():
            if name.startswith('max_'):
                assert info['accepted_step_physical_validation']['metrics'][name]>=value-1e-12
