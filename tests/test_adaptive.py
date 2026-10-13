import jax
import jax.numpy as jnp
import numpy as np
import pytest
from walking_robot import solve_initial_pose
from walking_robot.design import RobotDesign
from walking_robot.adaptive import AdaptiveDynamics
from walking_robot.geometry import points


@pytest.mark.parametrize('method',['DOP853','Radau'])
def test_adaptive_free_fall_matches_com_motion_and_checks_internal_nodes(method):
    pytest.importorskip('scipy')
    with jax.default_device(jax.devices('cpu')[0]):
        p = RobotDesign(profile='matlab',voltage_v=0.).parameters()
        x = jax.jit(solve_initial_pose)(p).at[1].add(.005).at[6].set(.02)
        solver = AdaptiveDynamics('hard')
        info,result = solver.integrate(x,p,.0002,method=method,rtol=1e-8,sample_dt=1e-5,max_step=2e-5)
        assert info['success']
        weight = np.asarray(p[:4]/p[:4].sum())
        com = (points(result['states'],np.asarray(p))['centers']*weight[None,:,None]).sum(1)
        t = result['times']
        np.testing.assert_allclose(com[:,0]-com[0,0],.02*t,atol=1e-10)
        # Preserve the inherited trace regularization's small force-balance bias.
        np.testing.assert_allclose(com[:,1]-com[0,1],-.5*9.81*t*t,atol=1e-10)
        assert len(result['check_times'])>=len(t)


def test_adaptive_budget_exhaustion_never_returns_a_successful_partial_trajectory():
    pytest.importorskip('scipy')
    with jax.default_device(jax.devices('cpu')[0]):
        p = RobotDesign(profile='matlab').parameters()
        x = jax.jit(solve_initial_pose)(p)
        info,result = AdaptiveDynamics('smooth').integrate(x,p,.0002,sample_dt=1e-5,max_rhs_calls=1)
        assert not info['success'] and result is None
        assert 'budget' in info['message']


def test_variational_contact_sensitivity_matches_short_independent_perturbations():
    pytest.importorskip('scipy')
    with jax.default_device(jax.devices('cpu')[0]):
        base = RobotDesign(profile='matlab')
        def prepare(u):
            p = base.parameters(adhesive_thickness_um=1.+u[0],front_angle_deg=95.+u[1])
            x = solve_initial_pose(p).at[1].add(-1e-4).at[6].set(.02)
            return p,x
        u = jnp.zeros(2)
        p,x = jax.jit(prepare)(u)
        dp,dx = jax.jit(jax.jacfwd(prepare))(u)
        solver = AdaptiveDynamics('smooth')
        kwargs = dict(rtol=1e-9,sample_dt=1e-5,max_step=1e-5)
        info,result = solver.integrate(x,p,.0002,initial_sensitivity=dx,parameter_sensitivity=dp,**kwargs)
        assert info['success']
        for axis in range(2):
            values = []
            for sign in [-1,1]:
                pp,xx = jax.jit(prepare)(u.at[axis].set(sign*1e-4))
                check,path = solver.integrate(xx,pp,.0002,**kwargs)
                assert check['success']
                values.append(path['states'][-1])
            fd = (values[1]-values[0])/(2e-4)
            np.testing.assert_allclose(result['sensitivities'][-1,:,axis],fd,rtol=.003,atol=1e-8)
