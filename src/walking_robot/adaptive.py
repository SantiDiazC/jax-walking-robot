"""Optional SciPy adaptive integration of the unchanged analytical JAX RHS.

The host integrator uses float64 scaled states. DOP853 is explicit; Radau is
implicit and receives a JAX state Jacobian. Contact smoothing is opt-in.
These diagnostic integrations are not themselves JAX-differentiable.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from .dynamics import dynamics_rhs
from .contact_residual import ContactSmoothing,contact_rhs
from .learning_spec import STATE_SCALES


class IntegrationBudgetExceeded(RuntimeError):
    pass


class AdaptiveDynamics:
    def __init__(self,mode='hard',smoothing=ContactSmoothing()):
        if mode not in ['hard','smooth']:
            raise ValueError('Contact mode must be hard or smooth.')
        self.mode = mode
        self.scale = np.asarray(STATE_SCALES)
        scale = jnp.asarray(self.scale)
        law = dynamics_rhs if mode=='hard' else lambda t,x,p:contact_rhs(t,x,p,mode='smooth',smoothing=smoothing)
        def scaled_rhs(t,y,p):
            return law(t,y*scale,p)/scale
        self.rhs = jax.jit(scaled_rhs)
        self.jacobian = jax.jit(jax.jacfwd(scaled_rhs,argnums=1))
        # Directional parameter derivatives avoid materializing a 14x42 tensor.
        def parameter_tangent(t,y,p,dp):
            return jax.vmap(lambda direction:jax.jvp(lambda pp:scaled_rhs(t,y,pp),(p,),(direction,))[1],in_axes=1,out_axes=1)(dp)
        self.parameter_tangent = jax.jit(parameter_tangent)

    def integrate(self,initial,parameters,horizon,*,method='DOP853',rtol=1e-7,max_step=2e-4,
                  sample_dt=5e-6,start_time=0.,max_rhs_calls=300000,max_wall_s=180.,initial_sensitivity=None,
                  parameter_sensitivity=None):
        from scipy.integrate import solve_ivp
        if method not in ['DOP853','Radau'] or any(not np.isfinite(v) or v<=0 for v in [horizon,rtol,max_step,sample_dt,max_wall_s]):
            raise ValueError('Invalid adaptive solver settings.')
        if not np.isfinite(start_time) or type(max_rhs_calls) is not int or max_rhs_calls<1:
            raise ValueError('Invalid phase or RHS evaluation budget.')
        n = round(horizon/sample_dt)
        if n<2 or n%2 or not np.isclose(n*sample_dt,horizon,rtol=0,atol=1e-12):
            raise ValueError('Sampling must divide the horizon into an even number of steps.')
        initial = np.asarray(initial,dtype=float)
        parameters = np.asarray(parameters,dtype=float)
        if initial.shape!=(14,) or parameters.shape!=(42,) or not np.isfinite(initial).all() or not np.isfinite(parameters).all():
            raise ValueError('Expected finite initial state (14,) and parameters (42,).')
        tangent = initial_sensitivity is not None or parameter_sensitivity is not None
        if tangent and (self.mode!='smooth' or method!='DOP853'):
            raise ValueError('Variational sensitivity integration requires smooth contact and DOP853.')
        y0 = initial/self.scale
        p = jnp.asarray(parameters)
        # Warmup is measured separately from integration and its budget.
        warm = time.perf_counter()
        self.rhs(start_time,jnp.asarray(y0),p).block_until_ready()
        if method=='Radau' or tangent:
            self.jacobian(start_time,jnp.asarray(y0),p).block_until_ready()
        if tangent:
            s0,dp = np.asarray(initial_sensitivity),jnp.asarray(parameter_sensitivity)
            if s0.shape!=(14,2) or dp.shape!=(42,2):
                raise ValueError('Expected initial/parameter sensitivities (14,2)/(42,2).')
            self.parameter_tangent(start_time,jnp.asarray(y0),p,dp).block_until_ready()
            y0 = np.concatenate([y0,(s0/self.scale[:,None]).ravel()])
        warmup = time.perf_counter()-warm
        started = time.perf_counter()
        rhs_calls = 0
        furthest_rhs_time = start_time
        def rhs(t,y):
            nonlocal rhs_calls,furthest_rhs_time
            rhs_calls += 1
            furthest_rhs_time = max(furthest_rhs_time,t)
            if rhs_calls>max_rhs_calls or time.perf_counter()-started>max_wall_s:
                raise IntegrationBudgetExceeded('Adaptive integration budget exhausted; no partial success is reported.')
            state = jnp.asarray(y[:14])
            value = np.asarray(self.rhs(t,state,p))
            if not tangent:
                return value
            a = np.asarray(self.jacobian(t,state,p))
            b = np.asarray(self.parameter_tangent(t,state,p,dp))
            return np.concatenate([value,(a@y[14:].reshape(14,2)+b).ravel()])
        def jac(t,y):
            return np.asarray(self.jacobian(t,jnp.asarray(y),p))
        try:
            sol = solve_ivp(rhs,(start_time,start_time+horizon),y0,method=method,rtol=rtol,atol=rtol*.01,
                max_step=max_step,dense_output=True,**({'jac':jac} if method=='Radau' else {}))
        except (IntegrationBudgetExceeded,ValueError,np.linalg.LinAlgError) as error:
            return dict(success=False,message=str(error),rhs_calls=rhs_calls,elapsed_s=time.perf_counter()-started,
                        furthest_rhs_time_s=furthest_rhs_time,warmup_s=warmup,method=method,rtol=rtol,contact_mode=self.mode),None
        info = dict(success=bool(sol.success),message=sol.message,rhs_calls=rhs_calls,nfev=sol.nfev,njev=sol.njev,
            nlu=sol.nlu,accepted_steps=len(sol.t)-1,elapsed_s=time.perf_counter()-started,warmup_s=warmup,
            method=method,rtol=rtol,scaled_atol=rtol*.01,contact_mode=self.mode,
            sampling='Accepted solver nodes plus uniform dense-output samples; no RK-stage or continuous-time guarantee.')
        if not sol.success:
            return info,None
        uniform = np.arange(n+1)*sample_dt
        absolute = start_time+uniform
        sampled = sol.sol(absolute)
        # Include adaptive nodes so a short contact peak between uniform
        # samples is not discarded. Retain uniform samples for comparisons.
        nodes = np.unique(np.concatenate([uniform,np.clip(sol.t-start_time,0,horizon)]))
        checked = sol.sol(start_time+nodes)[:14].T*self.scale
        result = dict(times=uniform,states=sampled[:14].T*self.scale,
            check_times=nodes,check_states=checked)
        if tangent:
            result['sensitivities'] = sampled[14:].T.reshape(-1,14,2)*self.scale[None,:,None]
        return info,result
