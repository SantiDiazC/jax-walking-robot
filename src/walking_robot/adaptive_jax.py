"""Compiled adaptive Dormand-Prince 5(4) integration and forward sensitivities.

The standard embedded RK coefficients implement the Dormand-Prince method.
Every requested output time is an integration boundary, so screening does not
depend on a dense-output interpolant. Rejected steps are never emitted.
Unlike a black-box adaptive ODE wrapper, exhaustion has an explicit status.
The original analytical RHS and optional smooth contact remain unchanged.
"""
import time
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
from .adaptive import AdaptiveDynamics
from .contact_residual import ContactSmoothing
from .geometry import points,point_velocities
from .validation import PhysicalLimits


def physical_quantities(x,p):
    g,v = points(x,p,xp=jnp),point_velocities(x,p,xp=jnp)
    return jnp.stack([jnp.maximum(-g['feet'][:,1],0).max(),jnp.maximum(-g['body'][:,1],0).max(),
        g['body'][:,1].max(),jnp.linalg.norm(v['centers'],axis=-1).max(),
        jnp.linalg.norm(v['feet'],axis=-1).max(),jnp.abs(x[8:12]).max(),jnp.abs(jnp.diff(x[2:6])).max()])


def dopri_step(rhs,t,y,h,first):
    """Fifth-order value, embedded fourth-order error, reusable final RHS."""
    a = first
    b = rhs(t+h/5,y+h*a/5)
    c = rhs(t+3*h/10,y+h*(3*a/40+9*b/40))
    d = rhs(t+4*h/5,y+h*(44*a/45-56*b/15+32*c/9))
    e = rhs(t+8*h/9,y+h*(19372*a/6561-25360*b/2187+64448*c/6561-212*d/729))
    f = rhs(t+h,y+h*(9017*a/3168-355*b/33+46732*c/5247+49*d/176-5103*e/18656))
    value = y+h*(35*a/384+500*c/1113+125*d/192-2187*e/6784+11*f/84)
    last = rhs(t+h,value)
    error = h*(71*a/57600-71*c/16695+71*d/1920-17253*e/339200+22*f/525-last/40)
    return value,error,last


class JaxAdaptiveDynamics:
    def __init__(self,mode='smooth',smoothing=ContactSmoothing()):
        self.base = AdaptiveDynamics(mode,smoothing)
        self.mode = mode
        self.scale = self.base.scale
        self.compiled = {}
        def variational(t,z,p,dp):
            y,s = z[:14],z[14:].reshape(14,2)
            value,linear = jax.linearize(lambda yy,pp:self.base.rhs(t,yy,pp),y,p)
            tangent = jax.vmap(linear,in_axes=(1,1),out_axes=1)(s,dp)
            return jnp.concatenate([value,tangent.ravel()])
        self.variational_rhs = jax.jit(variational)

    def integrate(self,initial,parameters,horizon,*,rtol=1e-8,sample_dt=1e-5,max_step=2e-4,
                  max_rhs_calls=12000000,start_time=0.,initial_sensitivity=None,parameter_sensitivity=None,
                  physical_limits=None):
        if not jax.config.jax_enable_x64:
            raise ValueError('Enable jax_enable_x64 before constructing dynamics inputs.')
        if any(not np.isfinite(v) or v<=0 for v in [horizon,rtol,sample_dt,max_step]) or not np.isfinite(start_time):
            raise ValueError('Invalid integration settings.')
        if type(max_rhs_calls) is not int or max_rhs_calls<7:
            raise ValueError('RHS budget must allow at least one embedded RK step.')
        n = round(horizon/sample_dt)
        if n<2 or n%2 or not np.isclose(n*sample_dt,horizon,rtol=0,atol=1e-12):
            raise ValueError('Sampling must divide the horizon into an even number of steps.')
        x,p = np.asarray(initial),np.asarray(parameters)
        if x.shape!=(14,) or p.shape!=(42,) or not np.isfinite(x).all() or not np.isfinite(p).all():
            raise ValueError('Expected finite initial state (14,) and parameters (42,).')
        tangent = initial_sensitivity is not None or parameter_sensitivity is not None
        if tangent and self.mode!='smooth':
            raise ValueError('Forward variational integration requires smooth contact.')
        z = x/self.scale
        dp = np.zeros((42,2))
        if tangent:
            sx,dp = np.asarray(initial_sensitivity),np.asarray(parameter_sensitivity)
            if sx.shape!=(14,2) or dp.shape!=(42,2) or not np.isfinite(sx).all() or not np.isfinite(dp).all():
                raise ValueError('Expected finite state/parameter sensitivities (14,2)/(42,2).')
            z = np.concatenate([z,(sx/self.scale[:,None]).ravel()])
        key = (horizon,rtol,sample_dt,max_step,max_rhs_calls,tangent)
        if key not in self.compiled:
            attempts_limit = (max_rhs_calls-1)//6
            def integrate(z,p,dp,start,limits):
                rhs = (lambda t,y:self.variational_rhs(t,y,p,dp)) if tangent else (lambda t,y:self.base.rhs(t,y,p))
                first = rhs(start,z)
                # carry: state, time, next dt, reusable RHS, attempts, accepts,
                # success. The budget is global across all output intervals.
                peaks = physical_quantities(z[:14]*jnp.asarray(self.scale),p)
                bad_time = jnp.where(jnp.any((peaks>limits)|~jnp.isfinite(peaks)),start,jnp.nan)
                carry = (z,start,jnp.minimum(sample_dt,max_step),first,jnp.int64(0),jnp.int64(0),jnp.bool_(True),peaks,bad_time)
                def output(carry,target):
                    def condition(v):
                        _,t,_,_,attempts,_,valid,_,_ = v
                        return valid & (t<target) & (attempts<attempts_limit)
                    def advance(v):
                        y,t,h,first,attempts,accepted,valid,peaks,bad_time = v
                        h = jnp.minimum(h,target-t)
                        value,error,last = dopri_step(rhs,t,y,h,first)
                        denominator = rtol*.01+rtol*jnp.maximum(jnp.abs(y),jnp.abs(value))
                        ratio = jnp.sqrt(jnp.mean((error/denominator)**2))
                        finite = jnp.isfinite(ratio)&jnp.all(jnp.isfinite(value))&jnp.all(jnp.isfinite(last))
                        take = finite & (ratio<=1.)
                        factor = jnp.clip(.9*jnp.maximum(ratio,1e-16)**(-.2),.2,5.)
                        h_new = jnp.minimum(h*factor,max_step)
                        t_new = jnp.where(take,t+h,t)
                        valid = valid & finite & ((t_new>=target)|(h_new>1e-14))
                        current = jax.lax.cond(take,lambda:physical_quantities(value[:14]*jnp.asarray(self.scale),p),lambda:peaks)
                        bad_time = jnp.where(jnp.isnan(bad_time)&take&jnp.any((current>limits)|~jnp.isfinite(current)),t_new,bad_time)
                        peaks = jnp.maximum(peaks,current)
                        return (jnp.where(take,value,y),t_new,h_new,jnp.where(take,last,first),attempts+1,
                                accepted+take.astype(jnp.int64),valid,peaks,bad_time)
                    carry = jax.lax.while_loop(condition,advance,carry)
                    y,t,h,first,attempts,accepted,valid,peaks,bad_time = carry
                    valid = valid & (t>=target)
                    carry = (y,t,h,first,attempts,accepted,valid,peaks,bad_time)
                    return carry,jnp.where(valid,y,jnp.full_like(y,jnp.nan))
                targets = start+jnp.arange(1,n+1)*sample_dt
                carry,path = jax.lax.scan(output,carry,targets)
                return jnp.concatenate([z[None],path]),carry[1],carry[4],carry[5],carry[6],carry[7],carry[8]
            self.compiled[key] = jax.jit(integrate)
        run = self.compiled[key]
        limits = asdict(physical_limits or PhysicalLimits())
        args = (jnp.asarray(z),jnp.asarray(p),jnp.asarray(dp),jnp.asarray(start_time),jnp.asarray(list(limits.values())))
        warm = time.perf_counter()
        executable = run.lower(*args).compile()
        compilation = time.perf_counter()-warm
        started = time.perf_counter()
        path,t,attempts,accepted,valid,peaks,bad_time = jax.tree.map(np.asarray,executable(*args))
        elapsed = time.perf_counter()-started
        success = bool(valid and np.isfinite(path).all())
        info = dict(success=success,method='compiled_dopri5',contact_mode=self.mode,rtol=rtol,scaled_atol=rtol*.01,
            rhs_calls=int(1+6*attempts),accepted_steps=int(accepted),rejected_steps=int(attempts-accepted),
            elapsed_s=elapsed,compilation_s=compilation,final_accepted_time_s=float(t),max_rhs_calls=max_rhs_calls,
            message='Completed all requested integration boundaries.' if success else 'Budget, nonfinite state, or step-size failure; no partial success.',
            sampling='Every internal accepted step is physically screened; uniform outputs are exact integration boundaries.',
            accepted_step_physical_validation=dict(passed=bool(success and np.isnan(bad_time)),metrics=dict(zip(limits,map(float,peaks))),
                first_violation_time_s=None if np.isnan(bad_time) else float(bad_time),limits=limits))
        if not success:
            return info,None
        times = np.arange(n+1)*sample_dt
        result = dict(times=times,states=path[:,:14]*self.scale)
        if tangent:
            result['sensitivities'] = path[:,14:].reshape(-1,14,2)*self.scale[None,:,None]
        return info,result
