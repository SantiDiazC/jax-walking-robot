"""Experimental JAX contact residual with the original analytical mechanics.

The small shared foot network predicts bounded, positive normal-force and
friction-coefficient multipliers. It cannot supply a free acceleration or an
airborne contact force. Hard MATLAB contact remains the default reference;
the C1 contact approximation is explicitly selected for sensitivity studies.
This comparator is independent of the official NeRD Transformer implementation.
"""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from .contact import computeM, contact_forces, independent_force
from .geometry import points,point_velocities


@dataclass(frozen=True)
class ContactSmoothing:
    gap_width_m: float = 2e-6
    velocity_width_m_s: float = 1e-4
    force_width_n: float = 1e-6

    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for v in [self.gap_width_m,self.velocity_width_m_s,self.force_width_n]):
            raise ValueError("Contact smoothing widths must be finite and positive.")


def positive_part_c1(value,width):
    """Exact zero for value<=0, exact identity for value>=width, C1 joins."""
    z = jnp.clip(value/width,0.,1.)
    transition = width*(2*z*z-z*z*z)
    return jnp.where(value>=width,value,transition)


def smooth_contact_forces(x,p,smoothing=ContactSmoothing(),*,normal_scale=None,friction_scale=None):
    feet = points(x,p,xp=jnp)['feet']
    velocity = point_velocities(x,p,xp=jnp)['feet']
    stiffness = jnp.array([p[26],jnp.where(p[41]>0,p[41],p[26])])
    depth = jnp.clip(-feet[:,1]/smoothing.gap_width_m,0.,1.)
    active_gap = depth*depth*(3-2*depth)
    normal = active_gap*positive_part_c1(-stiffness*feet[:,1]-p[27]*velocity[:,1],smoothing.force_width_n)
    if normal_scale is not None:
        normal = normal*normal_scale
    vx = velocity[:,0]
    backward = .5*(1-jnp.tanh(vx/smoothing.velocity_width_m_s))
    mu_k = p[28]+backward*(p[29]-p[28])
    mu_s = p[30]+backward*(p[31]-p[30])
    if friction_scale is not None:
        mu_k,mu_s = mu_k*friction_scale,mu_s*friction_scale
    damping = p[39]+backward*(p[40]-p[39])
    speed_abs = jnp.sqrt(vx*vx+(smoothing.velocity_width_m_s*.01)**2)
    mu = mu_k+(mu_s-mu_k)*jnp.exp(-(speed_abs/p[33])**p[34])
    spring = p[35]*stiffness*x[12:]+p[36]*p[27]*vx
    spring_abs = jnp.sqrt(spring*spring+(smoothing.force_width_n*.01)**2)
    stick = jax.nn.sigmoid((p[32]-speed_abs)/smoothing.velocity_width_m_s)*jax.nn.sigmoid(
        (mu_s*normal-spring_abs)/smoothing.force_width_n)
    n = jnp.clip(normal/smoothing.force_width_n,0.,1.)
    active_normal = n*n*(3-2*n)
    slip = -(mu*normal*jnp.tanh(vx/p[32])+damping*vx)
    tangent = active_normal*(stick*(-spring)+(1-stick)*slip)
    sdot = active_normal*(stick*vx-(1-stick)*p[37]*x[12:])-(1-active_normal)*p[38]*x[12:]
    return jnp.stack([tangent,normal],axis=-1),sdot


def contact_features(t,x,p):
    feet = points(x,p,xp=jnp)['feet']
    velocity = point_velocities(x,p,xp=jnp)['feet']
    ky = jnp.array([p[26],jnp.where(p[41]>0,p[41],p[26])])
    phase = 2*jnp.pi*p[22]*t
    global_features = jnp.concatenate([jnp.array([p[:4].sum()/1e-4,p[19]/1e-4,p[20]/1e-7,p[21]/1e-5]),
        jnp.diff(x[2:6])/.2,jnp.array([jnp.sin(phase),jnp.cos(phase)])])
    angle = x[jnp.array([2,5])]
    local = jnp.stack([feet[:,1]/1e-4,velocity[:,0]/.5,velocity[:,1]/.5,x[12:]/1e-7,ky/20,
        jnp.sin(p[15:17]),jnp.cos(p[15:17]),jnp.sin(angle),jnp.cos(angle),jnp.array([-1.,1.])],axis=-1)
    return jnp.concatenate([local,jnp.broadcast_to(global_features,(2,len(global_features)))],axis=-1)


def initialize_contact_network(seed=42,width=32):
    keys = jax.random.split(jax.random.PRNGKey(seed),2)
    return dict(w1=jax.random.normal(keys[0],(19,width),dtype=jnp.float64)*math.sqrt(2/(19+width)),
        b1=jnp.zeros(width,dtype=jnp.float64),
        w2=jax.random.normal(keys[1],(width,width),dtype=jnp.float64)*math.sqrt(1/width),
        b2=jnp.zeros(width,dtype=jnp.float64),w3=jnp.zeros((width,2),dtype=jnp.float64),b3=jnp.zeros(2,dtype=jnp.float64))


def contact_multipliers(weights,t,x,p,max_log_correction=.25):
    f = contact_features(t,x,p)
    hidden = jnp.tanh(f@weights['w1']+weights['b1'])
    hidden = jnp.tanh(hidden@weights['w2']+weights['b2'])
    return jnp.exp(max_log_correction*jnp.tanh(hidden@weights['w3']+weights['b3']))


def contact_rhs(t,x,p,weights=None,*,mode='hard',smoothing=ContactSmoothing(),max_log_correction=.25):
    scales = contact_multipliers(weights,t,x,p,max_log_correction) if weights is not None else None
    kwargs = {} if scales is None else dict(normal_scale=scales[:,0],friction_scale=scales[:,1])
    if mode=='hard':
        force,memory = contact_forces(x,p,**kwargs)
    elif mode=='smooth':
        force,memory = smooth_contact_forces(x,p,smoothing,**kwargs)
    else:
        raise ValueError('Contact mode must be hard or smooth.')
    matrix = computeM(t,x,p)
    rhs = independent_force(t,x,p,forces=force)
    lam = 1e-9*jnp.trace(matrix)/12
    acceleration = jnp.linalg.solve(matrix+lam*jnp.eye(12,dtype=x.dtype),rhs)[:6]
    return jnp.concatenate([x[6:12],acceleration,memory])


def contact_step(x,t,p,dt,weights=None,**kwargs):
    rhs = lambda tt,xx:contact_rhs(tt,xx,p,weights,**kwargs)
    a = rhs(t,x)
    b = rhs(t+dt/2,x+dt*a/2)
    c = rhs(t+dt/2,x+dt*b/2)
    d = rhs(t+dt,x+dt*c)
    return x+dt*(a+2*b+2*c+d)/6


def contact_rollout(x,p,dt,steps,weights=None,*,start_time=0.,substeps=1,return_substeps=False,**kwargs):
    """Return elapsed times and macro states, or every internal RK4 substep."""
    if type(steps) is not int or steps<1 or type(substeps) is not int or substeps<1:
        raise ValueError('steps and substeps must be positive static integers.')
    def macro(state,index):
        time = start_time+index*dt
        def micro(j,current):
            return contact_step(current,time+j*dt/substeps,p,dt/substeps,weights,**kwargs)
        if return_substeps:
            def scan_micro(current,j):
                nxt = micro(j,current)
                return nxt,nxt
            nxt,path = jax.lax.scan(scan_micro,state,jnp.arange(substeps))
            return nxt,path
        nxt = jax.lax.fori_loop(0,substeps,micro,state)
        return nxt,nxt
    _,tail = jax.lax.scan(macro,x,jnp.arange(steps))
    if return_substeps:
        return jnp.arange(steps*substeps+1)*(dt/substeps),jnp.concatenate([x[None],tail.reshape((-1,14))])
    return jnp.arange(steps+1)*dt,jnp.concatenate([x[None],tail])
