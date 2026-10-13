"""Explicit MATLAB contact law with independent front/hind stiffnesses.

P[41] > 0 selects this formulation; P[41] == 0 retains the original exports.
The switches are hard. No contact force or invalid trajectory is smoothed away.
"""
import jax.numpy as jnp
from jax import lax
from . import _generated
from .geometry import points, point_velocities

computeM = _generated.computeM


def contact_forces(x, p, *, normal_scale=None, friction_scale=None):
    geometry = points(x, p, xp=jnp)
    feet, velocity = geometry["feet"], point_velocities(x, p, xp=jnp)["feet"]
    stiffness = jnp.array([p[26], jnp.where(p[41] > 0, p[41], p[26])])
    normal = jnp.where(feet[:, 1] < 0, jnp.maximum(0., -stiffness*feet[:, 1]-p[27]*velocity[:, 1]), 0.)
    if normal_scale is not None:
        normal = normal*normal_scale
    vx = velocity[:, 0]
    backward = vx <= 0
    mu_k = jnp.where(backward, p[29], p[28])
    mu_s = jnp.where(backward, p[31], p[30])
    if friction_scale is not None:
        mu_k,mu_s = mu_k*friction_scale,mu_s*friction_scale
    damping = jnp.where(backward, p[40], p[39])
    mu = mu_k+(mu_s-mu_k)*jnp.exp(-(jnp.abs(vx)/p[33])**p[34])
    spring = p[35]*stiffness*x[12:]+p[36]*p[27]*vx
    stick = (jnp.abs(vx) < p[32]) & (jnp.abs(spring) <= mu_s*normal) & (normal > 0)
    slip = -(mu*normal*jnp.tanh(vx/p[32])+damping*vx)
    tangent = jnp.where(normal > 0, jnp.where(stick, -spring, slip), 0.)
    sdot = jnp.where(normal > 0, jnp.where(stick, vx, -p[37]*x[12:]), -p[38]*x[12:])
    return jnp.stack([tangent, normal], axis=-1), sdot


def independent_force(t, x, p, *, forces=None):
    """Newton/Euler balances, including exact centripetal terms."""
    geometry = points(x, p, xp=jnp)
    if forces is None:
        forces, _ = contact_forces(x, p)
    theta, omega = x[2:6], x[8:12]
    tau = p[19]*(jnp.diff(theta)-p[23:26])+p[20]*jnp.diff(omega)+p[21]*jnp.sin(2*jnp.pi*p[22]*t)
    direction = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    terms = p[9:13, None]*omega[:, None]**2*direction
    coriolis = jnp.stack([jnp.zeros(2, dtype=x.dtype),
                         .5*(terms[0]+terms[1]),
                         .5*terms[0]+terms[1]+.5*terms[2],
                         .5*terms[0]+terms[1]+terms[2]+.5*terms[3]])
    linear = -p[:4, None]*coriolis
    linear = linear.at[:, 1].add(-p[:4]*p[8])
    linear = linear.at[0].add(forces[0]).at[3].add(forces[1])
    lever = geometry["feet"]-geometry["centers"][[0, 3], :]
    moment = lever[:, 0]*forces[:, 1]-lever[:, 1]*forces[:, 0]
    rotational = jnp.array([tau[0]+moment[0], -tau[0]+tau[1], -tau[1]+tau[2], -tau[2]+moment[1]])
    return jnp.concatenate([linear, rotational[:, None]], axis=-1).reshape(12)


def computeF(t, x, p):
    return lax.cond(p[41] > 0, lambda: independent_force(t, x, p), lambda: _generated.computeF(t, x, p))


def computeSdot(t, x, p):
    return lax.cond(p[41] > 0, lambda: contact_forces(x, p)[1], lambda: _generated.computeSdot(t, x, p))
