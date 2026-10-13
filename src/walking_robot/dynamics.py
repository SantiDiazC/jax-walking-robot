"""Continuous dynamics and optional contact ramp."""
from __future__ import annotations
import jax.numpy as jnp
from .contact import computeM, computeF, computeSdot

def dynamics_rhs(t: jnp.ndarray, x: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time dynamics: xdot = f(t, x, P)."""
    M = computeM(t, x, P)
    F = computeF(t, x, P)
    sdot = computeSdot(t, x, P)

    lam = 1e-9 * (jnp.trace(M) / M.shape[0])
    sol = jnp.linalg.solve(M + lam*jnp.eye(M.shape[0], dtype=M.dtype), F)
    acc = sol[:6]
    vel = x[6:12]
    return jnp.concatenate([vel, acc, sdot], axis=0)


def ramp_contact_P(P: jnp.ndarray, t: jnp.ndarray, Tr: float = 0.005, soft: float = 0.1) -> jnp.ndarray:
    """Optional 'ramp-contact' trick: scale contact-related parameters at early times.

    This is NOT used by default (to keep exact MATLAB dynamics).
    It mirrors the ramp in run_insect_sweep.py (ky, dy, dx_fwd, dx_bwd).
    """
    r = jnp.minimum(t / Tr, 1.0)
    ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * r))
    scale = soft + (1.0 - soft) * ramp

    P2 = P
    P2 = P2.at[26].set(P[26] * scale)  # ky
    P2 = P2.at[41].set(P[41] * scale)  # hind stiffness (zero in legacy)
    P2 = P2.at[27].set(P[27] * scale)  # dy
    P2 = P2.at[39].set(P[39] * scale)  # dx_fwd
    P2 = P2.at[40].set(P[40] * scale)  # dx_bwd
    return P2
