"""Fixed-step single and batched trajectories."""
from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax
from .dynamics import dynamics_rhs

def dynamics_step(x: jnp.ndarray, t: jnp.ndarray, P: jnp.ndarray, dt: float = 1e-3, method: str = "rk4") -> jnp.ndarray:
    """One fixed-step integrator step."""
    dt = jnp.asarray(dt, dtype=x.dtype)

    if method == "euler":
        return x + dt * dynamics_rhs(t, x, P)

    if method != "rk4":
        raise ValueError(f"Unknown method: {method}")

    k1 = dynamics_rhs(t, x, P)
    k2 = dynamics_rhs(t + dt / 2.0, x + dt * k1 / 2.0, P)
    k3 = dynamics_rhs(t + dt / 2.0, x + dt * k2 / 2.0, P)
    k4 = dynamics_rhs(t + dt, x + dt * k3, P)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rollout_scan(
    x0: jnp.ndarray,
    P: jnp.ndarray,
    dt: float = 1e-3,
    steps: int = 1000,
    method: str = "rk4",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Roll out a trajectory with jax.lax.scan.

    Returns:
        ts: (steps+1,)
        xs: (steps+1, 14)
    """
    dt = float(dt)
    ts = jnp.arange(steps + 1, dtype=x0.dtype) * dt

    def body(carry, i):
        x = carry
        t = ts[i]
        x_next = dynamics_step(x, t, P, dt=dt, method=method)
        return x_next, x_next

    xT, xs_tail = lax.scan(body, x0, jnp.arange(steps))
    xs = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    return ts, xs


def rollout_batched(
    x0_batch: jnp.ndarray,
    P_batch: jnp.ndarray,
    dt: float = 1e-3,
    steps: int = 1000,
    method: str = "rk4",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched rollout via vmap.

    x0_batch: (B, 14)
    P_batch:  (B, 42)
    Returns:
        ts: (steps+1,)
        xs: (B, steps+1, 14)
    """
    vm = jax.vmap(lambda x0, P: rollout_scan(x0, P, dt=dt, steps=steps, method=method)[1], in_axes=(0, 0))
    xs = vm(x0_batch, P_batch)
    ts = jnp.arange(steps + 1, dtype=x0_batch.dtype) * float(dt)
    return ts, xs
