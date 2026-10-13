"""Bounded Gauss-Newton initialization, compatible with JIT and autodiff."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax

def solve_initial_pose(P: jnp.ndarray, iters: int = 40, damping: float = 1e-6) -> jnp.ndarray:
    """JAX Gauss-Newton approximation to solveInitialPose_gen.

    This solves for z = [x, y, th1, th2, th3, th4] such that:
      foot1_y == 0, foot2_y == 0, foot1_x == 0,
      (th2-th1)==theta1_init, (th3-th2)==theta2_init, (th4-th3)==theta3_init

    Returns:
        X0: (14,)
    """
    l1, l2, l3, l4 = P[9], P[10], P[11], P[12]
    lleg1, lleg2 = P[13], P[14]
    al1, al2 = P[15], P[16]
    h1, h2 = P[17], P[18]
    theta1_init, theta2_init, theta3_init = P[23], P[24], P[25]

    def residual(z):
        x, y, th1, th2, th3, th4 = z
        cg1 = jnp.array([x, y], dtype=z.dtype)

        pin1 = cg1 + jnp.array([-l1 / 2 * jnp.cos(th1), -l1 / 2 * jnp.sin(th1)], dtype=z.dtype)
        cg2 = pin1 + jnp.array([-l2 / 2 * jnp.cos(th2), -l2 / 2 * jnp.sin(th2)], dtype=z.dtype)
        pin2 = cg2 + jnp.array([-l2 / 2 * jnp.cos(th2), -l2 / 2 * jnp.sin(th2)], dtype=z.dtype)
        cg3 = pin2 + jnp.array([-l3 / 2 * jnp.cos(th3), -l3 / 2 * jnp.sin(th3)], dtype=z.dtype)
        pin3 = cg3 + jnp.array([-l3 / 2 * jnp.cos(th3), -l3 / 2 * jnp.sin(th3)], dtype=z.dtype)
        cg4 = pin3 + jnp.array([-l4 / 2 * jnp.cos(th4), -l4 / 2 * jnp.sin(th4)], dtype=z.dtype)

        joint1 = cg1 + jnp.array([(l1 / 2 - h1) * jnp.cos(th1), (l1 / 2 - h1) * jnp.sin(th1)], dtype=z.dtype)
        joint2 = cg4 + jnp.array([-(l4 / 2 - h2) * jnp.cos(th4), -(l4 / 2 - h2) * jnp.sin(th4)], dtype=z.dtype)

        foot1 = joint1 + jnp.array([lleg1 * jnp.cos(al1 - th1), -lleg1 * jnp.sin(al1 - th1)], dtype=z.dtype)
        foot2 = joint2 + jnp.array([lleg2 * jnp.cos(al2 - th4), -lleg2 * jnp.sin(al2 - th4)], dtype=z.dtype)

        return jnp.array(
            [
                foot1[1],
                foot2[1],
                foot1[0],
                (th2 - th1) - theta1_init,
                (th3 - th2) - theta2_init,
                (th4 - th3) - theta3_init,
            ],
            dtype=z.dtype,
        )

    # Initial guess matches solveInitialPose_gen
    z = jnp.array(
        [0.0, 0.003, 0.0, theta1_init, theta1_init + theta2_init, theta1_init + theta2_init + theta3_init],
        dtype=P.dtype,
    )

    lb = jnp.array([-0.02, 0.0005, -jnp.pi, -jnp.pi, -jnp.pi, -jnp.pi], dtype=P.dtype)
    ub = jnp.array([0.02, 0.01, jnp.pi, jnp.pi, jnp.pi, jnp.pi], dtype=P.dtype)

    def gn_step(z):
        r = residual(z)
        J = jax.jacfwd(residual)(z)  # (6,6)
        A = J.T @ J + damping * jnp.eye(6, dtype=z.dtype)
        b = -(J.T @ r)
        dz = jnp.linalg.solve(A, b)
        z_new = z + dz
        return jnp.clip(z_new, lb, ub)

    z = lax.fori_loop(0, iters, lambda _, zz: gn_step(zz), z)
    X0 = jnp.concatenate([z, jnp.zeros((8,), dtype=P.dtype)], axis=0)
    return X0
