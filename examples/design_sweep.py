"""Evaluate three designs with the same compiled batch function."""
import jax
import jax.numpy as jnp
import numpy as np
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, rollout_batched


def main():
    jax.config.update("jax_enable_x64", True)
    # Rows: damping ratio, adhesive thickness (m), voltage (V).
    designs = jnp.array([[0.01, 1e-6, 200.], [0.02, 2e-6, 150.], [0.005, 0.5e-6, 230.]])
    parameters = jax.jit(jax.vmap(calculate_P_from_meta_jax))(*designs.T)
    initial_states = jax.jit(jax.vmap(solve_initial_pose))(parameters)
    run = jax.jit(lambda x, p: rollout_batched(x, p, dt=1e-5, steps=1000))
    _, states = run(initial_states, parameters)
    states.block_until_ready()
    if not np.isfinite(np.asarray(states)).all():
        raise RuntimeError("Non-finite trajectory in design sweep.")
    for design, dx in zip(np.asarray(designs), np.asarray(states[:, -1, 0] - states[:, 0, 0])):
        print(f"zeta={design[0]:g}, had={design[1]:g} m, voltage={design[2]:g} V: dx={dx:.6g} m")


if __name__ == "__main__":
    main()
