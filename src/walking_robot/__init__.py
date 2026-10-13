"""JAX dynamics for a planar piezoelectric walking robot.

Enable ``jax_enable_x64`` in the application before constructing arrays for
numerical comparisons. Importing this package does not change JAX settings.
"""
from .contact import computeM, computeF, computeSdot
from .parameters import calculate_P_from_meta_jax
from .initial_pose import solve_initial_pose
from .dynamics import dynamics_rhs, ramp_contact_P
from .integrators import dynamics_step, rollout_scan, rollout_batched

__all__ = [
    "computeM", "computeF", "computeSdot", "calculate_P_from_meta_jax",
    "solve_initial_pose", "dynamics_rhs", "ramp_contact_P", "dynamics_step",
    "rollout_scan", "rollout_batched",
]
