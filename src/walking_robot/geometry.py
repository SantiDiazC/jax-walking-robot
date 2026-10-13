"""Planar geometry shared by diagnostics, plotting and differentiable features.

Inputs may have leading dimensions. ``xp`` can be numpy, jax.numpy or torch.
Positions follow the MATLAB plotting order; lengths are in meters.
"""
import numpy as np


def points(state, parameters, xp=np):
    stack = (lambda items, axis: xp.stack(items, dim=axis)) if xp.__name__ == "torch" else xp.stack
    q = state[..., :6]
    p = parameters
    direction = lambda a: stack([xp.cos(a), xp.sin(a)], -1)
    u = [direction(q[..., i]) for i in range(2, 6)]
    scaled = lambda value, vec: value[..., None] * vec
    cg1 = q[..., :2]
    pin1 = cg1 - scaled(p[..., 9] / 2, u[0])
    cg2 = pin1 - scaled(p[..., 10] / 2, u[1])
    pin2 = cg2 - scaled(p[..., 10] / 2, u[1])
    cg3 = pin2 - scaled(p[..., 11] / 2, u[2])
    pin3 = cg3 - scaled(p[..., 11] / 2, u[2])
    cg4 = pin3 - scaled(p[..., 12] / 2, u[3])
    edge1 = cg1 + scaled(p[..., 9] / 2, u[0])
    edge2 = cg4 - scaled(p[..., 12] / 2, u[3])
    joint1 = cg1 + scaled(p[..., 9] / 2 - p[..., 17], u[0])
    joint2 = cg4 - scaled(p[..., 12] / 2 - p[..., 18], u[3])
    foot1 = joint1 + scaled(p[..., 13], direction(q[..., 2] - p[..., 15]))
    foot2 = joint2 + scaled(p[..., 14], direction(q[..., 5] - p[..., 16]))
    return {
        "body": stack([edge1, pin1, pin2, pin3, edge2], -2),
        "centers": stack([cg1, cg2, cg3, cg4], -2),
        "joints": stack([joint1, joint2], -2),
        "feet": stack([foot1, foot2], -2),
        "matlab_points": stack([foot1, joint1, edge1, cg1, pin1, cg2, pin2,
                                cg3, pin3, cg4, edge2, joint2, foot2], -2),
    }


def point_velocities(state, parameters, xp=np):
    """Exact kinematic time derivative without finite differences."""
    stack = (lambda items, axis: xp.stack(items, dim=axis)) if xp.__name__ == "torch" else xp.stack
    p = parameters
    q = state[..., :6]
    perp = lambda a: stack([-xp.sin(a), xp.cos(a)], -1)
    term = lambda length, rate, angle: (length * rate)[..., None] * perp(angle)
    v1 = state[..., 6:8]
    w = [state[..., i] for i in range(8, 12)]
    vp1 = v1 - term(p[..., 9]/2, w[0], q[..., 2])
    v2 = vp1 - term(p[..., 10]/2, w[1], q[..., 3])
    vp2 = v2 - term(p[..., 10]/2, w[1], q[..., 3])
    v3 = vp2 - term(p[..., 11]/2, w[2], q[..., 4])
    vp3 = v3 - term(p[..., 11]/2, w[2], q[..., 4])
    v4 = vp3 - term(p[..., 12]/2, w[3], q[..., 5])
    vf = v1 + term(p[..., 9]/2-p[..., 17], w[0], q[..., 2]) + term(p[..., 13], w[0], q[..., 2]-p[..., 15])
    vh = v4 - term(p[..., 12]/2-p[..., 18], w[3], q[..., 5]) + term(p[..., 14], w[3], q[..., 5]-p[..., 16])
    return {"centers": stack([v1, v2, v3, v4], -2), "feet": stack([vf, vh], -2)}
