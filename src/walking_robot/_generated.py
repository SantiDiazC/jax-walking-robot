"""MATLAB-exported equations. Keep algebra aligned with reference/matlab/generated."""
from __future__ import annotations
from typing import Callable
import jax.numpy as jnp
from jax import lax

def _cond_chain(pairs, default: Callable[[], jnp.ndarray]):
    """Conditional selections preserving MATLAB's first-match priority.

    The accumulated fallback is computed before each selection; this helper
    preserves the original port's evaluation order.

    pairs: list of (cond_bool, expr_fn) where expr_fn returns a scalar/array.
    default: expr_fn for else branch.
    """
    out = default()
    for cond, fn in reversed(pairs):
        out = lax.cond(cond, fn, lambda: out)
    return out


def computeM(t, x, P):
    """Mass/constraint matrix (12x12). Auto-translated from computeM_gen.m."""
    I1 = P[4]
    I2 = P[5]
    I3 = P[6]
    I4 = P[7]
    l1 = P[9]
    l2 = P[10]
    l3 = P[11]
    l4 = P[12]
    m1 = P[0]
    m2 = P[1]
    m3 = P[2]
    m4 = P[3]
    th1 = x[2]
    th2 = x[3]
    th3 = x[4]
    th4 = x[5]
    mt1 = jnp.array([m1, 0.0, 0.0, m2, 0.0, 0.0, m3, 0.0, 0.0, m4, 0.0, 0.0, 0.0, m1, 0.0, 0.0, m2, 0.0, 0.0, m3, 0.0, 0.0, m4, 0.0, 0.0, 0.0, I1, (l1*m2*jnp.sin(th1))/2.0, l1*m2*jnp.cos(th1)*(-1.0/2.0), 0.0, (l1*m3*jnp.sin(th1))/2.0, l1*m3*jnp.cos(th1)*(-1.0/2.0), 0.0, (l1*m4*jnp.sin(th1))/2.0, l1*m4*jnp.cos(th1)*(-1.0/2.0), 0.0, 0.0, 0.0, 0.0, (l2*m2*jnp.sin(th2))/2.0, l2*m2*jnp.cos(th2)*(-1.0/2.0), I2, l2*m3*jnp.sin(th2), -l2*m3*jnp.cos(th2), 0.0, l2*m4*jnp.sin(th2), -l2*m4*jnp.cos(th2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (l3*m3*jnp.sin(th3))/2.0, l3*m3*jnp.cos(th3)*(-1.0/2.0), I3, l3*m4*jnp.sin(th3), -l3*m4*jnp.cos(th3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (l4*m4*jnp.sin(th4))/2.0, l4*m4*jnp.cos(th4)*(-1.0/2.0), I4, 1.0, 0.0], dtype=x.dtype)
    mt2 = jnp.array([(l1*jnp.sin(th1))/2.0, -1.0, 0.0, (l2*jnp.sin(th2))/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, l1*jnp.cos(th1)*(-1.0/2.0), 0.0, -1.0, l2*jnp.cos(th2)*(-1.0/2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, (l2*jnp.sin(th2))/2.0, -1.0, 0.0, (l3*jnp.sin(th3))/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, l2*jnp.cos(th2)*(-1.0/2.0), 0.0, -1.0, l3*jnp.cos(th3)*(-1.0/2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, (l3*jnp.sin(th3))/2.0, -1.0, 0.0, (l4*jnp.sin(th4))/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, l3*jnp.cos(th3)*(-1.0/2.0), 0.0, -1.0, l4*jnp.cos(th4)*(-1.0/2.0)], dtype=x.dtype)
    v = jnp.concatenate([mt1, mt2], axis=0)
    # MATLAB reshape uses column-major; equivalent to row-major reshape + transpose.
    M_final = jnp.reshape(v, (12, 12)).T
    return M_final


def computeF(t, x, P):
    """Auto-translated from MATLAB Symbolic Toolbox output."""
    # x: (14,), P: (42,)
    # MATLAB used in2 (state) and in3 (params) with 1-based indexing.
    # This function aims to match MATLAB exactly (hard branches, no smoothing).

    al1 = P[15]
    al2 = P[16]
    ct_ratio = P[36]
    damping = P[20]
    dx_bwd = P[40]
    dx_fwd = P[39]
    dy_gnd = P[27]
    freq = P[22]
    g = P[8]
    h1 = P[17]
    h2 = P[18]
    kspring = P[19]
    kt_ratio = P[35]
    ky_gnd = P[26]
    l1 = P[9]
    l2 = P[10]
    l3 = P[11]
    l4 = P[12]
    lleg1 = P[13]
    lleg2 = P[14]
    m1 = P[0]
    m2 = P[1]
    m3 = P[2]
    m4 = P[3]
    mu_k_bwd = P[29]
    mu_k_fwd = P[28]
    mu_s_bwd = P[31]
    mu_s_fwd = P[30]
    pcoeff = P[34]
    s1 = x[12]
    s2 = x[13]
    tau_mag = P[21]
    th1 = x[2]
    th2 = x[3]
    th3 = x[4]
    th4 = x[5]
    theta1_init = P[23]
    theta2_init = P[24]
    theta3_init = P[25]
    vdead = P[32]
    vscale = P[33]
    vx = x[6]
    vy = x[7]
    w1 = x[8]
    w2 = x[9]
    w3 = x[10]
    w4 = x[11]
    y = x[1]
    et1 = (lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+damping*(w3-w4)
    et2 = -(lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*(ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)
    et3 = jnp.tanh((vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vdead)*(mu_k_bwd-jnp.exp(-(-(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vscale)**pcoeff)*(mu_k_bwd-mu_s_bwd))
    et4 = jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))
    et5 = (lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+damping*(w3-w4)
    et6 = -(lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*(et3*et4+dx_bwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3)))-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)
    et7 = jnp.tanh((vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vdead)*(mu_k_fwd-jnp.exp(-(jnp.abs(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vscale)**pcoeff)*(mu_k_fwd-mu_s_fwd))
    et8 = jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))
    et9 = (lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))
    et10 = -(lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*(et7*et8+dx_fwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3)))+damping*(w3-w4)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)
    et11 = jnp.tanh((vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vdead)*(mu_k_bwd-jnp.exp(-(-(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vscale)**pcoeff)*(mu_k_bwd-mu_s_bwd))
    et12 = -jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))
    et13 = jnp.tanh((vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vdead)*(mu_k_fwd-jnp.exp(-(jnp.abs(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))/vscale)**pcoeff)*(mu_k_fwd-mu_s_fwd))
    et14 = -jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))
    et15 = -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)+(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))
    et16 = -(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))*(dx_bwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+jnp.tanh((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vdead)*(mu_k_bwd-jnp.exp(-(-(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vscale)**pcoeff)*(mu_k_bwd-mu_s_bwd))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))
    et17 = -(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))*(dx_fwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+jnp.tanh((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vdead)*(mu_k_fwd-jnp.exp(-(jnp.abs(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vscale)**pcoeff)*(mu_k_fwd-mu_s_fwd))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))
    et18 = -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)+(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))
    mt1 = jnp.zeros((0,), dtype=x.dtype)
    t0 = _cond_chain([
        (((jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) | (lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0) <= y), lambda: 0.0),
        ((((0.0 < jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < 0.0) & (0.0 < vdead+vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) | ((0.0 <= vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < vdead)))) & (((0.0 < vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))) | ((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))))) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: -kt_ratio*ky_gnd*s1-ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        ((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: -dx_bwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))-jnp.tanh((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vdead)*(mu_k_bwd-jnp.exp(-(-(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vscale)**pcoeff)*(mu_k_bwd-mu_s_bwd))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        (vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0, lambda: -dx_bwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0), lambda: -dx_fwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))-jnp.tanh((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vdead)*(mu_k_fwd-jnp.exp(-(jnp.abs(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))/vscale)**pcoeff)*(mu_k_fwd-mu_s_fwd))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
    ], default=lambda: -dx_fwd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))
    mt2 = jnp.array([t0], dtype=x.dtype)
    t0 = _cond_chain([
        (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0), lambda: -g*m1+jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
    ], default=lambda: -g*m1)
    mt3 = jnp.array([t0], dtype=x.dtype)
    t0 = _cond_chain([
        ((((jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) | (lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0) <= y)) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)+(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        (((jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) | (lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0) <= y), lambda: -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)),
        ((((0.0 < jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < 0.0) & (0.0 < vdead+vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) | ((0.0 <= vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < vdead)))) & (((0.0 < vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))) | ((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))))) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: -(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))*(kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))-damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)+(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0))*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        ((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: et15+et16),
        (vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0, lambda: -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)-dx_bwd*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))),
        (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0), lambda: et17+et18),
    ], default=lambda: -damping*(w1-w2)+tau_mag*jnp.sin(freq*t*jnp.pi*2.0)-kspring*(th1-th2+theta1_init)-dx_fwd*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))
    mt4 = jnp.array([t0], dtype=x.dtype)
    mt5 = jnp.array([-m2*((l1*w1**2*jnp.cos(th1))/2.0+(l2*w2**2*jnp.cos(th2))/2.0), -g*m2-m2*((l1*w1**2*jnp.sin(th1))/2.0+(l2*w2**2*jnp.sin(th2))/2.0), damping*(w1-w2)-damping*(w2-w3)+kspring*(th1-th2+theta1_init)-kspring*(th2-th3+theta2_init), -m3*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+(l3*w3**2*jnp.cos(th3))/2.0), -g*m3-m3*((l1*w1**2*jnp.sin(th1))/2.0+l2*w2**2*jnp.sin(th2)+(l3*w3**2*jnp.sin(th3))/2.0)], dtype=x.dtype)
    mt6 = jnp.array([damping*(w2-w3)-damping*(w3-w4)+kspring*(th2-th3+theta2_init)-kspring*(th3-th4+theta3_init)], dtype=x.dtype)
    t0 = _cond_chain([
        ((l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0 <= y*2.0+h2*jnp.sin(th4)*2.0) | ((jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3))) <= 0.0) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)), lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)),
        ((((0.0 < jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)) & (((0.0 < vdead*2.0+vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0) & (vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < 0.0)) | ((vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < vdead*2.0) & (0.0 <= vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0)))) & (((((0.0 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (h2*w4*jnp.sin(th4)*2.0 < vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0)) | ((((0.0 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0))), lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)-ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))-kt_ratio*ky_gnd*s2),
        ((y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0) & (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0), lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)+et11*et12-dx_bwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))),
        (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0, lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)-dx_bwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))),
        (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0, lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)+et13*et14-dx_fwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))),
    ], default=lambda: -m4*((l1*w1**2*jnp.cos(th1))/2.0+l2*w2**2*jnp.cos(th2)+l3*w3**2*jnp.cos(th3)+(l4*w4**2*jnp.cos(th4))/2.0)-dx_fwd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3)))
    mt7 = jnp.array([t0], dtype=x.dtype)
    t0 = _cond_chain([
        (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0, lambda: -g*m4-m4*((l1*w1**2*jnp.sin(th1))/2.0+l2*w2**2*jnp.sin(th2)+l3*w3**2*jnp.sin(th3)+(l4*w4**2*jnp.sin(th4))/2.0)+jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))),
    ], default=lambda: -g*m4-m4*((l1*w1**2*jnp.sin(th1))/2.0+l2*w2**2*jnp.sin(th2)+l3*w3**2*jnp.sin(th3)+(l4*w4**2*jnp.sin(th4))/2.0))
    mt8 = jnp.array([t0], dtype=x.dtype)
    t0 = _cond_chain([
        ((y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0) & ((l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0 <= y*2.0+h2*jnp.sin(th4)*2.0) | ((jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3))) <= 0.0) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0))), lambda: (lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+damping*(w3-w4)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)),
        ((l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0 <= y*2.0+h2*jnp.sin(th4)*2.0) | ((jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3))) <= 0.0) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)), lambda: damping*(w3-w4)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)),
        ((((0.0 < jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)) & (((0.0 < vdead*2.0+vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0) & (vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < 0.0)) | ((vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < vdead*2.0) & (0.0 <= vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0)))) & (((((0.0 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (h2*w4*jnp.sin(th4)*2.0 < vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0)) | ((((0.0 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0))), lambda: et1+et2),
        ((y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0) & (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0), lambda: et5+et6),
        (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0, lambda: damping*(w3-w4)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)-dx_bwd*(lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))),
        (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0, lambda: et9+et10),
    ], default=lambda: damping*(w3-w4)-tau_mag*jnp.sin(freq*t*jnp.pi*2.0)+kspring*(th3-th4+theta3_init)-dx_fwd*(lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3)))
    mt9 = jnp.array([t0], dtype=x.dtype)
    F_total = jnp.concatenate([mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8, mt9], axis=0)
    return F_total


def computeSdot(t, x, P):
    """Auto-translated from MATLAB Symbolic Toolbox output."""
    # x: (14,), P: (42,)
    # MATLAB used in2 (state) and in3 (params) with 1-based indexing.
    # This function aims to match MATLAB exactly (hard branches, no smoothing).

    al1 = P[15]
    al2 = P[16]
    ct_ratio = P[36]
    dy_gnd = P[27]
    h1 = P[17]
    h2 = P[18]
    kt_ratio = P[35]
    ky_gnd = P[26]
    l1 = P[9]
    l2 = P[10]
    l3 = P[11]
    l4 = P[12]
    lambda_off = P[38]
    lambda_slip = P[37]
    lleg1 = P[13]
    lleg2 = P[14]
    mu_s_bwd = P[31]
    mu_s_fwd = P[30]
    s1 = x[12]
    s2 = x[13]
    th1 = x[2]
    th2 = x[3]
    th3 = x[4]
    th4 = x[5]
    vdead = P[32]
    vx = x[6]
    vy = x[7]
    w1 = x[8]
    w2 = x[9]
    w3 = x[10]
    w4 = x[11]
    y = x[1]
    mt1 = jnp.zeros((0,), dtype=x.dtype)
    t0 = _cond_chain([
        (((jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= 0.0) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) | (lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0) <= y), lambda: -lambda_off*s1),
        ((((0.0 < jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < 0.0) & (0.0 < vdead+vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) | ((0.0 <= vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) < vdead)))) & (((0.0 < vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_fwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))) | ((vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)) <= 0.0) & (((kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) < 0.0) & (0.0 <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))+kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))) | ((0.0 <= kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)))) & (kt_ratio*ky_gnd*s1+ct_ratio*dy_gnd*(vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))) <= mu_s_bwd*jnp.maximum(0.0, -dy_gnd*(vy+w1*(lleg1*jnp.cos(al1-th1)-jnp.cos(th1)*(h1-l1/2.0)))+ky_gnd*(-y+lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))))))))) & (y < lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0)), lambda: vx+w1*(lleg1*jnp.sin(al1-th1)+jnp.sin(th1)*(h1-l1/2.0))),
    ], default=lambda: -lambda_slip*s1)
    mt2 = jnp.array([t0], dtype=x.dtype)
    t0 = _cond_chain([
        ((l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0 <= y*2.0+h2*jnp.sin(th4)*2.0) | ((jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3))) <= 0.0) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)), lambda: -lambda_off*s2),
        ((((0.0 < jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (y*2.0+h2*jnp.sin(th4)*2.0 < l1*jnp.sin(th1)+l2*jnp.sin(th2)*2.0+l3*jnp.sin(th3)*2.0+l4*jnp.sin(th4)*2.0+lleg2*jnp.sin(al2-th4)*2.0)) & (((0.0 < vdead*2.0+vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0) & (vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < 0.0)) | ((vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0 < vdead*2.0) & (0.0 <= vx*2.0+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0)))) & (((((0.0 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_fwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (h2*w4*jnp.sin(th4)*2.0 < vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0)) | ((((0.0 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))+ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2) & (ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 < 0.0)) | ((ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2 <= mu_s_bwd*jnp.maximum(0.0, ky_gnd*(-y+(l1*jnp.sin(th1))/2.0+l2*jnp.sin(th2)+l3*jnp.sin(th3)+(l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+dy_gnd*(-vy-w4*(l4*jnp.cos(th4)*(-1.0/2.0)+lleg2*jnp.cos(al2-th4)+jnp.cos(th4)*(h2-l4/2.0))+(l1*w1*jnp.cos(th1))/2.0+l2*w2*jnp.cos(th2)+l3*w3*jnp.cos(th3)))) & (0.0 <= ct_ratio*dy_gnd*(vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3))+kt_ratio*ky_gnd*s2))) & (vx*2.0+l1*w1*jnp.sin(th1)+l2*w2*jnp.sin(th2)*2.0+l3*w3*jnp.sin(th3)*2.0+l4*w4*jnp.sin(th4)*2.0+lleg2*w4*jnp.sin(al2-th4)*2.0 <= h2*w4*jnp.sin(th4)*2.0))), lambda: vx+w4*((l4*jnp.sin(th4))/2.0+lleg2*jnp.sin(al2-th4)-jnp.sin(th4)*(h2-l4/2.0))+(l1*w1*jnp.sin(th1))/2.0+l2*w2*jnp.sin(th2)+l3*w3*jnp.sin(th3)),
    ], default=lambda: -lambda_slip*s2)
    mt3 = jnp.array([t0], dtype=x.dtype)
    out1 = jnp.concatenate([mt1, mt2, mt3], axis=0)
    return out1
