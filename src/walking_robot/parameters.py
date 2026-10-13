"""Material and design inputs mapped to the legacy 42-element parameter vector."""
from __future__ import annotations
import jax.numpy as jnp

def calculate_P_from_meta_jax(zeta: jnp.ndarray, had: jnp.ndarray, volt: jnp.ndarray = 200.0,
                              front_angle_deg: jnp.ndarray = 95.0,
                              hind_angle_deg: jnp.ndarray = 125.0,
                              *, profile: str = "legacy") -> jnp.ndarray:
    """Construct the legacy design parameter vector (SI units).

    Returns:
        P: (42,) float array
    """
    if profile not in {"legacy", "matlab"}:
        raise ValueError("Parameter profile must be legacy or matlab.")
    # Constants & geometry
    w = 15e-3
    l_total = 25e-3
    l_seg = l_total / 4.0

    # Materials
    ha = 80e-6
    Ya = 2.8e9
    ra = 1853.0
    hp = 75e-6
    Yp = 4.0e9
    rp = 1461.0
    Yad = 61e6
    rad = 1000.0  # iCVD

    # Structural calcs
    rhoA = w * (ra * ha + rp * hp + rad * had)

    num = Ya * w * ha * (hp + had + ha / 2) + Yad * w * had * (hp + had / 2) + Yp * w * hp * (hp / 2)
    den = Ya * w * ha + Yad * w * had + Yp * w * hp
    y_bar = num / den

    I_a = (w * ha**3) / 12 + w * ha * (hp + had + ha / 2 - y_bar) ** 2
    I_ad = (w * had**3) / 12 + w * had * (hp + had / 2 - y_bar) ** 2
    I_p = (w * hp**3) / 12 + w * hp * (hp / 2 - y_bar) ** 2
    EI = Ya * I_a + Yad * I_ad + Yp * I_p

    beta1 = 4.73
    omega_n = (beta1**2) * jnp.sqrt(EI / (rhoA * l_total**4))
    omega_2nd = 22.373 / (l_total**2) * jnp.sqrt(EI / rhoA)
    freq_val = jnp.ceil(omega_2nd / (2 * jnp.pi))

    # Mass & inertia (per segment)
    m_seg_val = rhoA * l_seg
    I_seg_val = (1.0 / 12.0) * m_seg_val * l_seg**2

    kspring_val = 2 * EI / l_seg / 2
    damping_val = 2 * zeta * kspring_val / omega_n

    d31 = -22e-12
    he = had
    term = 2 * y_bar * hp - 2 * hp * (hp + ha + he) + hp**2
    tau_mag_val = 0.5 * (w * Yp * d31 * volt / hp * term)
    if profile == "matlab":
        # Reproduce the actual MomentPiezo call in the archived MATLAB script:
        # (w,d31,V,Ya,ha,hp,had,y_bar), including its material/layer choice.
        tau_mag_val = .5*(w*Ya*d31*volt/ha*(2*y_bar*ha-2*ha*(ha+had+hp)+ha**2))

    P = jnp.zeros((42,), dtype=jnp.result_type(zeta, had, volt, front_angle_deg, hind_angle_deg))

    # Mass [0-8]
    P = P.at[0:4].set(m_seg_val)
    P = P.at[4:8].set(I_seg_val)
    P = P.at[8].set(9.81)

    # Geometry [9-18]
    P = P.at[9:13].set(l_seg)
    P = P.at[13].set(2.5e-3)  # lleg1
    P = P.at[14].set(2.5e-3)  # lleg2
    P = P.at[15].set(jnp.deg2rad(front_angle_deg))
    P = P.at[16].set(jnp.deg2rad(hind_angle_deg))
    P = P.at[17].set(2.8e-3 if profile == "matlab" else 2.5e-4)
    P = P.at[18].set(2.5e-3 if profile == "matlab" else 2.5e-4)

    # Control [19-22]
    P = P.at[19].set(kspring_val)
    P = P.at[20].set(damping_val)
    P = P.at[21].set(tau_mag_val)
    P = P.at[22].set(freq_val)

    # Angles [23-25] (theta1_init, theta2_init, theta3_init)
    P = P.at[23:26].set(jnp.deg2rad(12.0 / 3.0))

    # Ground [26-27] (ky, dy)  — scalar simplification keeps kyf[0], kyh[0]
    theta_front_deg = front_angle_deg
    theta_hind_deg = hind_angle_deg
    theta_total_deg = 12.0
    theta1 = jnp.deg2rad(theta_total_deg / 3.0)
    theta2 = theta1
    theta3 = theta1

    sin05_sq = jnp.sin(jnp.deg2rad(0.5)) ** 2
    phi_degf = 180.0 - theta_front_deg + jnp.rad2deg(theta1) + jnp.rad2deg(theta2) / 2.0
    denf = jnp.maximum(jnp.sin(jnp.deg2rad(phi_degf)) ** 2, sin05_sq)
    phi_degh = 180.0 - theta_hind_deg + jnp.rad2deg(theta3) + jnp.rad2deg(theta2) / 2.0
    denh = jnp.maximum(jnp.sin(jnp.deg2rad(phi_degh)) ** 2, sin05_sq)

    tfront = 25e-6
    thind = 25e-6
    Ihind = (w * thind**3) / 12.0
    lfront = 2.5e-3
    lhind = 2.5e-3

    # linspace(1,5,5)[0] == 1.0, so we take the first element directly for kyf[0], kyh[0]
    kyf2_0 = 3.0 * (Ihind * Yp) / (lfront**3) * 1.0
    kyh2_0 = 3.0 * (Ihind * Yp) / (lhind**3) * 1.0
    if profile == "matlab":
        # MATLAB rounds the beam term before combining hinge compliance.
        kyf2_0, kyh2_0 = jnp.ceil(kyf2_0), jnp.ceil(kyh2_0)

    foldlinef = 65e-6
    foldlineh = 40e-6
    foldlinef_plus = foldlinef * 0.5
    foldlinef_min = foldlinef * 2.0
    foldlineh_plus = foldlineh * 0.5
    foldlineh_min = foldlineh * 2.0

    D_strip = Yp * (tfront**3) / 12.0
    hingef_plus = D_strip * w / foldlinef_plus
    hingef_min = D_strip * w / foldlinef_min
    hingeh_plus = D_strip * w / foldlineh_plus
    hingeh_min = D_strip * w / foldlineh_min

    compliancef_plus = 1.0 / kyf2_0 + (lfront**2) / hingef_plus
    compliancef_min = 1.0 / kyf2_0 + (lfront**2) / hingef_min
    complianceh_plus = 1.0 / kyh2_0 + (lfront**2) / hingeh_plus
    complianceh_min = 1.0 / kyh2_0 + (lfront**2) / hingeh_min

    stifff_plus_0 = 1.0 / compliancef_plus * (1.0 / denf)
    stifff_min_0 = 1.0 / compliancef_min * (1.0 / denf)
    stiffh_plus_0 = 1.0 / complianceh_plus * (1.0 / denh)
    stiffh_min_0 = 1.0 / complianceh_min * (1.0 / denh)

    kyf0 = jnp.ceil((stifff_plus_0 + stifff_min_0) / 2.0)
    kyh0 = jnp.ceil((stiffh_plus_0 + stiffh_min_0) / 2.0)
    ky_avg = 0.5 * (kyf0 + kyh0)

    P = P.at[26].set(ky_avg)
    if profile == "matlab":
        # The old padding slot now stores hind stiffness. Zero retains legacy
        # algebra and shared P[26]; positive values select independent feet.
        P = P.at[26].set(kyf0).at[41].set(kyh0)
    P = P.at[27].set(8e-5)

    # Friction [28-31]
    P = P.at[28].set(0.03)  # mu_fwd
    P = P.at[29].set(0.4)   # mu_bwd
    P = P.at[30].set(0.03)  # mu_s_fwd
    P = P.at[31].set(0.8)   # mu_s_bwd

    # Advanced friction [32-40]
    P = P.at[32].set(1e-3)    # vdead
    P = P.at[33].set(50e-3)   # vscale
    P = P.at[34].set(2.0)     # pcoeff
    P = P.at[35].set(0.2)     # kt_ratio
    P = P.at[36].set(0.1)     # ct_ratio
    P = P.at[37].set(200.0)   # lambda_slip
    P = P.at[38].set(1000.0)  # lambda_off
    P = P.at[39].set(1e-4)    # dx_fwd
    P = P.at[40].set(3e-3)    # dx_bwd

    return P
