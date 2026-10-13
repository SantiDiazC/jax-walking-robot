"""Configurable screening limits; passing is not experimental validation."""
from dataclasses import asdict, dataclass
import numpy as np
from .geometry import points, point_velocities


@dataclass(frozen=True)
class PhysicalLimits:
    max_foot_penetration_m: float = 0.00025  # 10% of nominal 2.5 mm legs.
    max_body_penetration_m: float = 0.0001
    max_body_height_m: float = 0.010
    max_center_speed_m_s: float = 3.5  # About 3x the saved MATLAB peak.
    max_foot_speed_m_s: float = 5.0
    max_angular_speed_rad_s: float = 2000.0
    max_relative_angle_rad: float = 1.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Physical limits must be finite and positive.")


def inspect_trajectory(times, states, parameters, limits=None):
    limits = limits or PhysicalLimits()
    t, x, p = np.asarray(times,dtype=float), np.asarray(states,dtype=float), np.asarray(parameters,dtype=float)
    if x.ndim != 2 or x.shape[1] != 14 or t.shape != (len(x),) or p.shape != (42,) or len(t) < 2:
        raise ValueError("Expected times (T,), states (T,14), parameters (42,), T >= 2.")
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0) or not np.isfinite(p).all():
        raise ValueError("Times must increase and times/parameters must be finite.")
    finite = np.isfinite(x).all(axis=1)
    first_bad = int(np.flatnonzero(~finite)[0]) if not finite.all() else None
    # Diagnose the finite prefix; do not replace bad values or hide the failure.
    stop = first_bad if first_bad is not None else len(x)
    metrics, violations = {}, ["nonfinite_state"] if first_bad is not None else []
    if stop:
        xx = x[:stop]
        with np.errstate(over="ignore", invalid="ignore"):
            g, v = points(xx, p), point_velocities(xx, p)
        quantities = {
            "foot_penetration_m": np.maximum(-g["feet"][..., 1], 0).max(axis=1),
            "body_penetration_m": np.maximum(-g["body"][..., 1], 0).max(axis=1),
            "body_height_m": g["body"][..., 1].max(axis=1),
            "center_speed_m_s": np.hypot(v["centers"][...,0],v["centers"][...,1]).max(axis=1),
            "foot_speed_m_s": np.hypot(v["feet"][...,0],v["feet"][...,1]).max(axis=1),
            "angular_speed_rad_s": np.abs(xx[:, 8:12]).max(axis=1),
            "relative_angle_rad": np.abs(np.diff(xx[:, 2:6], axis=1)).max(axis=1),
        }
        for name, values in quantities.items():
            peak = float(np.max(values))
            metrics["max_" + name] = peak if np.isfinite(peak) else None
            exceed = np.flatnonzero((values > getattr(limits, "max_" + name)) | ~np.isfinite(values))
            if len(exceed):
                violations.append(name)
                index = int(exceed[0])
                first_bad = min(first_bad, index) if first_bad is not None else index
        masses = p[:4]
        com_x = (g["centers"][..., 0] * (masses / masses.sum())).sum(axis=1)
        if stop > 1:
            halfway = (stop-1)//2
            with np.errstate(over="ignore", invalid="ignore"):
                for name, start in [("mean_com_speed_m_s", 0), ("late_mean_com_speed_m_s", halfway)]:
                    speed = float((com_x[-1]-com_x[start])/(t[stop-1]-t[start]))
                    metrics[name] = speed if np.isfinite(speed) else None
                    if not np.isfinite(speed):
                        violations.append("nonfinite_"+name)
                        first_bad = min(first_bad, stop-1) if first_bad is not None else stop-1
    return {"passed": not violations, "violations": violations,
            "first_violation_index": first_bad,
            "first_violation_time_s": float(t[first_bad]) if first_bad is not None else None,
            "finite_samples": stop, "total_samples": len(x),
            "metrics": metrics, "limits": asdict(limits),
            "status": "provisional_screening_limits"}


def assess_surrogate(cases):
    """Gate on held-out validation designs, never test-set performance."""
    validation = [r for r in cases if r["split"] == "validation"]
    admissible = [r for r in validation if r["reference_physical_validation"]["passed"]]
    def fidelity(record):
        return all(record.get(name) is not None and np.isfinite(record[name]) and record[name] <= limit
                   for name, limit in [("normalized_state_rmse", .10), ("late_mean_speed_error_m_s", .05)])
    false_accept = [r["id"] for r in validation if r["neural_physical_validation"]["passed"]
                    and not r["reference_physical_validation"]["passed"]]
    passed = len(admissible) >= 3 and all(fidelity(r) and r["neural_physical_validation"]["passed"] for r in admissible) and not false_accept
    return dict(validation_gate_passed=bool(passed),
                required_minimum_physically_valid_validation_designs=3,
                physically_valid_validation_designs=len(admissible), false_accept_design_ids=false_accept,
                thresholds=dict(normalized_state_rmse=.10, late_mean_speed_error_m_s=.05),
                gate_uses="Validation split only; test split reported independently.")


def assess_candidate(checks):
    """Require a feasible improvement and aggregate agreement at two timesteps."""
    timesteps = sorted({r["dt_s"] for r in checks})
    if len(timesteps) != 2 or len(checks) != 4 or not all(r["validation"]["passed"] for r in checks):
        return False
    for dt in timesteps:
        pair = {r["label"]:r["validation"]["metrics"] for r in checks if r["dt_s"] == dt}
        if set(pair) != {"baseline", "candidate"}:
            return False
        speeds = [pair[label].get("late_mean_com_speed_m_s") for label in ["baseline", "candidate"]]
        if any(v is None or not np.isfinite(v) for v in speeds) or speeds[1] <= speeds[0]:
            return False
    for label in ["baseline", "candidate"]:
        pair = [r["validation"]["metrics"] for r in checks if r["label"] == label]
        for name, tolerance in [("late_mean_com_speed_m_s", .01), ("max_foot_penetration_m", 2e-5)]:
            values = [r.get(name) for r in pair]
            if any(v is None or not np.isfinite(v) for v in values) or abs(values[0]-values[1]) > tolerance:
                return False
    return True
