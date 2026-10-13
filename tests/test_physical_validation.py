import numpy as np
import copy
import json
import pytest
import jax
import jax.numpy as jnp
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose
from walking_robot.geometry import points, point_velocities
from walking_robot.validation import PhysicalLimits, inspect_trajectory, assess_surrogate, assess_candidate


@pytest.fixture(scope="module")
def initial():
    p = calculate_P_from_meta_jax(.01, 1e-6, 200.)
    return np.asarray(p), np.asarray(jax.jit(solve_initial_pose)(p))


def test_geometry_lengths_and_velocities(initial):
    p, x = initial
    x = x.copy()
    x[6:12] = [.1, -.05, 3., -2., 1., 4.]
    g = points(x, p)
    np.testing.assert_allclose(np.linalg.norm(np.diff(g["body"], axis=0), axis=1), p[9:13], atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(g["feet"]-g["joints"], axis=1), p[13:15], atol=1e-15)
    epsilon = 1e-7
    ahead, behind = x.copy(), x.copy()
    ahead[:6] += epsilon*x[6:12]
    behind[:6] -= epsilon*x[6:12]
    actual = point_velocities(x, p)
    for key in ["feet", "centers"]:
        expected = (points(ahead, p)[key]-points(behind, p)[key])/(2*epsilon)
        np.testing.assert_allclose(actual[key], expected, atol=1e-10)
    jax_feet = points(jnp.asarray(x), jnp.asarray(p), xp=jnp)["feet"]
    np.testing.assert_allclose(jax_feet, g["feet"], atol=1e-15)


@pytest.mark.parametrize("kind", ["penetration", "flight", "speed", "angular_speed", "nan", "infinity"])
def test_rejects_unphysical_trajectories(initial, kind):
    p, x = initial
    states = np.tile(x, (5, 1))
    if kind == "penetration":
        states[3:, 1] -= .001
    elif kind == "flight":
        states[3:, 1] += 1.
    elif kind == "speed":
        states[3:, 6] = 100.
    elif kind == "angular_speed":
        states[3:, 8] = 100000.
    else:
        states[3:, 0] = np.nan if kind == "nan" else np.inf
    report = inspect_trajectory(np.arange(5)*1e-4, states, p)
    assert not report["passed"]
    assert report["first_violation_index"] == 3
    assert report["violations"]


def test_static_pose_passes_and_limits_are_not_changed(initial):
    p, x = initial
    report = inspect_trajectory(np.arange(5)*1e-4, np.tile(x, (5, 1)), p)
    assert report["passed"]
    assert report["limits"]["max_foot_penetration_m"] == .00025
    assert report["first_violation_time_s"] is None
    with pytest.raises(ValueError):
        PhysicalLimits(max_foot_penetration_m=-1.)


def test_overflow_is_rejected_and_report_is_strict_json(initial):
    p, x = initial
    states = np.tile(x, (5, 1))
    states[3:, 0] = 1e308
    report = inspect_trajectory(np.arange(5)*1e-4, states, p)
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


def test_surrogate_gate_rejects_no_domain_errors_and_false_accepts():
    good = [dict(id=i, split="validation", reference_physical_validation={"passed": True},
                 neural_physical_validation={"passed": True}, normalized_state_rmse=.01,
                 late_mean_speed_error_m_s=.001) for i in range(3)]
    assert assess_surrogate(good)["validation_gate_passed"]
    for key, value in [("normalized_state_rmse", .5), ("late_mean_speed_error_m_s", None),
                       ("neural_physical_validation", {"passed":False}),
                       ("reference_physical_validation", {"passed":False})]:
        cases = copy.deepcopy(good)
        cases[0][key] = value
        assert not assess_surrogate(cases)["validation_gate_passed"]
    assert not assess_surrogate([dict(r, split="test") for r in good])["validation_gate_passed"]
    cases = [dict(r,reference_physical_validation={"passed":False}) for r in good]
    assert not assess_surrogate(cases)["validation_gate_passed"]


def test_candidate_needs_feasible_improvement_at_both_timesteps():
    checks = [dict(label=label,dt_s=dt,validation={"passed":True,"metrics":{
        "late_mean_com_speed_m_s":speed,"max_foot_penetration_m":.0001}})
        for dt in [1e-5,5e-6] for label,speed in [("baseline",.1),("candidate",.15)]]
    assert assess_candidate(checks)
    assert not assess_candidate(checks[:2])
    for key,value in [("late_mean_com_speed_m_s", .09), ("late_mean_com_speed_m_s", .2),
                      ("max_foot_penetration_m", .0002)]:
        invalid = copy.deepcopy(checks)
        invalid[-1]["validation"]["metrics"][key] = value
        assert not assess_candidate(invalid)
    checks[-1]["validation"]["passed"] = False
    assert not assess_candidate(checks)
