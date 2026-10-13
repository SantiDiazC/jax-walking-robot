from copy import deepcopy
from walking_robot.surrogate_checks import validated_objective_horizons


def evidence():
    results = {}
    for name,ids in [('validation',[0,1,2]),('test',[3,4,5])]:
        results[name] = dict(cases=[dict(id=i) for i in ids],trajectory_gate_passed=True,
            derivatives=dict(kinematic_gate_passed=True,kinematics=[dict(id=i,passed=True) for i in ids],
                cases=[dict(id=i,horizon_s=h,passed=(h!=.2 or i!=5)) for i in ids for h in [.2,1.,2.]]))
    return dict(surrogate_gates_passed=False,results=results)


def test_scoped_objective_preserves_the_global_failure():
    report = evidence()
    original = deepcopy(report)
    assert validated_objective_horizons(report)==[1.,2.]
    assert report==original and not report['surrogate_gates_passed']


def test_missing_or_duplicate_gradient_evidence_cannot_certify_that_horizon():
    report = evidence()
    rows = report['results']['test']['derivatives']['cases']
    rows[:] = [r for r in rows if not (r['id']==5 and r['horizon_s']==1.)]
    assert validated_objective_horizons(report)==[2.]
    rows.append(dict(id=4,horizon_s=2.,passed=True))
    assert validated_objective_horizons(report)==[]


def test_empty_or_missing_split_fails_closed_even_with_positive_flags():
    report = evidence()
    report['results']['test']['cases'] = []
    assert validated_objective_horizons(report)==[]
    assert validated_objective_horizons(dict(results={}))==[]
    assert validated_objective_horizons(dict(results={'validation':evidence()['results']['validation']}))==[]


def test_missing_kinematic_evidence_cannot_certify_an_objective():
    report = evidence()
    report['results']['test']['derivatives']['kinematics'].pop()
    assert validated_objective_horizons(report)==[]
