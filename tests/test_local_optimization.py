"""Verify rejection behavior and physical monotonicity independently of a NN."""
import numpy as np
import pytest
from walking_robot.local_optimization import LocalSettings,verified_local_search


INITIAL = np.array([.3,.5])


def neural(u):
    return .81+.2*(u[0]-.3),np.array([.2,0.])


def runs(u,*,slope=.2,coarse_bias=1e-6,passed=True):
    return [dict(dt_s=dt,validation=dict(passed=passed,metrics=dict(
        late_mean_com_speed_m_s=.81+slope*(u[0]-.3)+(coarse_bias if k==0 else 0.),
        max_foot_penetration_m=.0002))) for k,dt in enumerate([2.5e-6,1.25e-6])]


def search(evaluate,**kwargs):
    return verified_local_search(INITIAL,neural,evaluate,lambda a,b:True,LocalSettings(**kwargs))


def test_every_accepted_step_is_physically_improving_and_stays_local():
    original = INITIAL.copy()
    result = search(runs)
    assert result['accepted_steps']==3
    assert result['improvement_m_s']==pytest.approx(.0018)
    assert np.linalg.norm(np.array(result['final_unit'])-original)<=.01
    assert all(r['physical_gain_m_s']>0 and r['physical_check']['passed'] for r in result['history'])
    np.testing.assert_array_equal(INITIAL,original)


def test_wrong_neural_direction_never_changes_accepted_state():
    result = search(lambda u:runs(u,slope=-.2))
    assert result['accepted_steps']==0
    np.testing.assert_array_equal(result['final_unit'],INITIAL)
    assert len(result['history'])==3 and result['status']=='no_verified_step'


def test_physically_invalid_proposals_are_rejected_without_repair():
    result = search(lambda u:runs(u,passed=np.array_equal(u,INITIAL)))
    assert result['accepted_steps']==0
    assert all(not r['physical_check']['passed'] for r in result['history'])
    np.testing.assert_array_equal(result['final_unit'],INITIAL)


def test_gain_smaller_than_individual_resolution_changes_is_rejected():
    result = search(lambda u:runs(u,coarse_bias=.0004))
    assert result['accepted_steps']==0
    assert all(r['physical_check']['checks'][0]['required_improvement_m_s']>.001 for r in result['history'])


def test_discrete_boundary_backtracks_before_any_physical_evaluation():
    queried = []
    def evaluate(u):
        queried.append(u.copy())
        assert u[0]<=.301
        return runs(u)
    result = verified_local_search(INITIAL,neural,evaluate,lambda a,b:b[0]<=.301,LocalSettings(iterations=1))
    assert result['accepted_steps']==1 and len(queried)==2
    assert result['history'][-1]['unit_step']==pytest.approx(.00075)


@pytest.mark.parametrize('duplicate',[False,True])
def test_incomplete_or_duplicate_baseline_cannot_start_optimization(duplicate):
    def evaluate(u):
        row = runs(u)[0]
        return [row,row] if duplicate else [row]
    with pytest.raises(ValueError,match='Incomplete'):
        search(evaluate)


def test_local_surrogate_bias_stops_before_proposals():
    result = search(lambda u:runs(u,slope=.2,coarse_bias=.0001)[0:1]+[
        dict(dt_s=1.25e-6,validation=dict(passed=True,metrics=dict(late_mean_com_speed_m_s=.9,max_foot_penetration_m=.0002)))])
    assert result['status']=='local_surrogate_speed_mismatch' and result['accepted_steps']==0


def test_positive_gain_with_inaccurate_forecast_is_rejected():
    result = search(lambda u:runs(u,slope=.4))
    assert result['accepted_steps']==0
    assert all(r['reason']=='local_surrogate_forecast_mismatch' for r in result['history'])
