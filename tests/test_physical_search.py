import numpy as np
from walking_robot.physical_search import SearchSettings,grid_designs,refinement_designs,paired_acceptance


def paired_group(gain=.004,baseline_shift=1e-5,candidate_shift=2e-5):
    settings = SearchSettings()
    timesteps = [5e-6,2.5e-6]
    runs = []
    for dt in timesteps:
        for label in ['baseline','candidate']:
            speed = .74+(gain if label=='candidate' else 0)
            if dt==timesteps[0]:
                speed += candidate_shift if label=='candidate' else baseline_shift
            runs.append(dict(label=label,dt_s=dt,validation=dict(passed=True,
                metrics=dict(late_mean_com_speed_m_s=speed,max_foot_penetration_m=.0002))))
    return settings,dict(horizon_s=1.,condition_id=0,required_timesteps_s=timesteps,runs=runs)


def test_acceptance_rejects_an_improvement_smaller_than_numerical_changes():
    settings,group = paired_group(gain=.001,baseline_shift=.0004,candidate_shift=.0004)
    result = paired_acceptance([group],settings)
    assert not result['passed']
    assert result['checks'][0]['required_improvement_m_s']>.001


def test_acceptance_requires_complete_feasible_pairs_and_all_conditions():
    settings,good = paired_group()
    assert paired_acceptance([good],settings)['passed']
    _,missing = paired_group()
    missing['runs'].pop()
    assert not paired_acceptance([missing],settings)['passed']
    _,bad = paired_group()
    bad['condition_id'] = 1
    bad['runs'][0]['validation']['passed'] = False
    assert not paired_acceptance([good,bad],settings)['passed']
    assert not paired_acceptance([],settings)['passed']


def test_acceptance_rejects_a_gain_that_changes_sign_with_timestep():
    settings,group = paired_group(gain=.0007,candidate_shift=-.0008)
    assert not paired_acceptance([group],settings)['passed']


def test_local_refinement_stays_in_domain_and_avoids_remeasuring_grid():
    settings = SearchSettings()
    initial = grid_designs(settings,[1.,95.])
    refined = refinement_designs(settings,[2.,100.],initial)
    assert len(initial)==36 and len(refined)>0
    assert np.all((refined>=[.5,90])&(refined<=[2.,100]))
    assert all(not np.any(np.all(np.isclose(initial,d,rtol=0,atol=1e-12),axis=1)) for d in refined)
