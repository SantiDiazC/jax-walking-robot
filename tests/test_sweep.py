from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import yaml
from walking_robot.sweep import load_sweep,rank_cases,refinement_passed,export_mat


CONFIG=Path(__file__).resolve().parents[1]/'examples/sweeps/matlab_25_cases.yaml'


def test_matlab_grid_has_25_cases_and_explicit_original_controls():
    s,base,cases=load_sweep(CONFIG)
    assert len(cases)==25 and cases[0].damping_ratio==.01 and cases[-1].damping_ratio==.09
    assert cases[0].adhesive_thickness_um==1 and cases[-1].adhesive_thickness_um==50
    assert all(d.voltage_v==200 and d.drive_frequency_hz is None for d in cases)
    assert base.simulation.initial_clearance_um==501 and base.simulation.duration_s==1
    assert float(cases[0].parameters()[22])!=float(cases[-1].parameters()[22])


def test_supported_sweep_changes_initial_clearance_only():
    _,dropped,a=load_sweep(CONFIG)
    _,supported,b=load_sweep(CONFIG.parent/'supported_25_cases.yaml')
    assert a==b and dropped.simulation.initial_clearance_um==501 and supported.simulation.initial_clearance_um==0


@pytest.mark.parametrize('mutation',[lambda s:s['parameters'].update(bad_field=[1]),
    lambda s:s['parameters'].update(damping_ratio=[.01,.01]),lambda s:s['parameters'].update(damping_ratio=[True]),
    lambda s:s['verification'].update(timesteps_s=[.000005,.000007]),
    lambda s:s['execution'].update(saved_timestep_s=.000013)])
def test_invalid_sweeps_fail_before_simulation(tmp_path,mutation):
    s=yaml.safe_load(CONFIG.read_text());s['base_design']=str(CONFIG.parent/s['base_design']);mutation(s)
    path=tmp_path/'sweep.yaml';path.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError):load_sweep(path)


def test_ranking_excludes_failed_finite_prefix_and_missing_speeds():
    def row(i,v,passed=True):return dict(id=i,validation=dict(passed=passed,metrics={'late_mean_com_speed_m_s':v}))
    rows=[row(0,1),row(1,100,False),row(2,None),row(3,float('nan')),row(4,.5),row(5,1)]
    assert [r['id'] for r in rank_cases(rows,'late_mean_com_speed_m_s')]==[0,5,4]


def test_refinement_requires_full_physical_pairs_and_resolution_agreement():
    settings=dict(timesteps_s=[5e-6,2.5e-6],max_speed_difference_m_s=.001,max_penetration_difference_m=2e-6)
    def row(dt,v,passed=True):return dict(dt_s=dt,validation=dict(passed=passed,metrics=dict(late_mean_com_speed_m_s=v,max_foot_penetration_m=.0002)))
    a,b=row(5e-6,.8),row(2.5e-6,.80001)
    assert refinement_passed([a,b],settings,'late_mean_com_speed_m_s')
    assert not refinement_passed([a],settings,'late_mean_com_speed_m_s')
    assert not refinement_passed([a,row(2.5e-6,.9)],settings,'late_mean_com_speed_m_s')
    assert not refinement_passed([a,row(2.5e-6,.8,False)],settings,'late_mean_com_speed_m_s')


def test_matlab_exchange_preserves_named_inputs_and_sampling(tmp_path):
    from scipy.io import loadmat
    _,base,cases=load_sweep(CONFIG)
    p=np.asarray(cases[0].parameters());t=np.array([0.,.0001,.0002]);x=np.zeros((3,14))
    np.savez(tmp_path/'trajectory.npz',times=t,states=x,parameters=p)
    export_mat(tmp_path/'case.mat',base,tmp_path/'trajectory.npz',integration_timestep_s=2.5e-6)
    result=loadmat(tmp_path/'case.mat',simplify_cells=True)
    assert result['design']['adhesive_thickness_um']==1 and np.isnan(result['design']['drive_frequency_hz'])
    assert result['drive_frequency_hz_applied']==p[22] and result['dt_jax']==2.5e-6
    assert result['output_timestep_s']==.0001
    np.testing.assert_array_equal(result['x_jax'],x)
