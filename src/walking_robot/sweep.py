"""Deterministic physical design grids, independent of neural dependencies."""
from dataclasses import asdict,replace,fields
from itertools import product
from pathlib import Path
import numpy as np
import yaml
from .design import SimulationSettings
from .expanded_design import ExpandedDesignRun,ExpandedRobotDesign


def load_sweep(path):
    path = Path(path)
    s = yaml.safe_load(path.read_text())
    required = {'base_design','parameters','simulation','execution','objective','verification'}
    if not isinstance(s,dict) or not required<=s.keys() or set(s)-required-{'design_overrides'}:
        raise ValueError('Sweep YAML requires base_design, parameters, simulation, execution, objective and verification.')
    base = ExpandedDesignRun.from_file(path.resolve().parent/s['base_design'])
    base = replace(base,design=replace(base.design,**s.get('design_overrides',{})),simulation=SimulationSettings(**s['simulation']))
    allowed = {f.name for f in fields(ExpandedRobotDesign)}-{'profile'}
    if not isinstance(s['parameters'],dict) or not s['parameters'] or set(s['parameters'])-allowed:
        raise ValueError('Sweep parameters must be named numeric RobotDesign fields.')
    for name,values in s['parameters'].items():
        if not isinstance(values,list) or not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values):
            raise ValueError(f'{name} requires a nonempty list of finite numeric values.')
        if len(set(values))!=len(values):raise ValueError(f'Duplicate values in {name}.')
    if s['objective'] not in {'mean_com_speed_m_s','late_mean_com_speed_m_s'}:
        raise ValueError('Objective must be mean_com_speed_m_s or late_mean_com_speed_m_s.')
    execution=s['execution'];verification=s['verification']
    if set(execution)!={'batch_size','saved_timestep_s'} or type(execution['batch_size']) is not int or execution['batch_size']<1:
        raise ValueError('execution requires positive integer batch_size and saved_timestep_s.')
    if not np.isfinite(execution['saved_timestep_s']) or execution['saved_timestep_s']<=0:
        raise ValueError('Saved timestep must be finite and positive.')
    if set(verification)!={'top_cases','timesteps_s','max_speed_difference_m_s','max_penetration_difference_m'}:
        raise ValueError('Invalid verification settings.')
    if type(verification['top_cases']) is not int or verification['top_cases']<1:
        raise ValueError('top_cases must be a positive integer.')
    dts=verification['timesteps_s']
    if not isinstance(dts,list) or len(dts)<2 or not np.isfinite(dts).all() or min(dts)<=0 or not np.all(np.diff(dts)<0) or max(dts)>=base.simulation.timestep_s:
        raise ValueError('Verification needs at least two decreasing timesteps below the screening timestep.')
    for name in ['max_speed_difference_m_s','max_penetration_difference_m']:
        if not np.isfinite(verification[name]) or verification[name]<=0:raise ValueError(f'{name} must be positive.')
    for dt in [base.simulation.timestep_s]+dts:
        refined=replace(base.simulation,timestep_s=dt)
        if refined.steps%2:raise ValueError('Use an even number of steps for the late-half objective.')
        saved=execution['saved_timestep_s']/dt
        if saved<1 or not np.isclose(saved,round(saved),rtol=0,atol=1e-9):
            raise ValueError('Saved timestep must be an integer multiple of every integration timestep.')
    names=list(s['parameters'])
    cases=[replace(base.design,**dict(zip(names,values))) for values in product(*(s['parameters'][n] for n in names))]
    return s,base,cases


def rank_cases(rows,objective):
    """A failed trajectory's finite-prefix speed can never enter the ranking."""
    eligible=[r for r in rows if r['validation']['passed'] and
        r['validation']['metrics'].get(objective) is not None and np.isfinite(r['validation']['metrics'][objective])]
    return sorted(eligible,key=lambda r:(-r['validation']['metrics'][objective],r['id']))


def refinement_passed(runs,settings,objective):
    expected=settings['timesteps_s']
    if len(runs)!=len(expected) or sorted(r['dt_s'] for r in runs)!=sorted(expected):return False
    if any(not r['validation']['passed'] for r in runs):return False
    fine,coarse=sorted(runs,key=lambda r:r['dt_s'])[:2]
    a,b=[r['validation']['metrics'] for r in [fine,coarse]]
    values=[a.get(objective),b.get(objective),a.get('max_foot_penetration_m'),b.get('max_foot_penetration_m')]
    return bool(all(v is not None and np.isfinite(v) for v in values) and
        abs(values[0]-values[1])<=settings['max_speed_difference_m_s'] and
        abs(values[2]-values[3])<=settings['max_penetration_difference_m'])


def export_mat(path,configuration,trajectory,*,integration_timestep_s):
    """Exchange data for independent MATLAB checks; this is not validation itself."""
    from scipy.io import savemat
    with np.load(trajectory,allow_pickle=False) as f:t,x,p=[f[k] for k in ['times','states','parameters']]
    design={k:np.nan if v is None else v for k,v in asdict(configuration.design).items()}
    savemat(path,dict(design=design,P=p,x0=x[0],t_jax=t,x_jax=x,
        dt_jax=integration_timestep_s,output_timestep_s=t[1]-t[0],horizon_s=t[-1],
        drive_frequency_hz_applied=p[22],initial_clearance_um=configuration.simulation.initial_clearance_um,
        note='JAX trajectory is downsampled. Validate independently with original MATLAB equations, matching parameters, initial state and resolution.'),do_compression=True)
