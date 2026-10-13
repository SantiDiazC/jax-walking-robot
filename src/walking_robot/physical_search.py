"""Derivative-free physical reference search and conservative acceptance rules."""
from dataclasses import dataclass,asdict,field
from pathlib import Path
import json
import numpy as np
import yaml
from .design import DesignRun


@dataclass(frozen=True)
class SearchSettings:
    design: str = '../designs/matlab_supported.yaml'
    seed: int = 42
    thickness_bounds_um: list = field(default_factory=lambda:[.5,2.])
    front_angle_bounds_deg: list = field(default_factory=lambda:[90.,100.])
    grid_counts: list = field(default_factory=lambda:[5,7])
    refinement_count: int = 5
    batch_size: int = 6
    search_horizon_s: float = 1.
    search_timestep_s: float = 1e-5
    finalist_count: int = 3
    verification_horizons_s: list = field(default_factory=lambda:[1.,2.])
    verification_timesteps_s: list = field(default_factory=lambda:[1e-5,5e-6,2.5e-6])
    minimum_improvement_m_s: float = .0005
    uncertainty_factor: float = 2.
    max_speed_timestep_difference_m_s: float = .001
    max_penetration_timestep_difference_um: float = 2.
    robustness_variants: int = 3
    robustness_horizon_s: float = 1.
    robustness_timesteps_s: list = field(default_factory=lambda:[5e-6,2.5e-6])
    height_perturbation_um: float = 20.
    tilt_perturbation_deg: float = .05
    speed_perturbation_m_s: float = .01
    comparison_horizon_s: float = .2
    comparison_designs: list = field(default_factory=lambda:[[1.,95.],[1.5237432559624633,91.3975248360931]])
    comparison_methods: list = field(default_factory=lambda:['DOP853','Radau'])
    comparison_rtols: list = field(default_factory=lambda:[1e-5,1e-7])
    comparison_max_step_s: float = .0002
    comparison_sample_timestep_s: float = 5e-6
    comparison_max_rhs_calls: int = 300000
    comparison_max_wall_s: float = 180.
    compare_smooth_contact: bool = True
    gap_width_um: float = 2.
    velocity_width_m_s: float = 1e-4
    force_width_uN: float = 1.
    gradient_unit_epsilons: list = field(default_factory=lambda:[.01,.001])

    @classmethod
    def from_file(cls,path):
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values,dict):
            raise ValueError('Study YAML must contain a mapping.')
        result = cls(**values)
        result.validate()
        configuration = DesignRun.from_file(Path(path).resolve().parent/result.design)
        if configuration.design.profile!='matlab':
            raise ValueError('Physical reference search requires the reconciled MATLAB profile.')
        if not inside_domain([configuration.design.adhesive_thickness_um,configuration.design.front_angle_deg],result):
            raise ValueError('Baseline must lie inside the search domain.')
        return result,configuration

    def validate(self):
        for name in ['refinement_count','batch_size','finalist_count','robustness_variants','comparison_max_rhs_calls']:
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer.')
        if type(self.seed) is not int or self.seed<0 or len(self.grid_counts)!=2 or any(type(n) is not int or n<2 for n in self.grid_counts):
            raise ValueError('Invalid seed or grid counts.')
        for name in ['thickness_bounds_um','front_angle_bounds_deg']:
            bounds = getattr(self,name)
            if len(bounds)!=2 or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:
                raise ValueError(f'Invalid {name}.')
        if self.front_angle_bounds_deg[-1]>=180:
            raise ValueError('Leg angles must be below 180 degrees.')
        for name in ['search_horizon_s','search_timestep_s','minimum_improvement_m_s','uncertainty_factor',
                     'max_speed_timestep_difference_m_s','max_penetration_timestep_difference_um','robustness_horizon_s',
                     'comparison_horizon_s','comparison_max_step_s','comparison_sample_timestep_s','comparison_max_wall_s',
                     'gap_width_um','velocity_width_m_s','force_width_uN']:
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ['height_perturbation_um','tilt_perturbation_deg','speed_perturbation_m_s']:
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        for name in ['verification_timesteps_s','robustness_timesteps_s','comparison_rtols']:
            values = getattr(self,name)
            if len(values)<2 or not np.isfinite(values).all() or min(values)<=0 or not np.all(np.diff(values)<0):
                raise ValueError(f'{name} needs at least two strictly decreasing positive values.')
        if len(self.verification_horizons_s)<2 or not np.isfinite(self.verification_horizons_s).all() or not np.all(np.diff(self.verification_horizons_s)>0):
            raise ValueError('Verification needs increasing positive horizons.')
        for horizon in self.verification_horizons_s:
            for dt in self.verification_timesteps_s:
                step_count(horizon,dt)
        for dt in self.robustness_timesteps_s:
            step_count(self.robustness_horizon_s,dt)
        step_count(self.search_horizon_s,self.search_timestep_s)
        step_count(self.comparison_horizon_s,self.comparison_sample_timestep_s)
        if self.search_horizon_s not in self.verification_horizons_s:
            raise ValueError('Verification must include the search horizon.')
        if not self.comparison_methods or not set(self.comparison_methods)<={'DOP853','Radau'}:
            raise ValueError('Supported adaptive comparators: DOP853 and Radau.')
        if type(self.compare_smooth_contact) is not bool or len(self.gradient_unit_epsilons)<2 or any(not np.isfinite(e) or not 0<e<.1 for e in self.gradient_unit_epsilons):
            raise ValueError('Invalid smoothing flag or sensitivity perturbations.')
        if any(len(d)!=2 or not inside_domain(d,self) for d in self.comparison_designs):
            raise ValueError('Comparison designs must lie inside the domain.')


def step_count(horizon,dt):
    if not np.isfinite(horizon) or not np.isfinite(dt) or horizon<=0 or dt<=0:
        raise ValueError('Horizon and timestep must be finite and positive.')
    n = round(horizon/dt)
    if n<2 or n%2 or not np.isclose(n*dt,horizon,rtol=0,atol=1e-12):
        raise ValueError('Horizon must contain an even integer number of steps for the late-half objective.')
    return n


def inside_domain(design,settings):
    return bool(np.isfinite(design).all() and all(a<=v<=b for v,(a,b) in zip(design,
        [settings.thickness_bounds_um,settings.front_angle_bounds_deg])))


def grid_designs(settings,baseline):
    a = np.linspace(*settings.thickness_bounds_um,settings.grid_counts[0])
    b = np.linspace(*settings.front_angle_bounds_deg,settings.grid_counts[1])
    return unique_designs([baseline]+[[x,y] for x in a for y in b])


def unique_designs(designs):
    result = []
    for d in designs:
        if not any(np.allclose(d,e,rtol=0,atol=1e-12) for e in result):
            result.append(list(d))
    return np.asarray(result)


def refinement_designs(settings,center,previous):
    axes = []
    for i,bounds in enumerate([settings.thickness_bounds_um,settings.front_angle_bounds_deg]):
        radius = (bounds[1]-bounds[0])/(settings.grid_counts[i]-1)
        axes.append(np.linspace(max(bounds[0],center[i]-radius),min(bounds[1],center[i]+radius),settings.refinement_count))
    return np.asarray([d for d in unique_designs([[x,y] for x in axes[0] for y in axes[1]])
                       if not any(np.allclose(d,e,rtol=0,atol=1e-12) for e in previous)]).reshape(-1,2)


def paired_acceptance(rows,settings):
    """Require complete pairs and gains larger than observed refinement changes.

    The uncertainty estimate is a conservative empirical screen, not a
    mathematical error bound. Each row corresponds to one horizon/condition
    and contains exactly one baseline/candidate pair at every requested dt.
    """
    checks = []
    for group in rows:
        expected = group['required_timesteps_s']
        runs = group['runs']
        complete = len(runs)==2*len(expected) and all(
            sum(r['label']==label and r['dt_s']==dt for r in runs)==1
            for dt in expected for label in ['baseline','candidate'])
        finite = complete and all(r['validation']['passed'] and
            r['validation']['metrics'].get('late_mean_com_speed_m_s') is not None and
            np.isfinite(r['validation']['metrics']['late_mean_com_speed_m_s']) for r in runs)
        check = dict(horizon_s=group['horizon_s'],condition_id=group['condition_id'],complete=bool(complete),passed=False)
        if finite:
            by_dt = {dt:{r['label']:r['validation']['metrics'] for r in runs if r['dt_s']==dt} for dt in expected}
            coarse,fine = sorted(expected)[:2][::-1]
            uncertainty = sum(abs(by_dt[fine][label]['late_mean_com_speed_m_s']-by_dt[coarse][label]['late_mean_com_speed_m_s']) for label in ['baseline','candidate'])
            gain = by_dt[fine]['candidate']['late_mean_com_speed_m_s']-by_dt[fine]['baseline']['late_mean_com_speed_m_s']
            speed_change = max(abs(by_dt[fine][label]['late_mean_com_speed_m_s']-by_dt[coarse][label]['late_mean_com_speed_m_s']) for label in ['baseline','candidate'])
            penetration_change = max(abs(by_dt[fine][label]['max_foot_penetration_m']-by_dt[coarse][label]['max_foot_penetration_m']) for label in ['baseline','candidate'])
            positive_every_dt = all(p['candidate']['late_mean_com_speed_m_s']>p['baseline']['late_mean_com_speed_m_s'] for p in by_dt.values())
            threshold = max(settings.minimum_improvement_m_s,settings.uncertainty_factor*uncertainty)
            check.update(improvement_m_s=gain,estimated_pair_uncertainty_m_s=uncertainty,
                required_improvement_m_s=threshold,finest_speed_change_m_s=speed_change,
                finest_penetration_change_m=penetration_change,
                passed=bool(positive_every_dt and gain>threshold and speed_change<=settings.max_speed_timestep_difference_m_s and
                    penetration_change<=settings.max_penetration_timestep_difference_um*1e-6))
        else:
            check['reason'] = 'Missing pair, nonfinite objective, or provisional physical limit violation.'
        checks.append(check)
    return dict(passed=bool(checks) and all(c['passed'] for c in checks),checks=checks,
        uncertainty_scope='Empirical last-two-timestep changes; not a rigorous error bound.')


def write_json(path,value):
    def clean(v):
        if isinstance(v,dict):
            return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v,np.integer):
            return int(v)
        if isinstance(v,np.bool_):
            return bool(v)
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
