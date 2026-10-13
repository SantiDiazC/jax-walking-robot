"""Small neural proposals accepted exclusively by independent physical evidence."""
from dataclasses import dataclass
import numpy as np
from .physical_search import paired_acceptance


@dataclass(frozen=True)
class LocalSettings:
    horizon_s: float = 2.
    timesteps_s: tuple = (2.5e-6,1.25e-6)
    iterations: int = 3
    backtracks: int = 3
    unit_step: float = .003
    minimum_unit_step: float = .00025
    trust_radius: float = .01
    minimum_improvement_m_s: float = .0001
    uncertainty_factor: float = 2.
    max_speed_timestep_difference_m_s: float = .001
    max_penetration_timestep_difference_um: float = 2.
    max_forecast_relative_error: float = .25
    max_surrogate_speed_error_m_s: float = .0025

    def __post_init__(self):
        for name in ['iterations','backtracks']:
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer.')
        values = [v for k,v in self.__dict__.items() if k not in ['timesteps_s','iterations','backtracks']]
        if not np.isfinite(values).all() or min(values)<=0:
            raise ValueError('Local optimization settings must be finite and positive.')
        if (len(self.timesteps_s)<2 or not np.isfinite(self.timesteps_s).all()
            or min(self.timesteps_s)<=0 or not np.all(np.diff(self.timesteps_s)<0)):
            raise ValueError('Use decreasing positive verification timesteps.')
        if not self.minimum_unit_step<=self.unit_step<=.003 or self.trust_radius>.01:
            raise ValueError('Step/radius exceed the tested local protocol.')


def physical_speed(runs,settings):
    """A partial trajectory, duplicate run or failed physical limit is unusable."""
    if (len(runs)!=len(settings.timesteps_s) or any(sum(r['dt_s']==dt for r in runs)!=1 for dt in settings.timesteps_s)
        or not all(r['validation']['passed'] for r in runs)):
        raise ValueError('Incomplete or physically inadmissible reference.')
    values = [r['validation']['metrics'].get('late_mean_com_speed_m_s') for r in runs]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Nonfinite physical objective.')
    return min(runs,key=lambda r:r['dt_s'])['validation']['metrics']['late_mean_com_speed_m_s']


def verify_pair(baseline,candidate,settings,*,horizon=None,condition_id=0):
    return paired_acceptance([dict(horizon_s=settings.horizon_s if horizon is None else horizon,
        condition_id=condition_id,required_timesteps_s=list(settings.timesteps_s),
        runs=[dict(r,label=label) for label,rows in [('baseline',baseline),('candidate',candidate)] for r in rows])],settings)


def verified_local_search(initial,neural,evaluate,same_regime,settings,*,on_iteration=None):
    """Bounded monotonic search with callbacks for proposals and physical runs.

    Coordinates span [0,1] in the trained design domain. Rejected trials never
    update the accepted state. Every accepted update must improve the original
    physical objective beyond an empirical numerical-resolution screen.
    """
    initial = np.asarray(initial,dtype=float)
    if initial.shape!=(2,) or not np.isfinite(initial).all() or not np.all((initial>=0)&(initial<=1)):
        raise ValueError('Initial design must lie inside the normalized domain.')
    current = initial.copy()
    reference = evaluate(current)
    start_speed = physical_speed(reference,settings)
    history = []
    status = 'iteration_limit'
    for iteration in range(settings.iterations):
        value,gradient = neural(current)
        gradient = np.asarray(gradient,dtype=float)
        norm = np.linalg.norm(gradient)
        if gradient.shape!=(2,) or not np.isfinite(value) or not np.isfinite(norm) or norm<1e-8:
            status = 'invalid_neural_direction';break
        if abs(value-physical_speed(reference,settings))>settings.max_surrogate_speed_error_m_s:
            status = 'local_surrogate_speed_mismatch';break
        step = settings.unit_step
        accepted = False
        for attempt in range(settings.backtracks):
            if step<settings.minimum_unit_step:
                break
            proposal = current+step*gradient/norm
            row = dict(iteration=iteration+1,attempt=attempt+1,unit_step=step,
                from_unit=current.tolist(),proposed_unit=proposal.tolist(),accepted=False)
            in_domain = bool(np.all((proposal>=0)&(proposal<=1)))
            in_radius = bool(np.linalg.norm(proposal-initial)<=settings.trust_radius+1e-12)
            if not in_domain or not in_radius or not same_regime(initial,proposal):
                row['reason'] = 'domain_trust_radius_or_discrete_regime'
            else:
                proposed_value,_ = neural(proposal)
                prediction = norm*step
                neural_gain = proposed_value-value
                row.update(first_order_gain_m_s=float(prediction),neural_gain_m_s=float(neural_gain),
                    gradient_per_unit=gradient.tolist())
                if not np.isfinite(proposed_value) or abs(neural_gain-prediction)>settings.max_forecast_relative_error*prediction:
                    row['reason'] = 'neural_local_nonlinearity'
                else:
                    trial = evaluate(proposal)
                    outcome = verify_pair(reference,trial,settings)
                    row['physical_check'] = outcome
                    if not outcome['passed']:
                        row['reason'] = 'physical_improvement_or_resolution_check_failed'
                    else:
                        speed = physical_speed(trial,settings)
                        gain = speed-physical_speed(reference,settings)
                        error = abs(gain-prediction)/prediction
                        row.update(physical_gain_m_s=float(gain),forecast_relative_error=float(error),
                                   physical_speed_m_s=float(speed))
                        if error>settings.max_forecast_relative_error or abs(proposed_value-speed)>settings.max_surrogate_speed_error_m_s:
                            row['reason'] = 'local_surrogate_forecast_mismatch'
                        else:
                            row['accepted'] = accepted = True
                            current,reference = proposal,trial
            history.append(row)
            if on_iteration is not None:
                on_iteration(row)
            if accepted:
                break
            step *= .5
        if not accepted:
            status = 'no_verified_step';break
    return dict(status=status,initial_unit=initial.tolist(),final_unit=current.tolist(),history=history,
        accepted_steps=sum(r['accepted'] for r in history),initial_speed_m_s=float(start_speed),
        final_speed_m_s=float(physical_speed(reference,settings)),
        improvement_m_s=float(physical_speed(reference,settings)-start_speed))
