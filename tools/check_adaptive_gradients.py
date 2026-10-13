"""Focused, tolerance-refined smooth-contact design-gradient validation.

Optionally reuse completed same-settings/RHS state integrations as immutable
reference measurements. Failed or partial integrations are never reused.
This local check does not enable hard-contact or neural design optimization.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
from walking_robot import solve_initial_pose
from walking_robot.adaptive import AdaptiveDynamics
from walking_robot.contact_residual import ContactSmoothing
from walking_robot.geometry import points
from walking_robot.physical_search import SearchSettings,inside_domain,write_json
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/physical_search.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/adaptive_gradients'))
    parser.add_argument('--reference',type=Path,help='Previously completed integrator-comparison measurements.')
    parser.add_argument('--rtols',type=float,nargs='+',default=[1e-7,1e-8])
    parser.add_argument('--max-rhs-calls',type=int,default=600000)
    parser.add_argument('--direction-step',type=float,default=.001,help='Step in normalized design coordinates.')
    args = parser.parse_args()
    settings,config = SearchSettings.from_file(args.study)
    if len(args.rtols)<2 or not np.isfinite(args.rtols).all() or min(args.rtols)<=0 or not np.all(np.diff(args.rtols)<0):
        parser.error('Use at least two strictly decreasing positive tolerances.')
    if args.max_rhs_calls<1 or not np.isfinite(args.direction_step) or args.direction_step<=0:
        parser.error('Budgets and direction step must be positive.')
    if (args.output/'report.json').exists():
        parser.error('Use a fresh output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    original,reference_sha = [],None
    if args.reference:
        recorded = json.loads(args.reference.read_text())
        if recorded['fingerprint']['settings']!=json.loads(json.dumps(asdict(settings))) or recorded['fingerprint']['configuration']!=json.loads(json.dumps(asdict(config))):
            raise ValueError('Reference settings differ from this study.')
        if any(sources.get(name)!=digest for name,digest in recorded['fingerprint']['sources'].items()):
            raise ValueError('Reference physical source hashes differ.')
        if recorded['fingerprint']['backend']!=jax.default_backend():
            raise ValueError('Reference backend differs.')
        original = recorded['measurements']
        reference_sha = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    center = np.asarray(settings.comparison_designs[-1])
    lower = np.array([settings.thickness_bounds_um[0],settings.front_angle_bounds_deg[0]])
    span = np.array([np.diff(settings.thickness_bounds_um)[0],np.diff(settings.front_angle_bounds_deg)[0]])
    def prepare(u):
        d = lower+u*span
        p = config.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
    get = jax.jit(prepare)
    derivative = jax.jit(jax.jacfwd(prepare))
    solver = AdaptiveDynamics('smooth',ContactSmoothing(settings.gap_width_um*1e-6,settings.velocity_width_m_s,settings.force_width_uN*1e-6))
    def com(x,p):
        return (points(x,p,xp=jnp)['centers'][:,0]*p[:4]).sum()/p[:4].sum()
    rows = []
    def measure(d,rtol,tangent=False):
        if not inside_domain(d,settings):
            raise ValueError('Probe leaves the design domain.')
        previous = next((r for r in original if r['design']==d.tolist() and r.get('contact_mode')=='smooth' and
            r['method']=='DOP853' and r['rtol']==rtol and not r['sensitivity'] and not tangent and r['solver']['success']),None)
        if previous:
            row = dict(previous,source='Completed immutable reference integration')
        else:
            p,x = get(jnp.asarray((d-lower)/span))
            kwargs = {}
            if tangent:
                dp,dx = derivative(jnp.asarray((d-lower)/span))
                kwargs = dict(initial_sensitivity=dx,parameter_sensitivity=dp)
            print(f'Smooth adaptive check: design={d.tolist()}, rtol={rtol:g}, tangent={tangent}',flush=True)
            info,result = solver.integrate(x,p,settings.comparison_horizon_s,rtol=rtol,
                max_step=settings.comparison_max_step_s,sample_dt=settings.comparison_sample_timestep_s,
                max_rhs_calls=args.max_rhs_calls,max_wall_s=settings.comparison_max_wall_s,**kwargs)
            row = dict(design=d.tolist(),rtol=rtol,sensitivity=tangent,solver=info,source='New completed/failed integration')
            if result is not None:
                half = (len(result['times'])-1)//2
                duration = settings.comparison_horizon_s/2
                value = jax.vmap(lambda xx:com(xx,p))(jnp.asarray(result['states'][[half,-1]]))
                row.update(speed_m_s=float((value[-1]-value[0])/duration),
                    validation=inspect_trajectory(result['check_times'],result['check_states'],p,config.physical_limits))
                if tangent:
                    values = []
                    for k in [half,len(result['times'])-1]:
                        values.append(jax.vmap(lambda sx,sp:jax.jvp(com,(jnp.asarray(result['states'][k]),p),(sx,sp))[1],in_axes=(1,1))(jnp.asarray(result['sensitivities'][k]),dp))
                    row['gradient_per_unit'] = np.asarray((values[-1]-values[0])/duration).tolist()
        rows.append(row)
        write_json(args.output/'measurements.json',dict(configuration=asdict(settings),reference_sha256=reference_sha,measurements=rows))
        return row
    probes = [center]
    for eps in settings.gradient_unit_epsilons:
        for axis in range(2):
            for sign in [-1,1]:
                shift = np.zeros(2); shift[axis] = sign*eps*span[axis]
                probes.append(center+shift)
    parameters = np.asarray(jax.vmap(lambda d:prepare((d-lower)/span)[0])(jnp.asarray(probes)))
    if not np.all(parameters[:,[22,26,41]]==parameters[0,[22,26,41]]):
        raise ValueError('Probes cross discrete frequency/stiffness regimes.')
    cases = []
    for rtol in args.rtols:
        tangent = measure(center,rtol,True)
        physical = [measure(d,rtol) for d in probes]
        complete = 'gradient_per_unit' in tangent and all('speed_m_s' in r for r in physical)
        case = dict(rtol=rtol,complete=complete,passed=False)
        if complete:
            v = np.array([r['speed_m_s'] for r in physical])
            fd = np.array([[(v[1+4*k+2*a+1]-v[1+4*k+2*a])/(2*eps) for a in range(2)] for k,eps in enumerate(settings.gradient_unit_epsilons)])
            grad = np.asarray(tangent['gradient_per_unit'])
            error = np.linalg.norm(fd-grad,axis=1)/np.maximum(np.linalg.norm(fd,axis=1),1e-8)
            case.update(gradient_per_unit=grad.tolist(),finite_difference_per_unit=fd.tolist(),relative_errors=error.tolist(),
                objective_m_s=float(v[0]),passed=bool(np.all(error<=.25) and tangent['validation']['passed'] and all(r['validation']['passed'] for r in physical)))
        cases.append(case)
        print(json.dumps(case),flush=True)
    converged,direction = False,None
    if all(c['passed'] for c in cases):
        g0,g1 = [np.asarray(c['gradient_per_unit']) for c in cases[-2:]]
        error = float(np.linalg.norm(g0-g1)/max(np.linalg.norm(g1),1e-8))
        delta = args.direction_step*g1/max(np.linalg.norm(g1),1e-8)
        moved = measure(center+delta*span,args.rtols[-1])
        predicted = float(g1@delta)
        actual = moved.get('speed_m_s',float('nan'))-cases[-1]['objective_m_s']
        direction = dict(design=(center+delta*span).tolist(),predicted_gain_m_s=predicted,actual_gain_m_s=actual,
            passed=bool(np.isfinite(actual) and predicted>0 and actual>0 and abs(actual-predicted)/predicted<=.25 and moved['validation']['passed']))
        converged = bool(error<=.25 and direction['passed'])
    report = dict(configuration=asdict(settings),design=center.tolist(),horizon_s=settings.comparison_horizon_s,
        rtols=args.rtols,max_rhs_calls=args.max_rhs_calls,unit_epsilons=settings.gradient_unit_epsilons,cases=cases,
        gradient_tolerance_relative_difference=error if all(c['passed'] for c in cases) else None,
        direction_check=direction,local_smooth_gradient_gate_passed=converged,gradient_optimization_ready=False,
        reference_sha256=reference_sha,source_sha256=sources,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One interior design and 0.2-second smooth-contact objective only. No hard-contact, long-horizon, neural or hardware gradient certification.')
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(local_smooth_gradient_gate_passed=converged)),flush=True)


if __name__=='__main__':
    main()
