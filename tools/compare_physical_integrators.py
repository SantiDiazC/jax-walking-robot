"""Compare RK4, adaptive DOP853/Radau, and smooth variational sensitivities.

Run on CPU to avoid device transfers on each SciPy callback. Adaptive failures
retain their budgets and messages, never a score from a partial trajectory.
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
from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.adaptive import AdaptiveDynamics
from walking_robot.contact_residual import ContactSmoothing,contact_rollout
from walking_robot.geometry import points
from walking_robot.physical_search import SearchSettings,unique_designs,step_count,write_json
from walking_robot.validation import inspect_trajectory


def speed(times,states,p):
    com = (points(states,p)['centers'][...,0]*p[:4]).sum(-1)/p[:4].sum()
    half = (len(times)-1)//2
    return float((com[-1]-com[half])/(times[-1]-times[half]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/physical_search.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/integrator_comparison'))
    parser.add_argument('--search-report',type=Path,help='Also compare the selected physical-search candidate.')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--skip-sensitivities',action='store_true')
    args = parser.parse_args()
    settings,configuration = SearchSettings.from_file(args.study)
    jax.config.update('jax_enable_x64',True)
    args.output.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fingerprint = dict(settings=asdict(settings),configuration=asdict(configuration),jax_version=jax.__version__,
        backend=jax.default_backend(),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')
            if p.name in ['adaptive.py','physical_search.py','parameters.py','contact.py','_generated.py','dynamics.py','integrators.py','initial_pose.py','contact_residual.py','geometry.py','validation.py']},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint = json.loads(json.dumps(fingerprint))
    path = args.output/'measurements.json'
    rows = []
    if path.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume or a new directory.')
        old = json.loads(path.read_text())
        if old['fingerprint']!=fingerprint:
            raise ValueError('Comparison source/settings mismatch; use a new directory.')
        rows = old['measurements']
    def persist(row):
        rows.append(row)
        write_json(path,dict(fingerprint=fingerprint,measurements=rows))
        print(json.dumps({k:v for k,v in row.items() if k not in ['validation','path','key','gradient_per_unit']}),flush=True)
    smoothing = ContactSmoothing(settings.gap_width_um*1e-6,settings.velocity_width_m_s,settings.force_width_uN*1e-6)
    solvers = {m:AdaptiveDynamics(m,smoothing) for m in ['hard','smooth']}
    base = configuration.design
    def parameter_and_initial(d):
        p = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(configuration.simulation.initial_clearance_um*1e-6)
    prepare = jax.jit(parameter_and_initial)
    horizon = settings.comparison_horizon_s
    def adaptive(d,mode,method,rtol,*,sensitivity=False):
        d = np.asarray(d)
        key = json.dumps([d.tolist(),mode,method,rtol,sensitivity])
        previous = next((r for r in rows if r['key']==key),None)
        if previous is not None:
            return previous
        p,x0 = prepare(jnp.asarray(d))
        tangent = {}
        if sensitivity:
            lower = np.asarray([settings.thickness_bounds_um[0],settings.front_angle_bounds_deg[0]])
            span = np.asarray([np.diff(settings.thickness_bounds_um)[0],np.diff(settings.front_angle_bounds_deg)[0]])
            u = (d-lower)/span
            def normalized(u):
                return parameter_and_initial(lower+u*span)
            dp,dx = jax.jit(jax.jacfwd(normalized))(jnp.asarray(u))
            tangent = dict(initial_sensitivity=np.asarray(dx),parameter_sensitivity=np.asarray(dp))
        print(f'Adaptive {mode} {method}: design={d.tolist()}, rtol={rtol:g}, sensitivities={sensitivity}',flush=True)
        info,result = solvers[mode].integrate(x0,p,horizon,method=method,rtol=rtol,
            max_step=settings.comparison_max_step_s,sample_dt=settings.comparison_sample_timestep_s,
            max_rhs_calls=settings.comparison_max_rhs_calls,max_wall_s=settings.comparison_max_wall_s,**tangent)
        row = dict(key=key,design=d.tolist(),method=method,contact_mode=mode,rtol=rtol,sensitivity=sensitivity,solver=info)
        if result is not None:
            p = np.asarray(p)
            validation = inspect_trajectory(result['check_times'],result['check_states'],p,configuration.physical_limits)
            filename = f'adaptive_{len(rows):04d}.npz'
            np.savez_compressed(args.output/filename,t=result['times'],x=result['states'],P=p)
            row.update(validation=validation,speed_m_s=speed(result['times'],result['states'],p),path=filename)
            if sensitivity:
                def com(x,p):
                    return (points(x,p,xp=jnp)['centers'][:,0]*p[:4]).sum()/p[:4].sum()
                half = (len(result['times'])-1)//2
                values = []
                for index in [half,len(result['times'])-1]:
                    xx = jnp.asarray(result['states'][index])
                    ss = jnp.asarray(result['sensitivities'][index])
                    values.append(jax.vmap(lambda sx,sp:jax.jvp(com,(xx,jnp.asarray(p)),(sx,sp))[1],in_axes=(1,1))(ss,jnp.asarray(tangent['parameter_sensitivity'])))
                row['gradient_per_unit'] = np.asarray((values[1]-values[0])/(horizon/2)).tolist()
        persist(row)
        return row
    designs = list(settings.comparison_designs)
    if args.search_report:
        search = json.loads(args.search_report.read_text())
        d = search.get('accepted_design') or search.get('proposed_design')
        if search.get('status')!='complete' or search['fingerprint']['configuration']!=fingerprint['configuration']:
            raise ValueError('Search report is incomplete or has a different base configuration.')
        if d is not None:
            designs.append(d)
    designs = unique_designs(designs)
    modes = ['hard','smooth'] if settings.compare_smooth_contact else ['hard']
    fixed = {}
    for mode in modes:
        for dt in settings.verification_timesteps_s:
            count = step_count(horizon,dt)
            def run(d,mode=mode,dt=dt,count=count):
                p,x = parameter_and_initial(d)
                if mode=='hard':
                    t,x = rollout_scan(x,p,dt,count)
                else:
                    t,x = contact_rollout(x,p,dt,count,mode='smooth',smoothing=smoothing)
                return t,x,p
            fixed[mode,dt] = jax.jit(run)
        for d in designs:
            for dt in settings.verification_timesteps_s:
                key = json.dumps([d.tolist(),mode,'RK4',dt])
                if any(r['key']==key for r in rows):
                    continue
                print(f'Fixed {mode} RK4: design={d.tolist()}, dt={dt:g}',flush=True)
                t,x,p = jax.tree.map(np.asarray,fixed[mode,dt](jnp.asarray(d)))
                filename = f'rk4_{len(rows):04d}.npz'
                stride = max(1,round(settings.comparison_sample_timestep_s/dt))
                np.savez_compressed(args.output/filename,t=t[::stride],x=x[::stride],P=p)
                persist(dict(key=key,design=d.tolist(),method='RK4',contact_mode=mode,dt_s=dt,path=filename,
                    speed_m_s=speed(t,x,p),validation=inspect_trajectory(t,x,p,configuration.physical_limits)))
            for method in settings.comparison_methods:
                for rtol in settings.comparison_rtols:
                    adaptive(d,mode,method,rtol)
    sensitivity = []
    if settings.compare_smooth_contact and not args.skip_sensitivities:
        # Revisit the previously problematic interior design. Perturbations
        # may not cross discrete parameter regimes when certifying this local
        # continuous sensitivity calculation.
        center = np.asarray(settings.comparison_designs[-1])
        span = np.array([np.diff(settings.thickness_bounds_um)[0],np.diff(settings.front_angle_bounds_deg)[0]])
        probes = [center]
        for epsilon in settings.gradient_unit_epsilons:
            for axis in range(2):
                for sign in [-1,1]:
                    delta = np.zeros(2); delta[axis] = sign*epsilon*span[axis]
                    probes.append(center+delta)
        p_all = np.asarray(jax.vmap(lambda d:parameter_and_initial(d)[0])(jnp.asarray(probes)))
        if not np.all(p_all[:,[22,26,41]]==p_all[0,[22,26,41]]):
            raise ValueError('Sensitivity probes cross rounded parameter regimes.')
        for rtol in settings.comparison_rtols:
            variational = adaptive(center,'smooth','DOP853',rtol,sensitivity=True)
            values = [adaptive(d,'smooth','DOP853',rtol) for d in probes]
            complete = 'gradient_per_unit' in variational and all('speed_m_s' in r for r in values)
            case = dict(rtol=rtol,design=center.tolist(),complete=complete,passed=False)
            if complete:
                v = np.array([r['speed_m_s'] for r in values])
                fd = np.array([[(v[1+4*k+2*a+1]-v[1+4*k+2*a])/(2*epsilon) for a in range(2)]
                    for k,epsilon in enumerate(settings.gradient_unit_epsilons)])
                grad = np.asarray(variational['gradient_per_unit'])
                errors = np.linalg.norm(fd-grad,axis=1)/np.maximum(np.linalg.norm(fd,axis=1),1e-8)
                case.update(gradient_per_unit=grad.tolist(),finite_difference_per_unit=fd.tolist(),relative_errors=errors.tolist(),
                    passed=bool(np.isfinite(errors).all() and np.all(errors<=.25) and variational['validation']['passed'] and all(r['validation']['passed'] for r in values)))
            sensitivity.append(case)
    comparisons = []
    for d in designs:
        for mode in modes:
            reference = next(r for r in rows if r['design']==d.tolist() and r['contact_mode']==mode and r['method']=='RK4' and r['dt_s']==min(settings.verification_timesteps_s))
            matches = [r for r in rows if r['design']==d.tolist() and r['contact_mode']==mode and r['method']!='RK4' and not r['sensitivity']]
            comparisons.append(dict(design=d.tolist(),contact_mode=mode,reference_rk4_dt_s=reference['dt_s'],
                reference_speed_m_s=reference['speed_m_s'],adaptive=[dict(method=r['method'],rtol=r['rtol'],success=r['solver']['success'],
                    speed_difference_to_rk4_m_s=r.get('speed_m_s',float('nan'))-reference['speed_m_s']) for r in matches]))
    converged = len(sensitivity)>=2 and all(c['passed'] for c in sensitivity)
    if converged:
        a,b = [np.asarray(c['gradient_per_unit']) for c in [sensitivity[-2],sensitivity[-1]]]
        converged = bool(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-8)<=.25)
    write_json(args.output/'report.json',dict(fingerprint=fingerprint,status='complete',comparisons=comparisons,
        sensitivities=sensitivity,smooth_local_sensitivity_gate_passed=converged,gradient_optimization_ready=False,
        scope='Adaptive smooth sensitivities concern a separate contact model and one local interior design; hard-contact/hardware fidelity is not certified.'))
    print(json.dumps(dict(smooth_local_sensitivity_gate_passed=converged,gradient_optimization_ready=False)),flush=True)


if __name__=='__main__':
    main()
