"""Refine physical design finite differences; retain all reference failures."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import time
from dataclasses import asdict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.design import DesignRun
from walking_robot.geometry import points
from walking_robot.validation import inspect_trajectory
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.contact_residual import ContactSmoothing,contact_rollout


def write(path,report):
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):
            return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(report),indent=2,allow_nan=False)+"\n")


def probes_for(center,epsilons):
    span = np.asarray(DESIGN_UPPER)-DESIGN_LOWER
    probes = [center]
    for epsilon in epsilons:
        for axis in range(2):
            for sign in [-1,1]:
                probe = center.copy()
                probe[axis] += sign*epsilon*span[axis]
                probes.append(probe)
    return np.asarray(probes)


def finite_differences(values,epsilons):
    return [[(values[1+4*k+2*a+1]-values[1+4*k+2*a])/(2*epsilon) for a in range(2)]
            for k,epsilon in enumerate(epsilons)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/contact_structure.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/contact_sensitivity'))
    parser.add_argument('--mode',choices=['hard','smooth'],default='hard')
    parser.add_argument('--timesteps',type=float,nargs='+',help='Override YAML timesteps for a focused diagnostic.')
    parser.add_argument('--autodiff',action='store_true',help='Check physical forward-mode autodiff at the finest timestep.')
    args = parser.parse_args()
    settings = yaml.safe_load(args.study.read_text())
    root = args.study.resolve().parent
    configuration = DesignRun.from_file(root/settings['design'])
    epsilons = settings['gradient_unit_epsilons']
    timesteps = args.timesteps or settings['physical_timesteps_s']
    smoothing = ContactSmoothing(settings['gap_width_um']*1e-6,settings['velocity_width_m_s'],settings['force_width_uN']*1e-6)
    if any(not np.isfinite(e) or e<=0 for e in epsilons+timesteps):
        parser.error('Finite-difference perturbations and timesteps must be positive.')
    if len(timesteps)<2 or len(epsilons)<2 or not np.all(np.diff(timesteps)<0):
        parser.error('Use at least two perturbations and two strictly decreasing timesteps.')
    dataset_path = root/settings['reference_dataset']
    with np.load(dataset_path,allow_pickle=False) as saved:
        data = {k:saved[k] for k in saved.files}
    dataset_metadata = json.loads(dataset_path.with_name('dataset_report.json').read_text())
    recorded = dataset_metadata['fingerprint']['configuration']
    if recorded['design']!=asdict(configuration.design) or recorded['simulation']['initial_clearance_um']!=configuration.simulation.initial_clearance_um:
        raise ValueError('Reference dataset fixed design inputs/initial clearance differ from the study.')
    prior_path = root/settings['prior_gradient_report']
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {'cases':[]}
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    base = configuration.design
    count_by_dt = {dt:round(settings['horizon_s']/dt) for dt in timesteps}
    if any(not np.isclose(count*dt,settings['horizon_s'],rtol=0,atol=1e-12) for dt,count in count_by_dt.items()):
        parser.error('Horizon must be an integer multiple of every timestep.')
    cases = []
    for design_id in settings['gradient_design_ids']:
        ids = np.flatnonzero((data['design_id']==design_id)&(data['variant']==0))
        if len(ids)!=1:
            raise ValueError('Each design ID must have one canonical trajectory.')
        center = data['designs'][ids[0]]
        actual_p = np.asarray(base.parameters(adhesive_thickness_um=center[0],front_angle_deg=center[1]))
        np.testing.assert_allclose(actual_p,data['parameters'][ids[0]],rtol=1e-12,atol=1e-12)
        probes = probes_for(center,epsilons)
        if not np.all((probes>=DESIGN_LOWER)&(probes<=DESIGN_UPPER)):
            raise ValueError('Finite differences leave the design domain.')
        previous = next((c for c in prior['cases'] if c['design_id']==design_id),None)
        rows = []
        for dt,count in count_by_dt.items():
            cached = next((r for r in previous['physical'] if r['dt_s']==dt),None) if previous and args.mode=='hard' else None
            if cached and previous['design']==center.tolist() and prior['horizon_s']==settings['horizon_s'] and [c['unit_epsilon'] for c in previous['checks']]==epsilons:
                row = dict(cached,source='Previously measured hard-contact reference; original report retained.')
            else:
                def trajectory(d):
                    p = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
                    x0 = solve_initial_pose(p).at[1].add(configuration.simulation.initial_clearance_um*1e-6)
                    if args.mode=='hard':
                        t,x = rollout_scan(x0,p,dt=dt,steps=count)
                    else:
                        t,x = contact_rollout(x0,p,dt,count,mode='smooth',smoothing=smoothing)
                    return t,x,p
                print(f'{args.mode} contact: design {design_id}, dt={dt:g}, {count} steps...',flush=True)
                started = time.perf_counter()
                t,x,p = jax.jit(jax.vmap(trajectory))(jnp.asarray(probes))
                t,x,p = np.asarray(t),np.asarray(x),np.asarray(p)
                centers = points(x,p[:,None,:])['centers'][...,0]
                com = (centers*p[:,None,:4]).sum(-1)/p[:,:4].sum(-1)[:,None]
                values = (com[:,-1]-com[:,count//2])/(t[0,-1]-t[0,count//2])
                reports = [inspect_trajectory(tt,xx,pp,configuration.physical_limits) for tt,xx,pp in zip(t,x,p)]
                row = dict(dt_s=dt,values_m_s=values.tolist(),all_probes_physically_valid=all(r['passed'] for r in reports),
                           physical_reports=reports,elapsed_s=time.perf_counter()-started,source=f'New {args.mode}-contact JAX RK4 measurement')
            row['finite_difference_per_unit'] = finite_differences(row['values_m_s'],epsilons)
            rows.append(row)
            cases_now = cases+[dict(design_id=design_id,design=center.tolist(),measurements=rows)]
            write(args.output/'report.json',dict(status='in_progress',cases=cases_now))
            print(json.dumps(dict(design_id=design_id,dt_s=dt,gradients=row['finite_difference_per_unit'])),flush=True)
        fine = np.asarray(rows[-1]['finite_difference_per_unit'])
        comparison = []
        for row in rows[:-1]:
            gradient = np.asarray(row['finite_difference_per_unit'])
            error = np.linalg.norm(gradient-fine,axis=1)/np.maximum(np.linalg.norm(fine,axis=1),1e-8)
            comparison.append(dict(dt_s=row['dt_s'],relative_gradient_difference_to_finest=error.tolist()))
        scale_error = float(np.linalg.norm(fine[0]-fine[-1])/max(np.linalg.norm(fine[-1]),1e-8))
        last = comparison[-1]['relative_gradient_difference_to_finest']
        case = dict(design_id=design_id,design=center.tolist(),measurements=rows,
            comparisons=comparison,finest_scale_relative_difference=scale_error,
            sensitivity_gate_passed=bool(all(r['all_probes_physically_valid'] for r in rows) and all(e<=.25 for e in last) and scale_error<=.25))
        if args.autodiff:
            dt,count = timesteps[-1],count_by_dt[timesteps[-1]]
            def objective(d):
                p = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
                x0 = solve_initial_pose(p).at[1].add(configuration.simulation.initial_clearance_um*1e-6)
                t,x = contact_rollout(x0,p,dt,count,mode=args.mode,smoothing=smoothing)
                centers = points(x,p,xp=jnp)['centers'][...,0]
                com = (centers*p[:4]).sum(-1)/p[:4].sum()
                return (com[-1]-com[count//2])/(t[-1]-t[count//2])
            print(f'Physical autodiff: design {design_id}, {args.mode} contact...',flush=True)
            derivative = np.asarray(jax.jit(jax.jacfwd(objective))(jnp.asarray(center)))*(np.asarray(DESIGN_UPPER)-DESIGN_LOWER)
            case['autodiff_per_unit'] = derivative.tolist() if np.isfinite(derivative).all() else None
            case['autodiff_matches_smallest_fd'] = bool(np.isfinite(derivative).all() and np.linalg.norm(derivative-fine[-1])/max(np.linalg.norm(fine[-1]),1e-8)<=.25)
            case['sensitivity_gate_passed'] &= case['autodiff_matches_smallest_fd']
        cases.append(case)
    report = dict(status='complete',cases=cases,horizon_s=settings['horizon_s'],unit_epsilons=epsilons,
        configuration=settings,mode=args.mode,timesteps_s=timesteps,dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        prior_report_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest() if prior_path.exists() else None,
        sensitivity_gate_passed=all(c['sensitivity_gate_passed'] for c in cases),
        scope=f'Local {args.mode}-contact finite differences; timestep and perturbation convergence required. No neural approximation.')
    write(args.output/'report.json',report)
    print(json.dumps(dict(sensitivity_gate_passed=report['sensitivity_gate_passed'])),flush=True)


if __name__=='__main__':
    main()
