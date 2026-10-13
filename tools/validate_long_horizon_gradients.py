"""Validate compiled adaptive sensitivities across designs and horizons.

Each integration extends to the longest horizon. Earlier objectives use exact
prefixes, with conservative full-horizon physical screening. Optional workers
use separate design directories; collection is read-only with respect to them.
No neural optimization or automatic design acceptance occurs in this study.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.adaptive_jax import JaxAdaptiveDynamics
from walking_robot.contact_residual import ContactSmoothing
from walking_robot.geometry import points
from walking_robot.physical_search import SearchSettings,inside_domain,step_count,write_json
from walking_robot.validation import inspect_trajectory


def load_protocol(path):
    protocol = yaml.safe_load(path.read_text())
    settings,config = SearchSettings.from_file(path.resolve().parent/protocol['study'])
    names = [d['name'] for d in protocol['designs']]
    if not names or len(set(names))!=len(names) or any(not re.fullmatch('[a-z0-9_]+',n) for n in names):
        raise ValueError('Design names must be unique lowercase identifiers.')
    if any(len(d['values'])!=2 or not inside_domain(d['values'],settings) for d in protocol['designs']):
        raise ValueError('Designs must lie inside the configured domain.')
    for key in ['rtols','unit_epsilons','hard_bridge_timesteps_s']:
        a = protocol[key]
        if len(a)<2 or not np.isfinite(a).all() or min(a)<=0 or not np.all(np.diff(a)<0):
            raise ValueError(f'{key} needs decreasing positive values.')
    hs = protocol['horizons_s']
    if not hs or not np.isfinite(hs).all() or min(hs)<=0 or not np.all(np.diff(hs)>0):
        raise ValueError('Horizons must be positive and increasing.')
    for name in ['direction_step','sample_timestep_s','refined_sample_timestep_s','gradient_relative_tolerance',
                 'direction_relative_tolerance','minimum_verified_gain_m_s','uncertainty_factor']:
        if not np.isfinite(protocol[name]) or protocol[name]<=0:
            raise ValueError(f'{name} must be positive.')
    if protocol['refined_sample_timestep_s']>=protocol['sample_timestep_s']:
        raise ValueError('The sampling refinement must be strictly finer.')
    if type(protocol['max_rhs_calls']) is not int or protocol['max_rhs_calls']<7:
        raise ValueError('Invalid integration budget.')
    for h in hs:
        for dt in [protocol['sample_timestep_s'],protocol['refined_sample_timestep_s']]+protocol['hard_bridge_timesteps_s']:
            step_count(h,dt)
    if not protocol['hard_bridge_horizons_s'] or any(h not in hs for h in protocol['hard_bridge_horizons_s']):
        raise ValueError('Bridge horizons must belong to the validation horizon list.')
    return protocol,settings,config


def com(x,p,xp=jnp):
    return (points(x,p,xp=xp)['centers'][...,0]*p[:4]).sum(-1)/p[:4].sum()


def observations(result,p,horizons,dp=None):
    dt = result['times'][1]
    values = {}
    for h in horizons:
        end = step_count(h,dt)
        half = end//2
        xx = result['states'][[half,end]]
        row = dict(speed_m_s=float((com(xx[-1],p,xp=np)-com(xx[0],p,xp=np))/(h/2)))
        if dp is not None:
            derivative = []
            for k in [half,end]:
                derivative.append(jax.vmap(lambda sx,sp:jax.jvp(com,(jnp.asarray(result['states'][k]),jnp.asarray(p)),(sx,sp))[1],in_axes=(1,1))(jnp.asarray(result['sensitivities'][k]),jnp.asarray(dp)))
            row['gradient_per_unit'] = np.asarray((derivative[-1]-derivative[0])/(h/2)).tolist()
        values[str(h)] = row
    return values


def run_design(protocol,settings,config,design_spec,output,resume):
    output.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fingerprint = dict(protocol=protocol,reference_settings=asdict(settings),configuration=asdict(config),design_spec=design_spec,
        backend=jax.default_backend(),jax_version=jax.__version__,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint = json.loads(json.dumps(fingerprint))
    path = output/'measurements.json'
    rows = []
    if path.exists():
        if not resume:
            raise ValueError('Output exists; use --resume or a new directory.')
        old = json.loads(path.read_text())
        if old['fingerprint']!=fingerprint:
            raise ValueError('Cached source/protocol mismatch; use a fresh directory.')
        rows = old['measurements']
    lower = np.array([settings.thickness_bounds_um[0],settings.front_angle_bounds_deg[0]])
    span = np.array([np.diff(settings.thickness_bounds_um)[0],np.diff(settings.front_angle_bounds_deg)[0]])
    center = np.asarray(design_spec['values'])
    def prepare(u):
        d = lower+u*span
        p = config.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
    get = jax.jit(prepare)
    get_derivative = jax.jit(jax.jacfwd(prepare))
    solver = JaxAdaptiveDynamics('smooth',ContactSmoothing(settings.gap_width_um*1e-6,settings.velocity_width_m_s,settings.force_width_uN*1e-6))
    horizons = protocol['horizons_s']
    max_horizon = max(horizons)
    physical = {}
    def measure(d,rtol,sample_dt,tangent=False,mode='smooth'):
        key = json.dumps([d.tolist(),rtol,sample_dt,tangent,mode])
        previous = next((r for r in rows if r['key']==key),None)
        if previous:
            return previous
        if not inside_domain(d,settings):
            raise ValueError('Requested design leaves the domain.')
        p,x = get(jnp.asarray((d-lower)/span))
        dp = None
        print(f"{design_spec['name']}: {mode}, T={max_horizon:g}, rtol={rtol}, sample={sample_dt:g}, tangent={tangent}, design={d.tolist()}",flush=True)
        started = time.perf_counter()
        if mode=='smooth':
            kwargs = {}
            if tangent:
                dp,dx = get_derivative(jnp.asarray((d-lower)/span))
                kwargs = dict(initial_sensitivity=dx,parameter_sensitivity=dp)
            info,result = solver.integrate(x,p,max_horizon,rtol=rtol,sample_dt=sample_dt,
                max_step=settings.comparison_max_step_s,max_rhs_calls=protocol['max_rhs_calls'],
                physical_limits=config.physical_limits,**kwargs)
        else:
            if sample_dt not in physical:
                physical[sample_dt] = jax.jit(lambda x,p:rollout_scan(x,p,sample_dt,step_count(max_horizon,sample_dt)))
            t,xx = jax.tree.map(np.asarray,physical[sample_dt](x,p))
            info = dict(success=bool(np.isfinite(xx).all()),method='hard_rk4',elapsed_s=time.perf_counter()-started,
                accepted_step_physical_validation=inspect_trajectory(t,xx,p,config.physical_limits))
            result = dict(times=t,states=xx) if info['success'] else None
        row = dict(key=key,design=d.tolist(),mode=mode,rtol=rtol,sample_timestep_s=sample_dt,tangent=tangent,solver=info,
                   physical_screening_horizon_s=max_horizon)
        if result is not None:
            row['observations'] = observations(result,np.asarray(p),horizons,None if dp is None else np.asarray(dp))
            row['physical_passed'] = info['accepted_step_physical_validation']['passed']
            if tangent:
                stride = max(1,round(1e-4/sample_dt))
                filename = f'trajectory_{len(rows):03d}.npz'
                np.savez_compressed(output/filename,t=result['times'][::stride],x=result['states'][::stride],P=np.asarray(p))
                row['display_trajectory'] = filename
        rows.append(row)
        write_json(path,dict(fingerprint=fingerprint,measurements=rows))
        print(f"Completed {design_spec['name']}: success={info['success']}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
        return row
    probes = [center]
    for eps in protocol['unit_epsilons']:
        for axis in range(2):
            for sign in [-1,1]:
                delta = np.zeros(2); delta[axis] = sign*eps*span[axis]
                probes.append(center+delta)
    p_all = np.asarray(jax.vmap(lambda d:prepare((d-lower)/span)[0])(jnp.asarray(probes)))
    if not np.all(p_all[:,[22,26,41]]==p_all[0,[22,26,41]]):
        raise ValueError('Local probes cross a discrete parameter regime.')
    rounded = p_all[0,[22,26,41]]
    coarse_dt,fine_dt = protocol['sample_timestep_s'],protocol['refined_sample_timestep_s']
    tangents,primal = {},{}
    for rtol in protocol['rtols']:
        tangents[rtol] = measure(center,rtol,coarse_dt,True)
        primal[rtol] = [measure(d,rtol,coarse_dt) for d in probes]
    finest = protocol['rtols'][-1]
    refined = measure(center,finest,fine_dt,True)
    cases = []
    for horizon in horizons:
        label = str(horizon)
        checks = []
        for rtol in protocol['rtols']:
            tangent = tangents[rtol]
            actual = primal[rtol]
            complete = 'observations' in tangent and all('observations' in r for r in actual)
            check = dict(rtol=rtol,complete=complete,passed=False)
            if complete:
                grad = np.asarray(tangent['observations'][label]['gradient_per_unit'])
                values = np.array([r['observations'][label]['speed_m_s'] for r in actual])
                fd = np.array([[(values[1+4*k+2*a+1]-values[1+4*k+2*a])/(2*eps) for a in range(2)] for k,eps in enumerate(protocol['unit_epsilons'])])
                error = np.linalg.norm(fd-grad,axis=1)/np.maximum(np.linalg.norm(fd,axis=1),1e-8)
                check.update(gradient_per_unit=grad.tolist(),finite_difference_per_unit=fd.tolist(),relative_errors=error.tolist(),
                    objective_m_s=float(values[0]),passed=bool(tangent['physical_passed'] and all(r['physical_passed'] for r in actual) and np.all(error<=protocol['gradient_relative_tolerance'])))
            checks.append(check)
        case = dict(horizon_s=horizon,tolerance_checks=checks,gradient_gate_passed=False,direction_check=None,hard_bridge=None)
        if all(c['passed'] for c in checks) and refined.get('physical_passed'):
            g0,g1 = [np.asarray(c['gradient_per_unit']) for c in checks[-2:]]
            gf = np.asarray(refined['observations'][label]['gradient_per_unit'])
            tolerance_error = float(np.linalg.norm(g0-g1)/max(np.linalg.norm(g1),1e-8))
            sample_error = float(np.linalg.norm(g1-gf)/max(np.linalg.norm(gf),1e-8))
            case.update(tolerance_relative_difference=tolerance_error,sampling_relative_difference=sample_error,
                refined_gradient_per_unit=gf.tolist())
            if max(tolerance_error,sample_error)<=protocol['gradient_relative_tolerance']:
                delta = protocol['direction_step']*gf/max(np.linalg.norm(gf),1e-8)
                moved = center+delta*span
                moved_p,_ = get(jnp.asarray((moved-lower)/span))
                regime = np.array_equal(np.asarray(moved_p)[[22,26,41]],rounded)
                if inside_domain(moved,settings) and regime:
                    changed = [measure(moved,r,coarse_dt) for r in protocol['rtols']]
                    if all(r.get('physical_passed') for r in changed):
                        gains = [r['observations'][label]['speed_m_s']-primal[t][0]['observations'][label]['speed_m_s'] for r,t in zip(changed,protocol['rtols'])]
                        uncertainty = sum(abs(a['observations'][label]['speed_m_s']-b['observations'][label]['speed_m_s']) for a,b in [(primal[protocol['rtols'][-2]][0],primal[finest][0]),(changed[-2],changed[-1])])
                        prediction = float(gf@delta)
                        discrepancy = abs(gains[-1]-prediction)/max(abs(prediction),1e-8)
                        threshold = max(protocol['minimum_verified_gain_m_s'],protocol['uncertainty_factor']*uncertainty)
                        passed = bool(all(v>0 for v in gains) and gains[-1]>threshold and discrepancy<=protocol['direction_relative_tolerance'])
                        case['direction_check'] = dict(design=moved.tolist(),predicted_gain_m_s=prediction,actual_gain_m_s=gains[-1],
                            estimated_uncertainty_m_s=uncertainty,required_gain_m_s=threshold,relative_prediction_error=discrepancy,passed=passed)
                        case['gradient_gate_passed'] = passed
                        if horizon in protocol['hard_bridge_horizons_s']:
                            pairs = []
                            for dt in protocol['hard_bridge_timesteps_s']:
                                a,b = measure(center,None,dt,mode='hard'),measure(moved,None,dt,mode='hard')
                                pairs.append(dict(dt_s=dt,baseline=a,candidate=b))
                            physical_pass = all(r.get('physical_passed') for pair in pairs for r in [pair['baseline'],pair['candidate']])
                            if physical_pass:
                                gains = [q['candidate']['observations'][label]['speed_m_s']-q['baseline']['observations'][label]['speed_m_s'] for q in pairs]
                                uncertainty = sum(abs(pairs[-1][side]['observations'][label]['speed_m_s']-pairs[-2][side]['observations'][label]['speed_m_s']) for side in ['baseline','candidate'])
                                threshold = max(protocol['minimum_verified_gain_m_s'],protocol['uncertainty_factor']*uncertainty)
                                case['hard_bridge'] = dict(gains_m_s=gains,estimated_uncertainty_m_s=uncertainty,required_gain_m_s=threshold,
                                    passed=bool(all(v>0 for v in gains) and gains[-1]>threshold),
                                    status='confirmed' if all(v>0 for v in gains) and gains[-1]>threshold else 'unconfirmed_at_tested_resolutions')
                            else:
                                case['hard_bridge'] = dict(passed=False,status='integration_or_physical_screening_failure')
                else:
                    case['direction_check'] = dict(passed=False,reason='Direction leaves the domain or discrete parameter regime.')
        cases.append(case)
        write_json(output/'report.json',dict(status='in_progress',fingerprint=fingerprint,cases=cases))
    report = dict(status='complete',fingerprint=fingerprint,cases=cases,
        smooth_gradient_gate_passed=all(c['gradient_gate_passed'] for c in cases),
        hard_bridge_gate_passed=all(c['hard_bridge'] is not None and c['hard_bridge']['passed'] for c in cases if c['horizon_s'] in protocol['hard_bridge_horizons_s']),
        neural_optimization_ready=False,accepted_design=None,
        scope='Local gradients within fixed rounded-parameter regimes. Full longest-horizon screening is used for every prefix. No neural or hardware certification.')
    write_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['smooth_gradient_gate_passed','hard_bridge_gate_passed','neural_optimization_ready']}),flush=True)


def collect(protocol,settings,config,output):
    reports = []
    missing = []
    provenance = None
    for d in protocol['designs']:
        path = output/d['name']/'report.json'
        if not path.exists():
            missing.append(d['name']); continue
        r = json.loads(path.read_text())
        if r['status']!='complete' or r['fingerprint']['protocol']!=protocol:
            raise ValueError('Worker report is incomplete or has a different protocol.')
        if r['fingerprint'].get('reference_settings')!=json.loads(json.dumps(asdict(settings))) or r['fingerprint']['configuration']!=json.loads(json.dumps(asdict(config))):
            raise ValueError('Worker reference settings differ or predate complete reference-setting provenance.')
        shared = {k:v for k,v in r['fingerprint'].items() if k not in ['protocol','design_spec']}
        if provenance is not None and shared!=provenance:
            raise ValueError('Worker reports use different source versions, backends or physical configurations.')
        provenance = shared
        reports.append(dict(name=d['name'],cases=r['cases'],smooth_gradient_gate_passed=r['smooth_gradient_gate_passed'],
            hard_bridge_gate_passed=r['hard_bridge_gate_passed'],report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(output/'report.json',dict(status='complete' if not missing else 'incomplete',protocol=protocol,provenance=provenance,designs=reports,missing_designs=missing,
        smooth_gradient_gate_passed=not missing and all(r['smooth_gradient_gate_passed'] for r in reports),
        hard_bridge_gate_passed=not missing and all(r['hard_bridge_gate_passed'] for r in reports),neural_optimization_ready=False,accepted_design=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('examples/learning/long_horizon_gradients.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/long_horizon_gradients'))
    parser.add_argument('--design',help='Run one named design in its own output subdirectory.')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--collect-only',action='store_true')
    args = parser.parse_args()
    protocol,settings,config = load_protocol(args.protocol)
    jax.config.update('jax_enable_x64',True)
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.collect_only:
        selected = [d for d in protocol['designs'] if args.design is None or d['name']==args.design]
        if not selected:
            parser.error('Unknown design name.')
        for d in selected:
            run_design(protocol,settings,config,d,args.output/d['name'],args.resume)
    if args.design is None or args.collect_only:
        collect(protocol,settings,config,args.output)


if __name__=='__main__':
    main()
