"""Refine unresolved physical checks of gradient-generated proposals once.

Completed coarse measurements are immutable evidence with matching numerical
source hashes. Original failures remain in the original study report. This
checks local improvement directions, not publication/design acceptance.
"""
import argparse
from dataclasses import asdict,replace
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
from walking_robot.physical_search import paired_acceptance,step_count,write_json
from walking_robot.validation import inspect_trajectory
from validate_long_horizon_gradients import load_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('examples/learning/long_horizon_gradients.yaml'))
    parser.add_argument('--study-output',type=Path,default=Path('outputs/long_horizon_gradients'))
    parser.add_argument('--output',type=Path,default=Path('outputs/gradient_proposal_verification'))
    parser.add_argument('--timestep',type=float,default=1.25e-6)
    args = parser.parse_args()
    protocol,settings,config = load_protocol(args.protocol)
    jax.config.update('jax_enable_x64',True)
    original_path = args.study_output/'report.json'
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    if original['status']!='complete' or original['protocol']!=protocol:
        raise ValueError('Original study is incomplete or mismatched.')
    if original['provenance'].get('reference_settings')!=json.loads(json.dumps(asdict(settings))):
        raise ValueError('Original reference settings differ or predate complete reference-setting provenance.')
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    if original['provenance']['sources']!=hashes or original['provenance']['backend']!=jax.default_backend():
        raise ValueError('Original numerical sources or backend differ.')
    if not np.isfinite(args.timestep) or args.timestep<=0 or args.timestep>=min(protocol['hard_bridge_timesteps_s']):
        raise ValueError('Choose a positive timestep finer than the original verification.')
    if (args.output/'report.json').exists():
        raise ValueError('Use a fresh output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    lower = np.array([settings.thickness_bounds_um[0],settings.front_angle_bounds_deg[0]])
    span = np.array([np.diff(settings.thickness_bounds_um)[0],np.diff(settings.front_angle_bounds_deg)[0]])
    def prepare(u):
        # Reproduce the normalized-coordinate parameter mapping used in the
        # original study, including its floating-point evaluation order.
        a = lower+u*span
        p = config.design.parameters(adhesive_thickness_um=a[0],front_angle_deg=a[1])
        return p,solve_initial_pose(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
    get = jax.jit(prepare)
    local_settings = replace(settings,minimum_improvement_m_s=protocol['minimum_verified_gain_m_s'],
                             uncertainty_factor=protocol['uncertainty_factor'])
    checks = []
    for design in original['designs']:
        spec = next(d for d in protocol['designs'] if d['name']==design['name'])
        worker_path = args.study_output/design['name']/'measurements.json'
        worker_bytes = worker_path.read_bytes()
        data = json.loads(worker_bytes)['measurements']
        for case in design['cases']:
            bridge = case['hard_bridge']
            if bridge is None or bridge['passed']:
                continue
            if bridge['status']!='unconfirmed_at_tested_resolutions' or not case['gradient_gate_passed']:
                checks.append(dict(name=design['name'],horizon_s=case['horizon_s'],passed=False,
                                   reason='Original proposal lacks smooth validation or physical admissibility.'))
                continue
            horizon = case['horizon_s']
            n = step_count(horizon,args.timestep)
            run = jax.jit(lambda x,p:rollout_scan(x,p,args.timestep,n))
            pair = []
            for label,d in [('baseline',spec['values']),('candidate',case['direction_check']['design'])]:
                old = next(r for r in data if r['mode']=='hard' and r['design']==d and r['sample_timestep_s']==min(protocol['hard_bridge_timesteps_s']))
                validation = json.loads(json.dumps(old['solver']['accepted_step_physical_validation']))
                validation['metrics']['late_mean_com_speed_m_s'] = old['observations'][str(horizon)]['speed_m_s']
                pair.append(dict(label=label,dt_s=old['sample_timestep_s'],validation=validation,
                    physical_screening_horizon_s=old['physical_screening_horizon_s'],objective_horizon_s=horizon,
                    source='Original completed physical integration'))
                print(f"Refining {design['name']} {label}, T={horizon:g}s, dt={args.timestep:g}",flush=True)
                p,x = get(jnp.asarray((np.asarray(d)-lower)/span))
                t,xx = jax.tree.map(np.asarray,run(x,p))
                check = inspect_trajectory(t,xx,p,config.physical_limits)
                pair.append(dict(label=label,dt_s=args.timestep,validation=check,
                    physical_screening_horizon_s=horizon,objective_horizon_s=horizon,source='New finer-step physical integration'))
            outcome = paired_acceptance([dict(horizon_s=horizon,condition_id=0,
                required_timesteps_s=[min(protocol['hard_bridge_timesteps_s']),args.timestep],runs=pair)],local_settings)
            checks.append(dict(name=design['name'],horizon_s=horizon,original_bridge=bridge,passed=outcome['passed'],
                refined_check=outcome,runs=pair,worker_measurements_sha256=hashlib.sha256(worker_bytes).hexdigest()))
            write_json(args.output/'progress.json',dict(checks=checks))
    required = [(d['name'],c['horizon_s']) for d in original['designs'] for c in d['cases']
                if c['horizon_s'] in protocol['hard_bridge_horizons_s']]
    confirmed = []
    for name,h in required:
        case = next(c for d in original['designs'] if d['name']==name for c in d['cases'] if c['horizon_s']==h)
        refined = next((r for r in checks if r['name']==name and r['horizon_s']==h),None)
        confirmed.append(dict(name=name,horizon_s=h,passed=bool((case['hard_bridge'] and case['hard_bridge']['passed']) or (refined and refined['passed'])),
            source='refined' if refined else 'original'))
    report = dict(status='complete',original_report_sha256=hashlib.sha256(original_bytes).hexdigest(),timestep_s=args.timestep,
        checks=checks,combined_checks=confirmed,physical_direction_gate_passed=bool(confirmed) and all(c['passed'] for c in confirmed),
        neural_optimization_ready=False,accepted_design=None,
        scope='Local proposal directions only. Original unresolved checks are preserved; no neural, fabrication or hardware certification.')
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(physical_direction_gate_passed=report['physical_direction_gate_passed'])),flush=True)


if __name__=='__main__':
    main()
