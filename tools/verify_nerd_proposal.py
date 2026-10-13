"""Verify one local neural gradient step with original-contact simulations.

Requires all final surrogate gates, or all gates for an explicitly requested
validated horizon, to pass. Original global failures remain recorded. It never accepts
a final robot design or enables unattended optimization. Every original-law
integration is screened in full before downsampling for display.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
import torch
from generate_smooth_training_data import load
from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.learning import parameters_torch
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.learning_study import trajectory_observables,write_json
from walking_robot.trajectory_nerd import TrajectoryNeRD,predict_trajectory
from walking_robot.validation import inspect_trajectory
from walking_robot.surrogate_checks import validated_objective_horizons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/nerd_trajectory_kinematic.yaml'))
    parser.add_argument('--training',type=Path,default=Path('outputs/nerd_trajectory/consistent'))
    parser.add_argument('--nerd-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_trajectory/proposals'))
    parser.add_argument('--id',type=int)
    parser.add_argument('--unit-step',type=float,default=.003)
    parser.add_argument('--horizon-s',type=float,help='Explicitly restrict the check to an already-validated objective horizon.')
    args = parser.parse_args()
    s,c,designs,split = load(args.study)
    index = len(designs)*2//3 if args.id is None else args.id
    if index not in np.flatnonzero(split!=0):
        raise ValueError('Choose a held-out design ID.')
    if not np.isfinite(args.unit_step) or not 0<args.unit_step<=.003:
        raise ValueError('This local protocol permits a normalized step of at most 0.003.')
    report_path = args.training/'report.json'
    training = json.loads(report_path.read_text())
    if training['settings']!=s:
        raise ValueError('Training settings differ.')
    horizon = s['horizon_s'] if args.horizon_s is None else args.horizon_s
    if horizon not in [.2,1.,s['horizon_s']]:
        raise ValueError('Requested horizon was not evaluated.')
    scoped = horizon in validated_objective_horizons(training)
    if not scoped or (args.horizon_s is None and not training['surrogate_gates_passed']):
        raise ValueError('Required surrogate gates failed; no proposal is permitted. An explicit horizon must pass all of its original gates.')
    model_path = args.training/'model.pt'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=training['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch.')
    saved = torch.load(model_path,map_location='cpu',weights_only=False)
    if saved['configuration']!=asdict(c):
        raise ValueError('Physical configuration mismatch.')
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    if any(hashlib.sha256((source/n).read_bytes()).hexdigest()!=h for n,h in saved['provenance']['adapter_sources'].items()):
        raise ValueError('Neural adapter source mismatch.')
    if any(hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest()!=h
           for n,h in saved['provenance']['teacher']['sources'].items()):
        raise ValueError('Physical reference source mismatch.')
    output = args.output/f'{index:03d}'
    if (output/'report.json').exists():
        raise ValueError('Use a fresh output directory for repeated verification.')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    model = TrajectoryNeRD(args.nerd_root,c.design,horizon=s['horizon_s'],width=s['width'],layers=s['layers'],
        clearance_um=c.simulation.initial_clearance_um,consistent_velocity=s.get('consistent_velocity',False)).double().eval()
    if saved['upstream']['sha256']!=model.upstream['sha256']:
        raise ValueError('Upstream backbone differs.')
    model.load_state_dict(saved['state_dict'])
    span = np.array(DESIGN_UPPER)-DESIGN_LOWER
    center = (designs[index]-DESIGN_LOWER)/span
    def objective(u):
        times = u.new_tensor([horizon/2,horizon])
        states = predict_trajectory(model,u,times)
        p = parameters_torch(model.physical_designs(u),model.base_design)
        com,_,_ = trajectory_observables(states,p,xp=torch)
        return (com[:,-1,0]-com[:,0,0])/(horizon/2)
    u = torch.tensor(center[None],dtype=torch.float64,requires_grad=True)
    value = objective(u)
    gradient, = torch.autograd.grad(value.sum(),u)
    gradient = gradient.detach().numpy()[0]
    norm = np.linalg.norm(gradient)
    if not np.isfinite(norm) or norm<1e-8:
        raise ValueError('Invalid or negligible neural gradient.')
    proposal = center+args.unit_step*gradient/norm
    if not np.all((proposal>=0)&(proposal<=1)):
        raise ValueError('Proposal leaves the design domain.')
    with torch.no_grad():
        neural_gain = float(objective(torch.tensor(proposal[None],dtype=torch.float64))-value.detach())
    jax.config.update('jax_enable_x64',True)
    def prepare(unit):
        d = jnp.array(DESIGN_LOWER)+unit*jnp.array(span)
        p = c.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(c.simulation.initial_clearance_um*1e-6)
    get = jax.jit(prepare)
    pairs = [get(jnp.array(a)) for a in [center,proposal]]
    if not np.array_equal(np.array(pairs[0][0])[[22,26,41]],np.array(pairs[1][0])[[22,26,41]]):
        raise ValueError('Neural proposal crosses a discrete frequency/stiffness regime.')
    rows,gains = [],[]
    for dt in [2.5e-6,1.25e-6]:
        count = round(horizon/dt)
        run = jax.jit(lambda x,p:rollout_scan(x,p,dt,count))
        speeds = []
        for label,(p,x) in zip(['baseline','proposal'],pairs):
            print(f'NeRD design {index}: original-law {label}, T={horizon:g}, dt={dt:g}',flush=True)
            started = time.perf_counter()
            times,states = jax.tree.map(np.asarray,run(x,p))
            validation = inspect_trajectory(times,states,np.asarray(p),c.physical_limits)
            speeds.append(validation['metrics']['late_mean_com_speed_m_s'])
            rows.append(dict(label=label,timestep_s=dt,elapsed_s=time.perf_counter()-started,validation=validation))
            if dt==1.25e-6:
                stride = round(s['output_timestep_s']/dt)
                np.savez_compressed(output/(label+'.npz'),times=times[::stride],states=states[::stride],parameters=np.asarray(p))
        gains.append(float(speeds[1]-speeds[0]))
        write_json(output/'progress.json',dict(runs=rows,gains_m_s=gains))
    uncertainty = abs(gains[1]-gains[0])
    prediction = norm*args.unit_step
    forecast_error = abs(gains[1]-prediction)/max(abs(prediction),1e-10)
    passed = bool(all(r['validation']['passed'] for r in rows) and min(gains)>max(1e-6,2*uncertainty)
                  and forecast_error<=.25 and abs(neural_gain-prediction)/max(abs(prediction),1e-10)<=.25)
    write_json(output/'report.json',dict(id=index,design=designs[index].tolist(),
        horizon_s=horizon,global_surrogate_gates_passed=training['surrogate_gates_passed'],horizon_surrogate_gates_passed=scoped,
        proposed_design=(np.array(DESIGN_LOWER)+proposal*span).tolist(),unit_step=args.unit_step,
        neural_gradient_per_unit=gradient.tolist(),first_order_gain_m_s=prediction,neural_gain_m_s=neural_gain,
        physical_gains_m_s=gains,empirical_gain_uncertainty_m_s=uncertainty,forecast_relative_error=forecast_error,
        runs=rows,passed=passed,accepted_design=None,optimization_enabled=False,
        provenance=dict(training_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            checkpoint_sha256=training['checkpoint_sha256'],tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numerical_sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in ['dynamics','contact','_generated','geometry','parameters','initial_pose','integrators','validation']}),
        scope='One local canonical-start direction in a fixed rounded-parameter regime; no final design acceptance or robustness certification.'))
    print(f'Independent original-law neural proposal check: {passed}',flush=True)


if __name__=='__main__':
    main()
