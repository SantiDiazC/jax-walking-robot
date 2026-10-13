"""Run bounded neural design optimization with original-law acceptance.

Every accepted step is independently simulated at two resolutions. A final
design is exported only after canonical and perturbed-start paired checks.
Historical global neural-validation failures remain unchanged.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
os.environ.setdefault('MPLCONFIGDIR','/tmp/walking-robot-mpl')
import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from generate_smooth_training_data import load
from walking_robot import solve_initial_pose,dynamics_step
from walking_robot.design import DesignRun,SimulationSettings
from walking_robot.learning import parameters_torch
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.learning_study import trajectory_observables
from walking_robot.local_optimization import LocalSettings,verified_local_search,verify_pair
from walking_robot.physical_search import step_count,write_json
from walking_robot.surrogate_checks import validated_objective_horizons
from walking_robot.trajectory_nerd import TrajectoryNeRD,predict_trajectory
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/learning/nerd_local_optimization.yaml'))
    parser.add_argument('--training',type=Path,default=Path('outputs/nerd_trajectory/consistent'))
    parser.add_argument('--nerd-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_local_optimization'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--animate',action='store_true')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text())
    s,base,_,_ = load(args.config.resolve().parent/protocol['training_study'])
    initial = DesignRun.from_file(args.config.resolve().parent/protocol['initial_design'])
    local = LocalSettings(**{k:protocol[k] for k in LocalSettings.__dataclass_fields__})
    for h in protocol['verification_horizons_s']+[local.horizon_s]:
        for dt in local.timesteps_s:
            step_count(h,dt)
    if (not protocol['verification_horizons_s'] or local.horizon_s not in protocol['verification_horizons_s']
        or min(protocol['verification_horizons_s'])<=0):
        raise ValueError('Final verification must include the optimization horizon.')
    if type(protocol['robustness_variants']) is not int or protocol['robustness_variants']<1:
        raise ValueError('At least one independent perturbed-start check is required.')
    for name in ['height_perturbation_um','tilt_perturbation_deg','speed_perturbation_m_s']:
        if not np.isfinite(protocol[name]) or protocol[name]<0:
            raise ValueError(f'{name} must be finite and nonnegative.')
    if type(protocol['random_actuator_phase']) is not bool:
        raise ValueError('random_actuator_phase must be boolean.')
    fixed = replace(initial.design,adhesive_thickness_um=base.design.adhesive_thickness_um,front_angle_deg=base.design.front_angle_deg)
    if fixed!=base.design or initial.physical_limits!=base.physical_limits or initial.simulation.initial_clearance_um!=base.simulation.initial_clearance_um:
        raise ValueError('Fixed physics, physical limits and initial clearance must match training.')
    raw = (args.training/'report.json').read_bytes()
    training = json.loads(raw)
    if training['settings']!=s or local.horizon_s not in validated_objective_horizons(training):
        raise ValueError('The requested neural objective horizon has not passed its original gates.')
    model_file = args.training/'model.pt'
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if model_sha!=training['checkpoint_sha256']:
        raise ValueError('Checkpoint fingerprint mismatch.')
    saved = torch.load(model_file,map_location='cpu',weights_only=False)
    if saved['configuration']!=asdict(base):
        raise ValueError('Checkpoint physical configuration mismatch.')
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    for n,h in saved['provenance']['adapter_sources'].items():
        if hashlib.sha256((source/n).read_bytes()).hexdigest()!=h:
            raise ValueError('Neural adapter source mismatch.')
    for n,h in saved['provenance']['teacher']['sources'].items():
        if hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest()!=h:
            raise ValueError('Physical reference source mismatch.')
    torch.set_num_threads(2)
    model = TrajectoryNeRD(args.nerd_root,base.design,horizon=s['horizon_s'],width=s['width'],layers=s['layers'],
        clearance_um=base.simulation.initial_clearance_um,consistent_velocity=s.get('consistent_velocity',False)).double().eval()
    if model.upstream['sha256']!=saved['upstream']['sha256']:
        raise ValueError('Upstream NeRD source mismatch.')
    model.load_state_dict(saved['state_dict'])
    jax.config.update('jax_enable_x64',True)
    lower,span = np.array(DESIGN_LOWER),np.array(DESIGN_UPPER)-DESIGN_LOWER
    starting = (np.array([initial.design.adhesive_thickness_um,initial.design.front_angle_deg])-lower)/span
    def neural(unit):
        u = torch.tensor(unit[None],dtype=torch.float64,requires_grad=True)
        times = u.new_tensor([local.horizon_s/2,local.horizon_s])
        states = predict_trajectory(model,u,times)
        p = parameters_torch(model.physical_designs(u),model.base_design)
        c,_,_ = trajectory_observables(states,p,xp=torch)
        value = (c[:,-1,0]-c[:,0,0])/(local.horizon_s/2)
        gradient, = torch.autograd.grad(value.sum(),u)
        return float(value.detach()),gradient.detach().numpy()[0]
    def prepare(unit):
        d = jnp.array(lower)+unit*jnp.array(span)
        p = base.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(base.simulation.initial_clearance_um*1e-6)
    get = jax.jit(prepare)
    def same_regime(a,b):
        return np.array_equal(np.asarray(get(jnp.array(a))[0])[[22,26,41]],np.asarray(get(jnp.array(b))[0])[[22,26,41]])
    # Freeze the independent perturbations before any search evaluations.
    rng = np.random.default_rng(protocol['robustness_seed'])
    conditions = [dict(id=0,offset=[0.]*14,phase_cycles=0.)]
    for i in range(protocol['robustness_variants']):
        offset = np.zeros(14)
        offset[1] = rng.uniform(0,protocol['height_perturbation_um'])*1e-6
        offset[2:6] = np.deg2rad(rng.uniform(-protocol['tilt_perturbation_deg'],protocol['tilt_perturbation_deg']))
        offset[6:8] = rng.uniform(-protocol['speed_perturbation_m_s'],protocol['speed_perturbation_m_s'],2)
        phase = float(rng.uniform(0,1)) if protocol['random_actuator_phase'] else 0.
        conditions.append(dict(id=i+1,offset=offset.tolist(),phase_cycles=phase))
    fingerprint = json.loads(json.dumps(dict(protocol=protocol,initial_configuration=asdict(initial),training_configuration=asdict(base),
        training_report_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=model_sha,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jax_version=jax.__version__,backend=jax.default_backend(),
        sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in
            ['local_optimization','physical_search','dynamics','contact','_generated','geometry','parameters','initial_pose','integrators','validation']})))
    args.output.mkdir(parents=True,exist_ok=True)
    cache_file = args.output/'measurements.json'
    records = []
    if cache_file.exists():
        previous = json.loads(cache_file.read_text())
        if not args.resume or previous['fingerprint']!=fingerprint:
            raise ValueError('Existing measurements require --resume and identical sources/settings.')
        records = previous['measurements']
    cache = {r['key']:r for r in records}
    compiled = {}
    def evaluate(unit,horizon=local.horizon_s,condition_id=0):
        p,x = get(jnp.array(unit))
        condition = conditions[condition_id]
        x = x+jnp.array(condition['offset'])
        phase_time = condition['phase_cycles']/p[22]
        result = []
        for dt in local.timesteps_s:
            key = json.dumps([unit.tolist(),horizon,dt,condition_id])
            if key not in cache:
                count = step_count(horizon,dt)
                if (horizon,dt) not in compiled:
                    def run(x,p,start,count=count,dt=dt):
                        def step(state,i):
                            nxt = dynamics_step(state,start+i*dt,p,dt=dt)
                            return nxt,nxt
                        _,tail = jax.lax.scan(step,x,jnp.arange(count))
                        return jnp.concatenate([x[None],tail])
                    compiled[horizon,dt] = jax.jit(run)
                print(f'Original contact: design={(lower+unit*span).tolist()}, T={horizon:g}s, dt={dt:g}, condition={condition_id}',flush=True)
                started = time.perf_counter()
                states = np.asarray(compiled[horizon,dt](x,p,phase_time))
                times = np.arange(count+1)*dt
                validation = inspect_trajectory(times,states,np.asarray(p),initial.physical_limits)
                filename = f'trajectory_{len(records):03d}.npz'
                stride = max(1,round(s['output_timestep_s']/dt))
                np.savez_compressed(args.output/filename,times=times[::stride],states=states[::stride],parameters=np.asarray(p))
                row = dict(key=key,design=(lower+unit*span).tolist(),unit=unit.tolist(),horizon_s=horizon,dt_s=dt,
                    condition_id=condition_id,validation=validation,display_trajectory=filename,elapsed_s=time.perf_counter()-started)
                records.append(row);cache[key] = row
                write_json(cache_file,dict(fingerprint=fingerprint,conditions=conditions,measurements=records))
            result.append(cache[key])
        return result
    iterations = []
    def progress(row):
        iterations.append(row)
        write_json(args.output/'progress.json',dict(mode='solver_verified_local',iterations=iterations))
        print(f"Iteration {row['iteration']}, attempt {row['attempt']}: accepted={row['accepted']}, reason={row.get('reason','verified physical improvement')}",flush=True)
    search = verified_local_search(starting,neural,evaluate,same_regime,local,on_iteration=progress)
    final_unit = np.array(search['final_unit'])
    proposed = lower+final_unit*span
    canonical,robustness = [],[]
    if search['accepted_steps']:
        for h in protocol['verification_horizons_s']:
            a,b = evaluate(starting,h),evaluate(final_unit,h)
            canonical.append(verify_pair(a,b,local,horizon=h))
        for condition in conditions[1:]:
            a,b = evaluate(starting,local.horizon_s,condition['id']),evaluate(final_unit,local.horizon_s,condition['id'])
            robustness.append(verify_pair(a,b,local,condition_id=condition['id']))
    accepted = bool(search['accepted_steps'] and all(c['passed'] for c in canonical) and all(c['passed'] for c in robustness))
    report = dict(fingerprint=fingerprint,mode='solver_verified_local',optimization_enabled=True,
        surrogate_only_optimization_enabled=False,global_surrogate_validation_passed=training['surrogate_gates_passed'],
        full_neural_dynamics_validation_passed=False,objective_horizon_s=local.horizon_s,
        search=search,initial_design=(lower+starting*span).tolist(),proposed_design=proposed.tolist(),
        canonical_verification=canonical,robustness_conditions=conditions[1:],robustness_verification=robustness,
        accepted_design=proposed.tolist() if accepted else None,
        acceptance_scope='Original-law local simulated improvement at the configured horizons, timestep refinements and initial conditions; no global optimum, fabrication or hardware certification.')
    write_json(args.output/'report.json',report)
    if accepted:
        design = replace(initial.design,adhesive_thickness_um=float(proposed[0]),front_angle_deg=float(proposed[1]))
        export = replace(initial,design=design,simulation=SimulationSettings(local.horizon_s,min(local.timesteps_s),initial.simulation.initial_clearance_um))
        (args.output/'optimized_design.yaml').write_text(yaml.safe_dump(asdict(export),sort_keys=False))
    plot_search(search,args.output)
    if args.animate:
        from walking_robot.visualization import render_trajectory
        for label,u in [('baseline',starting),('candidate',final_unit)]:
            row = min(evaluate(u),key=lambda r:r['dt_s'])
            with np.load(args.output/row['display_trajectory'],allow_pickle=False) as path:
                render_trajectory(path['times'],path['states'],path['parameters'],args.output/label,initial.physical_limits,
                    formats=['mp4','html'],max_frames=150,stop_at_violation=False)
    print(json.dumps(dict(optimization_enabled=True,accepted_design=report['accepted_design'],accepted_steps=search['accepted_steps'])),flush=True)


def plot_search(search,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in search['history'] if r['accepted']]
    fig,ax = plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.plot(range(len(rows)+1),[search['initial_speed_m_s']]+[r['physical_speed_m_s'] for r in rows],marker='o')
    ax.set(xlabel='Accepted local step',ylabel='Original-contact COM speed (m/s)',title='NeRD proposals verified with the physical solver')
    ax.set_xticks(range(len(rows)+1));ax.grid(alpha=.25)
    fig.savefig(output/'optimization.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
