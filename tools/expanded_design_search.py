"""Explore three design variables with fixed voltage/frequency and original contact.

Workers write independent case files. Candidate selection is frozen before
independent initial-condition tests. Failed numerical/physical cases remain
explicit. All accepted-step screening uses the full physical trajectory.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from walking_robot import solve_initial_pose,dynamics_step
from walking_robot.design import DesignRun,SimulationSettings,VisualizationSettings
from walking_robot.expanded_design import ExpandedRobotDesign,ExpandedDesignRun,sample_expanded_designs
from walking_robot.physical_search import step_count,paired_acceptance,write_json
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/learning/expanded_design_search.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/expanded_design_search'))
    parser.add_argument('--stage',choices=['screen','collect-screen','verify','collect-verify','robust','collect-final'],default='screen')
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--render',action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    protocol = yaml.safe_load(args.config.read_text())
    base = DesignRun.from_file(args.config.resolve().parent/protocol['base_design'])
    extended = ExpandedRobotDesign(**dict(asdict(base.design),voltage_v=protocol['voltage_v']),drive_frequency_hz=protocol['drive_frequency_hz'])
    if not 0<=args.worker<args.workers:
        raise ValueError('Invalid worker partition.')
    designs = sample_expanded_designs(protocol,extended)
    dts = protocol['verification_timesteps_s']
    if len(dts)<2 or min(dts)<=0 or not np.all(np.diff(dts)<0):
        raise ValueError('Verification timesteps must be positive and decreasing.')
    for h,dt in [(protocol['screen_horizon_s'],protocol['screen_timestep_s'])]+[(protocol['verification_horizon_s'],dt) for dt in dts]:
        step_count(h,dt)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fingerprint = json.loads(json.dumps(dict(protocol=protocol,base_configuration=asdict(base),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jax_version=jax.__version__,backend=jax.default_backend(),
        sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in
            ['expanded_design','parameters','dynamics','contact','_generated','geometry','initial_pose','integrators','validation','physical_search']})))
    args.output.mkdir(parents=True,exist_ok=True)
    settings = SimpleNamespace(**protocol)
    def parameters(d):
        return extended.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1],hind_angle_deg=d[2])
    get = jax.jit(parameters)
    initial = jax.jit(solve_initial_pose)
    compiled = {}
    conditions = [dict(id=0,offset=[0.]*14,phase_cycles=0.)]
    rng = np.random.default_rng(protocol['robustness_seed'])
    for i in range(protocol['robustness_variants']):
        offset = np.zeros(14)
        offset[1] = rng.uniform(0,protocol['height_perturbation_um'])*1e-6
        offset[2:6] = np.deg2rad(rng.uniform(-protocol['tilt_perturbation_deg'],protocol['tilt_perturbation_deg']))
        offset[6:8] = rng.uniform(-protocol['speed_perturbation_m_s'],protocol['speed_perturbation_m_s'],2)
        conditions.append(dict(id=i+1,offset=offset.tolist(),phase_cycles=float(rng.uniform(0,1))))
    def read(path):
        value = json.loads(path.read_text())
        if value['fingerprint']!=fingerprint:
            raise ValueError('Source/protocol mismatch; use a fresh directory.')
        return value
    def simulate(design_id,horizon,dt,condition=0):
        p = get(jnp.array(designs[design_id]));x = initial(p)
        x = x.at[1].add(base.simulation.initial_clearance_um*1e-6)+jnp.array(conditions[condition]['offset'])
        key = (horizon,dt)
        if key not in compiled:
            count = step_count(horizon,dt)
            def run(x,p,start):
                def step(state,i):
                    nxt = dynamics_step(state,start+i*dt,p,dt=dt)
                    return nxt,nxt
                _,tail = jax.lax.scan(step,x,jnp.arange(count))
                return jnp.concatenate([x[None],tail])
            compiled[key] = jax.jit(run)
        print(f'Fixed actuation: design {design_id}, T={horizon:g}, dt={dt:g}, condition={condition}',flush=True)
        times = np.arange(step_count(horizon,dt)+1)*dt
        states = np.asarray(compiled[key](x,p,conditions[condition]['phase_cycles']/p[22]))
        validation = inspect_trajectory(times,states,np.asarray(p),base.physical_limits)
        return dict(id=int(design_id),design=designs[design_id].tolist(),horizon_s=horizon,dt_s=dt,
                    condition_id=condition,validation=validation),times,states,np.asarray(p)
    def verified(design_id,condition=0):
        folder = args.output/('verification' if condition==0 else f'robustness_{condition}')
        folder.mkdir(exist_ok=True)
        path = folder/f'{design_id:03d}.json'
        if path.exists():
            return read(path)['runs']
        runs = []
        for k,dt in enumerate(dts):
            row,t,x,p = simulate(design_id,protocol['verification_horizon_s'],dt,condition)
            if k==len(dts)-1:
                stride = max(1,round(1e-4/dt))
                np.savez_compressed(folder/f'{design_id:03d}.npz',times=t[::stride],states=x[::stride],parameters=p)
            runs.append(row)
        write_json(path,dict(fingerprint=fingerprint,runs=runs))
        return runs
    def compare(a,b,condition=0):
        return paired_acceptance([dict(horizon_s=protocol['verification_horizon_s'],condition_id=condition,
            required_timesteps_s=dts,runs=[dict(r,label=label) for label,rows in [('baseline',a),('candidate',b)] for r in rows])],settings)
    if args.stage=='screen':
        folder = args.output/'screen';folder.mkdir(exist_ok=True)
        ids = [i for i in range(args.worker,len(designs),args.workers) if not (folder/f'{i:03d}.json').exists()]
        for i in range(args.worker,len(designs),args.workers):
            if (folder/f'{i:03d}.json').exists():read(folder/f'{i:03d}.json')
        horizon,dt = protocol['screen_horizon_s'],protocol['screen_timestep_s']
        count = step_count(horizon,dt);batch = protocol['batch_size']
        run = jax.jit(jax.vmap(lambda d:single(d)))
        def single(d):
            p = parameters(d);x = solve_initial_pose(p).at[1].add(base.simulation.initial_clearance_um*1e-6)
            def step(state,i):
                nxt = dynamics_step(state,i*dt,p,dt=dt)
                return nxt,nxt
            _,tail = jax.lax.scan(step,x,jnp.arange(count))
            return jnp.concatenate([x[None],tail]),p
        times = np.arange(count+1)*dt
        for begin in range(0,len(ids),batch):
            chunk = ids[begin:begin+batch]
            padded = chunk+[chunk[-1]]*(batch-len(chunk))
            xx,pp = jax.tree.map(np.asarray,run(jnp.array(designs[padded])))
            for k,index in enumerate(chunk):
                row = dict(fingerprint=fingerprint,id=index,design=designs[index].tolist(),
                           validation=inspect_trajectory(times,xx[k],pp[k],base.physical_limits))
                write_json(folder/f'{index:03d}.json',row)
                print(f"Screen {index}: {row['validation']['passed']}, speed={row['validation']['metrics']['late_mean_com_speed_m_s']}",flush=True)
        return
    if args.stage=='collect-screen':
        rows = [read(args.output/'screen'/f'{i:03d}.json') for i in range(len(designs))]
        ranked = sorted([r for r in rows if r['validation']['passed']],key=lambda r:r['validation']['metrics']['late_mean_com_speed_m_s'],reverse=True)
        ids = [r['id'] for r in ranked if r['id']!=0][:protocol['finalist_count']]
        write_json(args.output/'selection.json',dict(fingerprint=fingerprint,finalist_ids=ids,ranked=ranked,
            total_count=len(rows),physically_passed=len(ranked)))
        print('Finalists',[(r['id'],r['validation']['metrics']['late_mean_com_speed_m_s']) for r in ranked[:protocol['finalist_count']]],flush=True)
        return
    selection = read(args.output/'selection.json')
    ids = [0]+selection['finalist_ids']
    if args.stage=='verify':
        for i in ids[args.worker::args.workers]:verified(i)
        return
    if args.stage=='collect-verify':
        a = verified(0)
        candidates = [dict(id=i,design=designs[i].tolist(),check=compare(a,verified(i)),
            speed_m_s=verified(i)[-1]['validation']['metrics']['late_mean_com_speed_m_s']) for i in ids if i!=0]
        eligible = sorted([r for r in candidates if r['check']['passed']],key=lambda r:r['speed_m_s'],reverse=True)
        proposed = eligible[0]['id'] if eligible else None
        write_json(args.output/'proposal.json',dict(fingerprint=fingerprint,candidates=candidates,proposed_id=proposed))
        print('Frozen proposal:',proposed,eligible[:1],flush=True)
        return
    proposal = read(args.output/'proposal.json');chosen = proposal['proposed_id']
    if args.stage=='robust':
        if chosen is not None:
            for condition in range(1+args.worker,len(conditions),args.workers):
                verified(0,condition);verified(chosen,condition)
        return
    robustness = [] if chosen is None else [compare(verified(0,i),verified(chosen,i),i) for i in range(1,len(conditions))]
    accepted = chosen if chosen is not None and robustness and all(r['passed'] for r in robustness) else None
    report = dict(fingerprint=fingerprint,scope='Expanded design, fixed 200 V / 363 Hz; simulated cases only, no hardware or global-optimum certification.',
        screen=dict(total=selection['total_count'],physically_passed=selection['physically_passed']),
        canonical_candidates=proposal['candidates'],proposed_id=chosen,accepted_id=accepted,
        proposed_design=None if chosen is None else designs[chosen].tolist(),accepted_design=None if accepted is None else designs[accepted].tolist(),
        robustness_conditions=conditions[1:],robustness=robustness,actuation=dict(voltage_v=extended.voltage_v,drive_frequency_hz=extended.drive_frequency_hz))
    if chosen is not None:
        candidate = next(c for c in proposal['candidates'] if c['id']==chosen)
        report['measured_speed_m_s'] = candidate['speed_m_s']
        report['targets'] = [dict(target_m_s=t,met=bool(accepted is not None and candidate['speed_m_s']>=t)) for t in protocol['targets_m_s']]
        export = ExpandedDesignRun(design=replace(extended,adhesive_thickness_um=float(designs[chosen,0]),front_angle_deg=float(designs[chosen,1]),hind_angle_deg=float(designs[chosen,2])),
            simulation=SimulationSettings(protocol['verification_horizon_s'],min(dts),base.simulation.initial_clearance_um),
            physical_limits=base.physical_limits,visualization=VisualizationSettings(formats=['mp4','html'],max_frames=150))
        (args.output/('optimized_design.yaml' if accepted is not None else 'proposed_design.yaml')).write_text(yaml.safe_dump(asdict(export),sort_keys=False))
        if args.render:
            from walking_robot.visualization import render_trajectory
            for label,i in [('baseline',0),('candidate',chosen)]:
                with np.load(args.output/'verification'/f'{i:03d}.npz') as p:
                    render_trajectory(p['times'],p['states'],p['parameters'],args.output/label,base.physical_limits,
                        formats=['mp4','html'],max_frames=150,stop_at_violation=False)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(accepted_design=report['accepted_design'],speed_m_s=report.get('measured_speed_m_s'))),flush=True)


if __name__=='__main__':main()
