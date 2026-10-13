"""Coarse/refined physical grid search with independent paired verification."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from walking_robot import solve_initial_pose,dynamics_step
from walking_robot.design import SimulationSettings
from walking_robot.physical_search import (SearchSettings,grid_designs,refinement_designs,
    unique_designs,step_count,paired_acceptance,write_json)
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/physical_search.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/physical_reference'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--skip-video',action='store_true')
    args = parser.parse_args()
    settings,configuration = SearchSettings.from_file(args.study)
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fingerprint = dict(settings=asdict(settings),configuration=asdict(configuration),jax_backend=jax.default_backend(),jax_version=jax.__version__,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')
                      if p.name in ['physical_search.py','parameters.py','contact.py','_generated.py','dynamics.py','integrators.py','initial_pose.py','geometry.py','validation.py','design.py']},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint = json.loads(json.dumps(fingerprint))
    checkpoint = args.output/'measurements.json'
    records = []
    if checkpoint.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume or a fresh directory.')
        old = json.loads(checkpoint.read_text())
        if old['fingerprint']!=fingerprint:
            raise ValueError('Cached source/settings mismatch; use a fresh directory.')
        records = old['measurements']
    print(f'JAX device: {jax.devices()}; cached measurements: {len(records)}',flush=True)
    base = configuration.design
    baseline = [base.adhesive_thickness_um,base.front_angle_deg]
    make_p = jax.jit(jax.vmap(lambda d:base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])))
    initial = jax.jit(jax.vmap(solve_initial_pose))
    compiled = {}
    def key(d,horizon,dt,condition_id):
        return json.dumps([list(d),horizon,dt,condition_id])
    cache = {r['key']:r for r in records}
    def measure(designs,horizon,dt,stage,*,condition_id=0,offset=None,phase=0.,save_paths=False):
        designs = np.asarray(designs)
        count = step_count(horizon,dt)
        needed = [d for d in designs if key(d,horizon,dt,condition_id) not in cache]
        if (horizon,dt) not in compiled:
            def single(x,p,start):
                def step(state,i):
                    nxt = dynamics_step(state,start+i*dt,p,dt=dt)
                    return nxt,nxt
                _,tail = jax.lax.scan(step,x,jnp.arange(count))
                return jnp.concatenate([x[None],tail])
            compiled[horizon,dt] = jax.jit(jax.vmap(single))
        times = np.arange(count+1)*dt
        for begin in range(0,len(needed),settings.batch_size):
            chunk = needed[begin:begin+settings.batch_size]
            padded = chunk+[chunk[-1]]*(settings.batch_size-len(chunk))
            p = make_p(jnp.asarray(padded))
            x = initial(p).at[:,1].add(configuration.simulation.initial_clearance_um*1e-6)
            if offset is not None:
                x = x+jnp.asarray(offset)
            started = time.perf_counter()
            path = np.asarray(compiled[horizon,dt](x,p,phase/p[:,22]))
            parameters = np.asarray(p)
            for d,xx,pp in zip(chunk,path,parameters):
                report = inspect_trajectory(times,xx,pp,configuration.physical_limits)
                record = dict(key=key(d,horizon,dt,condition_id),design=d.tolist(),horizon_s=horizon,dt_s=dt,
                    condition_id=condition_id,initial_offset=(np.zeros(14) if offset is None else offset).tolist(),
                    actuator_phase_cycles=phase,stage=stage,validation=report,
                    rounded_parameters=dict(frequency_hz=float(pp[22]),front_stiffness_n_m=float(pp[26]),hind_stiffness_n_m=float(pp[41])))
                if save_paths:
                    stride = max(1,round(1e-4/dt))
                    filename = f'path_{len(records):04d}.npz'
                    np.savez_compressed(args.output/filename,t=times[::stride],x=xx[::stride],P=pp)
                    record['display_trajectory'] = filename
                    record['display_stride'] = stride
                records.append(record)
                cache[record['key']] = record
            write_json(checkpoint,dict(fingerprint=fingerprint,measurements=records))
            print(f'{stage}: {min(begin+len(chunk),len(needed))}/{len(needed)} new designs, T={horizon:g}s dt={dt:g}, batch {time.perf_counter()-started:.1f}s',flush=True)
        return [cache[key(d,horizon,dt,condition_id)] for d in designs]
    def admissible_sorted(rows):
        return sorted([r for r in rows if r['validation']['passed'] and r['validation']['metrics'].get('late_mean_com_speed_m_s') is not None],
            key=lambda r:r['validation']['metrics']['late_mean_com_speed_m_s'],reverse=True)
    grid = grid_designs(settings,baseline)
    first = measure(grid,settings.search_horizon_s,settings.search_timestep_s,'coarse_grid')
    admissible = admissible_sorted(first)
    if not admissible:
        write_json(args.output/'report.json',dict(fingerprint=fingerprint,status='no_admissible_grid_design',accepted_design=None))
        return
    refined = refinement_designs(settings,admissible[0]['design'],grid)
    second = measure(refined,settings.search_horizon_s,settings.search_timestep_s,'local_grid') if len(refined) else []
    ranked = admissible_sorted(first+second)
    finalists = unique_designs([r['design'] for r in ranked if not np.allclose(r['design'],baseline,rtol=0,atol=1e-12)][:settings.finalist_count])
    verification_designs = unique_designs([baseline]+finalists.tolist())
    all_verified = []
    for horizon in settings.verification_horizons_s:
        for dt in settings.verification_timesteps_s:
            all_verified.extend(measure(verification_designs,horizon,dt,'canonical_verification',save_paths=True))
    def groups_for(candidate,conditions,horizons,timesteps):
        return [dict(horizon_s=horizon,condition_id=condition,required_timesteps_s=timesteps,
            runs=[dict(cache[key(d,horizon,dt,condition)],label=label) for dt in timesteps
                  for label,d in [('baseline',baseline),('candidate',candidate)]])
            for condition in conditions for horizon in horizons]
    candidate_checks = []
    for candidate in finalists:
        groups = groups_for(candidate.tolist(),[0],settings.verification_horizons_s,settings.verification_timesteps_s)
        candidate_checks.append(dict(design=candidate.tolist(),canonical=paired_acceptance(groups,settings)))
    eligible = [c for c in candidate_checks if c['canonical']['passed']]
    # Canonical selection is frozen before new initial conditions are tested.
    proposed = eligible[0]['design'] if eligible else None
    robustness,conditions = None,[]
    if proposed is not None:
        rng = np.random.default_rng(settings.seed)
        for variant in range(1,settings.robustness_variants+1):
            offset = np.zeros(14)
            offset[1] = rng.uniform(0,settings.height_perturbation_um)*1e-6
            offset[2:6] = np.deg2rad(rng.uniform(-settings.tilt_perturbation_deg,settings.tilt_perturbation_deg))
            offset[6:8] = rng.uniform(-settings.speed_perturbation_m_s,settings.speed_perturbation_m_s,2)
            phase = float(rng.uniform(0,1))
            conditions.append(dict(id=variant,offset=offset.tolist(),phase_cycles=phase))
            for dt in settings.robustness_timesteps_s:
                measure([baseline,proposed],settings.robustness_horizon_s,dt,'paired_robustness',condition_id=variant,offset=offset,phase=phase)
        robustness = paired_acceptance(groups_for(proposed,list(range(1,settings.robustness_variants+1)),
            [settings.robustness_horizon_s],settings.robustness_timesteps_s),settings)
    accepted = proposed if robustness is not None and robustness['passed'] else None
    report = dict(fingerprint=fingerprint,status='complete',objective='Late-half mean COM velocity, with all-step provisional physical screening.',
        frequency_policy='Design-derived frequency with original MATLAB ceiling; geometry and excitation frequency change together.',
        search_type='Derivative-free coarse grid plus one local refinement; no global optimum guarantee.',
        baseline=baseline,grid_design_count=len(first),refinement_design_count=len(second),
        physically_admissible_search_count=len(ranked),ranked_search=ranked,canonical_candidates=candidate_checks,
        proposed_design=proposed,robustness_conditions=conditions,robustness=robustness,accepted_design=accepted,
        gradient_optimization_ready=False,
        scope='Acceptance concerns these simulated horizons, sampled timesteps and initial conditions only; no hardware validation.')
    write_json(args.output/'report.json',report)
    plot_search(first+second,baseline,accepted,settings,args.output)
    display = accepted or proposed or ranked[0]['design']
    finest,horizon = min(settings.verification_timesteps_s),max(settings.verification_horizons_s)
    # The baseline's search interval may already be cached without a display
    # path. The longest, finest verification interval always has a saved path.
    for label,d in [('baseline',baseline),('candidate',display)]:
        record = cache[key(d,horizon,finest,0)]
        with np.load(args.output/record['display_trajectory'],allow_pickle=False) as path:
            t,x,p = path['t'],path['x'],path['P']
        if not args.skip_video:
            from walking_robot.visualization import render_trajectory
            render_trajectory(t,x,p,args.output/label,configuration.physical_limits,formats=('mp4',),
                max_frames=240,fps=30,playback_speed=.25,stop_at_violation=False)
        design = replace(base,adhesive_thickness_um=d[0],front_angle_deg=d[1])
        export = replace(configuration,design=design,simulation=SimulationSettings(horizon,finest,configuration.simulation.initial_clearance_um))
        (args.output/(label+'_design.yaml')).write_text(yaml.safe_dump(asdict(export),sort_keys=False))
    print(json.dumps(dict(proposed_design=proposed,accepted_design=accepted,gradient_optimization_ready=False)),flush=True)


def plot_search(rows,baseline,accepted,settings,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r['validation']['passed']]
    invalid = [r for r in rows if not r['validation']['passed']]
    fig,ax = plt.subplots(figsize=(9,5),constrained_layout=True)
    if valid:
        dots = ax.scatter([r['design'][0] for r in valid],[r['design'][1] for r in valid],
            c=[r['validation']['metrics']['late_mean_com_speed_m_s'] for r in valid],s=70,cmap='viridis')
        fig.colorbar(dots,ax=ax,label='Late-half COM speed (m/s)')
    if invalid:
        ax.scatter([r['design'][0] for r in invalid],[r['design'][1] for r in invalid],marker='x',color='red',label='Physical screening failed')
    ax.scatter(*baseline,marker='*',s=220,facecolor='white',edgecolor='black',label='Baseline')
    if accepted is not None:
        ax.scatter(*accepted,marker='s',s=180,facecolor='none',edgecolor='red',linewidth=2,label='Verified improvement')
    ax.set(xlabel='Adhesive thickness (micrometers)',ylabel='Front leg angle (degrees)',
        title=f'Physical grid search, {settings.search_horizon_s:g} s; original MATLAB contact')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output/'search.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
