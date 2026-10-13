"""Run a YAML physical design grid, screen all cases and refine finalists."""
import argparse
from dataclasses import asdict,replace
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/sweeps/matlab_25_cases.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/matlab_25_sweep'))
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--render-best',action='store_true')
    parser.add_argument('--export-mat',action='store_true')
    args=parser.parse_args()
    os.environ['JAX_PLATFORMS']=args.backend
    os.environ.setdefault('MPLCONFIGDIR','/tmp/walking-robot-mpl')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from walking_robot import solve_initial_pose,rollout_batched
    from walking_robot.sweep import load_sweep,rank_cases,refinement_passed,export_mat
    from walking_robot.validation import inspect_trajectory
    from walking_robot.physical_search import write_json
    from walking_robot.design import VisualizationSettings
    jax.config.update('jax_enable_x64',True)
    started=time.perf_counter()
    s,base,designs=load_sweep(args.config)
    output=args.output
    source=Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fingerprint=json.loads(json.dumps(dict(settings=s,base_configuration=asdict(base),jax_version=jax.__version__,
        backend=jax.default_backend(),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in
            ['sweep','expanded_design','design','parameters','dynamics','contact','_generated','geometry','initial_pose','integrators','validation']})))
    if output.exists() and any(output.iterdir()) and not args.resume:raise ValueError('Use a fresh output directory or --resume with the exact same protocol.')
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprint:raise ValueError('Cached source/protocol/backend mismatch; use a fresh directory.')
    write_json(manifest,fingerprint)
    for folder in ['screen','verification']:(output/folder).mkdir(exist_ok=True)
    print(f'Simulating {len(designs)} designs on {jax.devices()}.',flush=True)
    parameters=jnp.stack([d.parameters() for d in designs])
    initial=jax.jit(jax.vmap(solve_initial_pose))(parameters).at[:,1].add(base.simulation.initial_clearance_um*1e-6)
    initial.block_until_ready()
    compiled={};batch_timings=[]
    batch=s['execution']['batch_size']
    def run(ids,dt,folder):
        path=output/folder
        pending=[i for i in ids if not (path/f'{i:03d}.json').exists()]
        for begin in range(0,len(pending),batch):
            chunk=pending[begin:begin+batch];padded=chunk+[chunk[-1]]*(batch-len(chunk))
            count=round(base.simulation.duration_s/dt)
            if dt not in compiled:
                compiled[dt]=jax.jit(lambda x,p,dt=dt,count=count:rollout_batched(x,p,dt=dt,steps=count))
            first=sum(r['dt_s']==dt for r in batch_timings)==0
            t0=time.perf_counter();t,xx=compiled[dt](initial[jnp.array(padded)],parameters[jnp.array(padded)])
            xx.block_until_ready();elapsed=time.perf_counter()-t0
            t,xx=np.asarray(t),np.asarray(xx);pp=np.asarray(parameters[jnp.array(padded)])
            batch_timings.append(dict(case_ids=chunk,dt_s=dt,compile_and_execute_s=elapsed,first_batch_at_this_resolution=first))
            for k,i in enumerate(chunk):
                validation=inspect_trajectory(t,xx[k],pp[k],base.physical_limits)
                row=dict(id=i,design=asdict(designs[i]),dt_s=dt,frequency_hz=float(pp[k,22]),validation=validation)
                stride=round(s['execution']['saved_timestep_s']/dt)
                np.savez_compressed(path/f'{i:03d}.npz',times=t[::stride],states=xx[k,::stride],parameters=pp[k])
                write_json(path/f'{i:03d}.json',row)
                print(f"{folder}: case {i}, physical={validation['passed']}, speed={validation['metrics'].get(s['objective'])}",flush=True)
    run(list(range(len(designs))),base.simulation.timestep_s,'screen')
    rows=[json.loads((output/'screen'/f'{i:03d}.json').read_text()) for i in range(len(designs))]
    ranked=rank_cases(rows,s['objective']);finalists=[r['id'] for r in ranked[:s['verification']['top_cases']]]
    refined=[]
    for dt in s['verification']['timesteps_s']:
        folder=f'verification/dt_{dt:.9g}';(output/folder).mkdir(exist_ok=True)
        run(finalists,dt,folder)
    for i in finalists:
        runs=[json.loads((output/'verification'/f'dt_{dt:.9g}'/f'{i:03d}.json').read_text()) for dt in s['verification']['timesteps_s']]
        refined.append(dict(id=i,runs=runs,passed=refinement_passed(runs,s['verification'],s['objective'])))
    eligible=[r for r in refined if r['passed']]
    eligible.sort(key=lambda r:-r['runs'][-1]['validation']['metrics'][s['objective']])
    chosen=eligible[0]['id'] if eligible else None
    finest=min(s['verification']['timesteps_s'])
    best=None
    if chosen is not None:
        best=replace(base,design=designs[chosen],simulation=replace(base.simulation,timestep_s=finest),
            visualization=VisualizationSettings(formats=['mp4','html'],max_frames=150))
        (output/'best_design.yaml').write_text(yaml.safe_dump(asdict(best),sort_keys=False))
        trajectory=output/'verification'/f'dt_{finest:.9g}'/f'{chosen:03d}.npz'
        if args.export_mat:export_mat(output/'best_case.mat',best,trajectory,integration_timestep_s=finest)
    csv_fields=['case_id']+list(s['parameters'])+['frequency_hz','physical_passed','objective_m_s','violations',
        'refinement_status','verified_objective_m_s','selected_best']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields);writer.writeheader()
        for r in rows:
            checked=next((q for q in refined if q['id']==r['id']),None)
            writer.writerow(dict(case_id=r['id'],**{k:r['design'][k] for k in s['parameters']},frequency_hz=r['frequency_hz'],
                physical_passed=r['validation']['passed'],objective_m_s=r['validation']['metrics'].get(s['objective']),
                violations='; '.join(r['validation']['violations']),
                refinement_status='not_checked' if checked is None else 'passed' if checked['passed'] else 'failed',
                verified_objective_m_s=None if checked is None else checked['runs'][-1]['validation']['metrics'].get(s['objective']),
                selected_best=r['id']==chosen))
    report=dict(fingerprint=fingerprint,hardware=[dict(device=str(d),kind=d.device_kind) for d in jax.devices()],
        total_cases=len(rows),screening_passed=len(ranked),screening_ranking_ids=[r['id'] for r in ranked],
        screening_cases=rows,refinement=refined,best_id=chosen,
        best_speed_m_s=None if chosen is None else eligible[0]['runs'][-1]['validation']['metrics'][s['objective']],
        objective=s['objective'],batch_timings=batch_timings,wall_time_s=time.perf_counter()-started,
        scope='Best physically passing, resolution-checked finalist. Other cases have screening resolution only. No global optimum, perturbed-start or hardware certification.',
        resumed=args.resume,states_saved_timestep_s=s['execution']['saved_timestep_s'])
    write_json(output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(s['parameters'])
    if len(names)==2:
        fig,axes=plt.subplots(1,2,figsize=(13,4));ax=axes[0]
        a,b=[s['parameters'][n] for n in names]
        speed=np.array([r['validation']['metrics'].get(s['objective']) if r['validation']['finite_samples']==r['validation']['total_samples'] else np.nan for r in rows],dtype=float).reshape(len(a),len(b))
        im=ax.imshow(np.ma.masked_invalid(speed),origin='lower',aspect='auto');fig.colorbar(im,ax=ax,label='Screening mean COM speed (m/s)')
        penetration=np.array([r['validation']['metrics'].get('max_foot_penetration_m') for r in rows],dtype=float).reshape(len(a),len(b))*1e3
        im=axes[1].imshow(np.ma.masked_invalid(penetration),origin='lower',aspect='auto');fig.colorbar(im,ax=axes[1],label='Maximum foot penetration (mm)')
        axes[0].set_title('Screening speed; × = physical limit violation')
        axes[1].set_title(f'Foot penetration; limit {base.physical_limits.max_foot_penetration_m*1e3:g} mm')
        for r in rows:
            if not r['validation']['passed']:axes[0].text(r['id']%len(b),r['id']//len(b),'×',ha='center',va='center',color='red',fontsize=18)
        for ax in axes:
            ax.set_xticks(range(len(b)),b);ax.set_yticks(range(len(a)),a);ax.set(xlabel=names[1],ylabel=names[0])
    else:
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar([str(r['id']) for r in ranked],[r['validation']['metrics'][s['objective']] for r in ranked]);ax.set(xlabel='Physically passing screening case',ylabel='Mean COM speed (m/s)')
    fig.tight_layout();fig.savefig(output/'sweep.png',dpi=150);plt.close(fig)
    if args.render_best and chosen is not None:
        from walking_robot.visualization import render_trajectory
        with np.load(trajectory,allow_pickle=False) as f:
            render_trajectory(f['times'],f['states'],f['parameters'],output/'best',base.physical_limits,
                formats=['mp4','html'],max_frames=150,stop_at_violation=False)
    print(json.dumps(dict(best_id=chosen,best_speed_m_s=report['best_speed_m_s'],wall_time_s=report['wall_time_s'])),flush=True)


if __name__=='__main__':main()
