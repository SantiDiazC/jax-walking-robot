"""Train a bounded JAX contact correction while retaining analytical mechanics.

This is a physics-structured comparator, not the official NeRD Transformer.
The explicitly selected, previously screened dataset is immutable reference
evidence; its checksum and original provenance are retained in every run.
"""
import argparse
from dataclasses import asdict
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

from walking_robot.contact_residual import (ContactSmoothing,contact_rollout,
    initialize_contact_network,contact_multipliers)
from walking_robot.design import DesignRun
from walking_robot.geometry import points,point_velocities
from walking_robot.integrators import dynamics_step
from walking_robot.learning_spec import STATE_SCALES
from walking_robot.validation import inspect_trajectory
from contact_sensitivity_study import write


def observables(states,parameters,xp=np):
    p = parameters[:,None,:]
    weight = p[...,:4]/p[...,:4].sum(-1,keepdims=True)
    g,v = points(states,p,xp=xp),point_velocities(states,p,xp=xp)
    return ((g['centers']*weight[...,None]).sum(-2),
            (v['centers']*weight[...,None]).sum(-2),g['feet'][...,1])


def ranking(reference,prediction,tolerance):
    pairs = [(i,j) for i in range(len(reference)) for j in range(i)
             if np.isfinite(reference[i]-reference[j]) and abs(reference[i]-reference[j])>tolerance]
    correct = sum(bool(np.isfinite(prediction[[i,j]]).all() and
                       (reference[i]-reference[j])*(prediction[i]-prediction[j])>0) for i,j in pairs)
    return dict(eligible_pairs=len(pairs),correct_pairs=correct,agreement=correct/len(pairs) if pairs else None)


def metrics(data,ids,full_prediction,settings,configuration,substeps):
    prediction = full_prediction[:,::substeps]
    ref = data['states'][ids]
    p = data['parameters'][ids]
    pc,_,pf = observables(prediction,p)
    tc,_,tf = observables(ref,p)
    half = (len(data['times'])-1)//2
    duration = data['times'][-1]-data['times'][half]
    ps,ts = (pc[:,-1,0]-pc[:,half,0])/duration,(tc[:,-1,0]-tc[:,half,0])/duration
    t = np.arange(full_prediction.shape[1])*(settings['neural_timestep_s']/substeps)
    cases = []
    for k,i in enumerate(ids):
        finite = np.isfinite(full_prediction[k]).all()
        cases.append(dict(id=int(i),design_id=int(data['design_id'][i]),variant=int(data['variant'][i]),
            reference_passed=bool(data['passed'][i]),
            prediction_validation=inspect_trajectory(t,full_prediction[k],p[k],configuration.physical_limits),
            state_rmse=float(np.sqrt(np.mean(((prediction[k]-ref[k])/STATE_SCALES)**2))) if finite else float('inf'),
            speed_error_m_s=float(abs(ps[k]-ts[k])) if finite else float('inf'),
            reference_speed_m_s=float(ts[k]),predicted_speed_m_s=float(ps[k]),
            foot_height_rmse_m=float(np.sqrt(np.mean((pf[k]-tf[k])**2))) if finite else float('inf')))
    valid = [c for c in cases if c['reference_passed']]
    canonical = [c for c in valid if c['variant']==0]
    order = ranking(np.array([c['reference_speed_m_s'] for c in canonical]),
                    np.array([c['predicted_speed_m_s'] for c in canonical]),settings['ranking_tolerance_m_s'])
    false_accept = [c['id'] for c in cases if not c['reference_passed'] and c['prediction_validation']['passed']]
    score = np.mean([c['state_rmse']+c['speed_error_m_s']/settings['max_speed_error_m_s']+
                     c['foot_height_rmse_m']/settings['max_foot_height_rmse_m'] for c in valid]) if valid else float('inf')
    fidelity = bool(valid) and all(c['prediction_validation']['passed'] and c['state_rmse']<=.1 and
        c['speed_error_m_s']<=settings['max_speed_error_m_s'] and c['foot_height_rmse_m']<=settings['max_foot_height_rmse_m']
        for c in valid) and not false_accept
    passed = (len(canonical)>=3 and fidelity and order['eligible_pairs']>=3 and
        order['agreement'] is not None and order['agreement']>=settings['min_ranking_agreement'])
    return dict(cases=cases,selection_score=float(score),fidelity_gate_passed=bool(fidelity),trajectory_gate_passed=bool(passed),
        canonical_ranking=order,false_accept_ids=false_accept,
        mean_speed_error_m_s=float(np.mean([c['speed_error_m_s'] for c in valid])),
        mean_foot_height_rmse_m=float(np.mean([c['foot_height_rmse_m'] for c in valid])),
        admissible_reference_count=len(valid),
        physically_passing_prediction_count=sum(c['prediction_validation']['passed'] for c in valid))


def load_reference(settings,configuration,root):
    path = root/settings['reference_dataset']
    with np.load(path,allow_pickle=False) as saved:
        data = {k:saved[k] for k in saved.files}
    provenance = json.loads(path.with_name('dataset_report.json').read_text())
    recorded = provenance['fingerprint']['configuration']
    if recorded['design']!=asdict(configuration.design) or recorded['simulation']['initial_clearance_um']!=configuration.simulation.initial_clearance_um:
        raise ValueError('Reference fixed design inputs or initial clearance differ from YAML.')
    if recorded['physical_limits']!=asdict(configuration.physical_limits):
        raise ValueError('Reference screening limits differ from YAML.')
    if not np.allclose(np.diff(data['times']),settings['neural_timestep_s'],rtol=0,atol=1e-12) or not np.isclose(data['times'][-1],settings['horizon_s'],rtol=0,atol=1e-12):
        raise ValueError('Reference horizon or sampling timestep differ from study.')
    base = configuration.design
    p = jax.jit(jax.vmap(lambda d:base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])))(jnp.asarray(data['designs']))
    np.testing.assert_allclose(p,data['parameters'],rtol=1e-12,atol=1e-12)
    # Contact hooks changed source hashes. Confirm numerical compatibility on
    # the first macro interval of every phase/initial condition independently.
    physical_dt = recorded['simulation']['timestep_s']
    n = round(settings['neural_timestep_s']/physical_dt)
    if n<1 or not np.isclose(n*physical_dt,settings['neural_timestep_s'],rtol=0,atol=1e-12):
        raise ValueError('Reference sampling must contain an integer number of physical steps.')
    def first_step(x,p,start):
        return jax.lax.fori_loop(0,n,lambda i,s:dynamics_step(s,start+i*physical_dt,p,dt=physical_dt),x)
    actual = jax.jit(jax.vmap(first_step))(jnp.asarray(data['states'][:,0]),p,jnp.asarray(data['start_times']))
    np.testing.assert_allclose(actual,data['states'][:,1],rtol=1e-9,atol=1e-10)
    sources = list((Path(__file__).resolve().parents[1]/'src'/'walking_robot').glob('*.py'))
    sources += [Path(__file__),Path(__file__).with_name('contact_model_gradients.py'),Path(__file__).with_name('contact_sensitivity_study.py')]
    return data,dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        implementation_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        original_fingerprint=provenance['fingerprint'],reference_compatibility='All parameters and first macro intervals agree.')


def plot_comparison(data,ids,identity,selected,smoothed,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for col,k in enumerate([0,len(ids)//2]):
        p = data['parameters'][ids[k:k+1]]
        paths = [(data['states'][ids[k:k+1]],'JAX reference'),(identity[k:k+1],'Analytical, coarse step'),
                 (selected[k:k+1],'Selected contact network'),(smoothed[k:k+1],'Smoothed contact, fine step')]
        for x,label in paths:
            com,vel,feet = observables(x,p)
            axes[0,col].plot(data['times'],(com[0,:,0]-com[0,0,0])*1000,label=label)
            axes[1,col].plot(data['times'],vel[0,:,0],label=label)
            axes[2,col].plot(data['times'],feet[0,:,0]*1000,label=label+' front')
        axes[0,col].set_title(f"Held-out design {data['design_id'][ids[k]]}, variant {data['variant'][ids[k]]}")
        for row,label in enumerate(['COM displacement (mm)','COM velocity (m/s)','Front foot height (mm)']):
            axes[row,col].set_ylabel(label)
            axes[row,col].set_xlabel('Elapsed time (s)')
            axes[row,col].grid(alpha=.25)
            axes[row,col].legend(fontsize=7)
        axes[2,col].axhline(0,color='black',lw=.7)
    fig.savefig(output/'comparison.png',dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/contact_structure.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/contact_structure'))
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--skip-video',action='store_true')
    args = parser.parse_args()
    settings = yaml.safe_load(args.study.read_text())
    for key in ['network_width','integration_substeps','training_iterations','batch_size','window_steps','validation_every']:
        if type(settings[key]) is not int or settings[key]<1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ['neural_timestep_s','horizon_s','max_log_correction','learning_rate','max_speed_error_m_s','max_foot_height_rmse_m','ranking_tolerance_m_s']:
        if not np.isfinite(settings[key]) or settings[key]<=0:
            raise ValueError(f'{key} must be finite and positive.')
    if settings['training_mode'] not in ['hard','smooth'] or not 0<settings['min_ranking_agreement']<=1:
        raise ValueError('Invalid contact mode or ranking threshold.')
    if type(settings['seed']) is not int or settings['seed']<0:
        raise ValueError('Seed must be a nonnegative integer.')
    root = args.study.resolve().parent
    configuration = DesignRun.from_file(root/settings['design'])
    if configuration.design.profile!='matlab':
        raise ValueError('This experiment requires the reconciled MATLAB profile.')
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True)
    data,provenance = load_reference(settings,configuration,root)
    print('Reference compatibility checked; training uses admissible training designs only.',flush=True)
    count,dt = len(data['times'])-1,settings['neural_timestep_s']
    window = settings['window_steps']
    if window>=count:
        raise ValueError('Window must be shorter than the trajectory.')
    smoothing = ContactSmoothing(settings['gap_width_um']*1e-6,settings['velocity_width_m_s'],settings['force_width_uN']*1e-6)
    options = dict(mode=settings['training_mode'],smoothing=smoothing,max_log_correction=settings['max_log_correction'])
    substeps = settings['integration_substeps']
    def build_evaluator(mode,substeps):
        return jax.jit(jax.vmap(lambda x,p,t,w:contact_rollout(x,p,dt,count,w,start_time=t,substeps=substeps,
            return_substeps=True,**dict(options,mode=mode))[1],in_axes=(0,0,0,None)))
    run = build_evaluator(settings['training_mode'],substeps)
    def evaluate(weights,ids,runner=run,micro=substeps):
        started = time.perf_counter()
        full = np.asarray(runner(jnp.asarray(data['states'][ids,0]),jnp.asarray(data['parameters'][ids]),
                                jnp.asarray(data['start_times'][ids]),weights))
        report = metrics(data,ids,full,settings,configuration,micro)
        report['evaluation_including_compilation_s'] = time.perf_counter()-started
        return report,full
    valid_ids = np.flatnonzero(data['split']==1)
    test_ids = np.flatnonzero(data['split']==2)
    train_ids = np.flatnonzero((data['split']==0)&data['passed'])
    if len(train_ids)<3 or data['passed'][valid_ids].sum()<3 or data['passed'][test_ids].sum()<3:
        raise ValueError('Insufficient admissible trajectories in a design split.')
    weights = initialize_contact_network(settings['seed'],settings['network_width'])
    identity_report,identity = evaluate(weights,valid_ids)
    write(args.output/'identity_validation.json',identity_report)
    print(json.dumps(dict(identity_speed_mae=identity_report['mean_speed_error_m_s'],
        physical_pass_count=identity_report['physically_passing_prediction_count'])),flush=True)
    best,selected_iteration,best_report = weights,0,identity_report
    history = []
    started = time.perf_counter()
    if args.evaluate_only:
        info = json.loads((args.output/'training.json').read_text())
        if info['configuration']!=settings or info['dataset_sha256']!=provenance['dataset_sha256']:
            raise ValueError('Checkpoint configuration or reference mismatch.')
        with np.load(args.output/'contact_network.npz',allow_pickle=False) as checkpoint:
            best = {k:jnp.asarray(checkpoint[k]) for k in checkpoint.files}
        selected_iteration = info['selected_iteration']
    else:
        rng = np.random.default_rng(settings['seed'])
        x,p,start = jnp.asarray(data['states'][train_ids]),jnp.asarray(data['parameters'][train_ids]),jnp.asarray(data['start_times'][train_ids])
        reference_feet = points(data['states'][train_ids],data['parameters'][train_ids,None,:])['feet'][...,1]<0
        events = np.argwhere(np.any(reference_feet[:,1:]!=reference_feet[:,:-1],axis=-1))
        delta_scale = jnp.asarray(np.maximum(np.std(np.diff(data['states'][train_ids,:,6:],axis=1),axis=(0,1)),1e-12))
        def loss(w,initial,target,p,t):
            pred = jax.vmap(lambda a,b,c:contact_rollout(a,b,dt,window,w,start_time=c,substeps=substeps,**options)[1][1:])(initial,p,t)
            state = jnp.mean(((pred-target)/jnp.asarray(STATE_SCALES))**2)
            first = jnp.mean(((pred[:,0,6:]-target[:,0,6:])/delta_scale)**2)
            pc,pv,pf = observables(pred,p,xp=jnp)
            tc,tv,tf = observables(target,p,xp=jnp)
            scales = jax.vmap(lambda tt,xx,pp:contact_multipliers(w,tt,xx,pp,settings['max_log_correction']))(t,initial,p)
            return (state+.25*first+jnp.mean(((pc-tc)/.0005)**2)+jnp.mean(((pv-tv)/.1)**2)+
                    jnp.mean(((pf-tf)/.0001)**2)+1e-3*jnp.mean(jnp.log(scales)**2))
        moments = jax.tree.map(jnp.zeros_like,weights)
        variance = jax.tree.map(jnp.zeros_like,weights)
        @jax.jit
        def update(w,m,v,iteration,initial,target,p,t):
            value,g = jax.value_and_grad(loss)(w,initial,target,p,t)
            norm = jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree.leaves(g)))
            finite = jnp.isfinite(value)&jnp.isfinite(norm)
            g = jax.tree.map(lambda a:jnp.where(finite,a/jnp.maximum(1.,norm),jnp.zeros_like(a)),g)
            new_m = jax.tree.map(lambda a,b:.9*a+.1*b,m,g)
            new_v = jax.tree.map(lambda a,b:.999*a+.001*b*b,v,g)
            rate = settings['learning_rate']*(.25+.75*.5*(1+jnp.cos(jnp.pi*iteration/settings['training_iterations'])))
            candidate = jax.tree.map(lambda a,b,c:a-rate*(b/(1-.9**iteration))/(jnp.sqrt(c/(1-.999**iteration))+1e-8),w,new_m,new_v)
            choose = lambda new,old:jax.tree.map(lambda a,b:jnp.where(finite,a,b),new,old)
            return choose(candidate,w),choose(new_m,m),choose(new_v,v),value,norm,finite
        rejected = 0
        for iteration in range(1,settings['training_iterations']+1):
            rows = rng.integers(len(train_ids),size=settings['batch_size'])
            positions = rng.integers(count-window+1,size=len(rows))
            if len(events):
                n = len(rows)//2
                chosen = events[rng.integers(len(events),size=n)]
                rows[:n] = chosen[:,0]
                positions[:n] = np.clip(chosen[:,1]-rng.integers(window,size=n),0,count-window)
            initial = x[rows,positions]
            target = x[rows[:,None],positions[:,None]+np.arange(1,window+1)]
            weights,moments,variance,value,norm,finite = update(weights,moments,variance,jnp.asarray(iteration,dtype=jnp.float64),
                initial,target,p[rows],start[rows]+jnp.asarray(positions)*dt)
            if not bool(finite):
                rejected += 1
            if iteration==1 or iteration%settings['validation_every']==0 or iteration==settings['training_iterations']:
                report,_ = evaluate(weights,valid_ids)
                history.append(dict(iteration=iteration,loss=float(value),gradient_norm=float(norm),
                    validation_score=report['selection_score'],speed_mae_m_s=report['mean_speed_error_m_s'],
                    physical_pass_count=report['physically_passing_prediction_count'],rejected_nonfinite_updates=rejected))
                if np.isfinite(report['selection_score']) and report['selection_score']<best_report['selection_score']:
                    best,best_report,selected_iteration = weights,report,iteration
                info = dict(configuration=settings,dataset_sha256=provenance['dataset_sha256'],history=history,
                    selected_iteration=selected_iteration,training_elapsed_s=time.perf_counter()-started,
                    implementation_sources=provenance['implementation_sources'],
                    rejected_nonfinite_updates=rejected)
                write(args.output/'training.json',info)
                np.savez_compressed(args.output/'contact_network.npz',**{k:np.asarray(v) for k,v in best.items()})
                print(json.dumps(history[-1]),flush=True)
        write(args.output/'last_candidate_validation.json',report)
        np.savez_compressed(args.output/'last_contact_network.npz',**{k:np.asarray(v) for k,v in weights.items()})
    selected_report,selected = evaluate(best,valid_ids)
    test_report,test = evaluate(best,test_ids)
    base_ids = np.flatnonzero((data['design_id']==0)&(data['variant']==0))
    base_report,base = evaluate(best,base_ids)
    physical_substeps = round(dt/configuration.simulation.timestep_s)
    smooth_report,smoothed = evaluate(None,valid_ids,build_evaluator('smooth',physical_substeps),physical_substeps)
    for name,report in [('validation',selected_report),('test',test_report),('base',base_report),('smoothed_validation',smooth_report)]:
        write(args.output/(name+'.json'),report)
    np.savez_compressed(args.output/'trajectories.npz',validation_ids=valid_ids,test_ids=test_ids,identity=identity,
        selected=selected,test=test,base=base,smoothed=smoothed)
    # Sensitivities are separate evidence; missing or failed reports never
    # authorize optimization. The selected network is not optimized here.
    sensitivity = {}
    for mode in ['hard','smooth']:
        path = root/settings[mode+'_sensitivity_report']
        if path.exists():
            r = json.loads(path.read_text())
            matches = r.get('status')=='complete' and r.get('horizon_s')==settings['horizon_s'] and r.get('dataset_sha256')==provenance['dataset_sha256']
            sensitivity[mode] = dict(report=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                compatible=matches,passed=bool(matches and r['sensitivity_gate_passed']))
        else:
            sensitivity[mode] = dict(passed=False,reason='Reference sensitivity report missing.')
    report = dict(configuration=settings,provenance=provenance,selected_iteration=selected_iteration,
        identity_validation=identity_report,selected_validation=selected_report,selected_test=test_report,
        selected_base=base_report,smoothed_validation=smooth_report,sensitivities=sensitivity,
        optimization_ready=False,accepted_design=None,
        optimization_block_reasons=['Full design-gradient fidelity is not certified; run tools/contact_model_gradients.py.',
            'This diagnostic tool does not implement an optimization loop.'],
        scope='Contact-only JAX comparator; no NeRD architecture or acceleration replacement. No optimization performed.',
        limitations=['Provisional physical limits; sampled RK4 substeps only.',
            'Original trace regularization retained, including its small force-balance bias.',
            'Full design-gradient fidelity of the selected network is not certified.',
            'Analytical solves are retained; no measured runtime speedup is claimed.'])
    write(args.output/'report.json',report)
    plot_comparison(data,valid_ids,identity[:,::substeps],selected[:,::substeps],smoothed[:,::physical_substeps],args.output)
    if not args.skip_video:
        from walking_robot.visualization import render_trajectory
        for label,states in [('reference',data['states'][base_ids[0]]),('selected',base[0])]:
            times = data['times'] if label=='reference' else np.arange(len(states))*dt/substeps
            render_trajectory(times,states,data['parameters'][base_ids[0]],args.output/label,configuration.physical_limits,
                formats=('mp4',),max_frames=240,fps=30,playback_speed=.025,stop_at_violation=False)
    print(json.dumps(dict(selected_iteration=selected_iteration,trajectory_gate_passed=selected_report['trajectory_gate_passed'],
                         optimization_ready=False)),flush=True)


if __name__=='__main__':
    main()
