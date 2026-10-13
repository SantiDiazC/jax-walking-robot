"""Train/check an experimental NeRD design/time trajectory decoder.

Validation selects weights; untouched test designs are evaluated only after
selection. Smooth-teacher state sensitivities supervise training designs only.
The optimizer stays disabled unless all trajectory, kinematic, physical and
design-gradient gates pass. No trajectory is clipped or repaired.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
os.environ.setdefault('MPLCONFIGDIR','/tmp/walking-robot-mpl')
import numpy as np
import torch
from generate_smooth_training_data import load
from walking_robot.learning import parameters_torch,STATE_SCALES,DESIGN_LOWER,DESIGN_UPPER
from walking_robot.learning_study import trajectory_observables,ranking_agreement,write_json
from walking_robot.trajectory_nerd import TrajectoryNeRD,predict_trajectory
from walking_robot.validation import inspect_trajectory


def tensor(value,device,dtype=torch.float32):
    return torch.as_tensor(value,device=device,dtype=dtype)


def units(designs):
    return (designs-np.array(DESIGN_LOWER))/(np.array(DESIGN_UPPER)-DESIGN_LOWER)


def evaluate(model,data,ids,s,config,device,*,dense=False):
    u = tensor(units(data['designs'][ids]),device,next(model.parameters()).dtype)
    tt = tensor(data['times'],device,u.dtype)
    with torch.no_grad():
        prediction = predict_trajectory(model,u,tt).cpu().numpy()
    ref = data['states'][ids]
    pc,_,pf = trajectory_observables(prediction,data['parameters'][ids])
    rc,_,rf = trajectory_observables(ref,data['parameters'][ids])
    rows = []
    for k,i in enumerate(ids):
        row = dict(id=int(i),design=data['designs'][i].tolist(),teacher_passed=bool(data['passed'][i]),
            state_rmse=float(np.sqrt(np.mean(((prediction[k]-ref[k])/STATE_SCALES)**2))),
            foot_height_rmse_m=float(np.sqrt(np.mean((pf[k]-rf[k])**2))),horizons=[])
        for h in [.2,1.,s['horizon_s']]:
            end = round(h/s['output_timestep_s']);half = end//2
            a,b = [(c[k,end,0]-c[k,half,0])/(h/2) for c in [pc,rc]]
            row['horizons'].append(dict(horizon_s=h,neural_speed_m_s=float(a),teacher_speed_m_s=float(b),
                                        speed_error_m_s=float(abs(a-b))))
        row['physical'] = inspect_trajectory(data['times'],prediction[k],data['parameters'][i],config.physical_limits)
        if dense:
            # Separate two sampling resolutions; no interpolated repaired state.
            row['physical_refinements'] = []
            for dt in [1e-5,5e-6]:
                t = np.arange(round(s['horizon_s']/dt)+1)*dt
                with torch.no_grad():
                    x = predict_trajectory(model,u[k:k+1],tensor(t,device,u.dtype)).cpu().numpy()[0]
                validation = inspect_trajectory(t,x,data['parameters'][i],config.physical_limits)
                row['physical_refinements'].append(dict(sample_timestep_s=dt,validation=validation))
        row['passed'] = bool(row['teacher_passed'] and row['physical']['passed']
            and all(v['validation']['passed'] for v in row.get('physical_refinements',[]))
            and row['state_rmse']<=s['max_state_rmse']
            and row['foot_height_rmse_m']<=s['max_foot_height_rmse_m']
            and all(v['speed_error_m_s']<=s['max_speed_error_m_s'] for v in row['horizons']))
        rows.append(row)
    ranks = [dict(horizon_s=h,**ranking_agreement(
        [r['horizons'][j]['teacher_speed_m_s'] for r in rows],
        [r['horizons'][j]['neural_speed_m_s'] for r in rows],s['ranking_tolerance_m_s']))
        for j,h in enumerate([.2,1.,s['horizon_s']])]
    score = np.mean([r['state_rmse']+r['foot_height_rmse_m']/s['max_foot_height_rmse_m']+
        np.mean([h['speed_error_m_s']/s['max_speed_error_m_s'] for h in r['horizons']]) for r in rows])
    return dict(cases=rows,ranking=ranks,selection_score=float(score),
        trajectory_gate_passed=bool(len(rows)>=3 and all(r['passed'] for r in rows)
            and all(r['eligible_pairs']>=1 and r['agreement'] is not None
                    and r['agreement']>=s['min_ranking_agreement'] for r in ranks))),prediction


def derivatives(model,data,ids,s,device):
    """Double precision neural AD vs teacher variational derivatives and self FD."""
    model = copy.deepcopy(model).double().eval()
    rows = []
    for i in ids:
        center = tensor(units(data['designs'][i:i+1]),device,torch.float64)
        # The teacher path is locally linearized only for computing its exact
        # first derivative. Geometry/parameters are recomputed differentiably.
        for h in [.2,1.,s['horizon_s']]:
            end = round(h/s['output_timestep_s']);half = end//2
            tt = tensor([h/2,h],device,torch.float64)
            x = tensor(data['states'][i,[half,end]],device,torch.float64)
            sx = tensor(data['sensitivities'][i,[half,end]],device,torch.float64)
            def objective(u,teacher=False):
                if teacher:
                    states = x[None]+torch.einsum('tka,ba->btk',sx,u-center)
                else:
                    states = predict_trajectory(model,u,tt)
                p = parameters_torch(model.physical_designs(u),model.base_design)
                c,_,_ = trajectory_observables(states,p,xp=torch)
                return (c[:,-1,0]-c[:,0,0])/(h/2)
            u = center.clone().requires_grad_(True)
            nv = objective(u);ng, = torch.autograd.grad(nv.sum(),u)
            rv = objective(u,True);rg, = torch.autograd.grad(rv.sum(),u)
            ng,rg = ng.detach().cpu().numpy()[0],rg.detach().cpu().numpy()[0]
            eps = s['sensitivity_epsilon']
            fd = []
            regime = parameters_torch(model.physical_designs(center),model.base_design)[:,[22,26,41]]
            regime_ok = True
            for e in [eps,eps/3]:
                probes = center.repeat(4,1)
                probes[0,0]-=e;probes[1,0]+=e;probes[2,1]-=e;probes[3,1]+=e
                pp = parameters_torch(model.physical_designs(probes),model.base_design)
                same = bool(((pp[:,[22,26,41]]==regime).all()) and ((probes>=0)&(probes<=1)).all())
                regime_ok &= same
                with torch.no_grad():
                    v = objective(probes).cpu().numpy()
                g = np.array([(v[1]-v[0])/(2*e),(v[3]-v[2])/(2*e)])
                fd.append(dict(unit_epsilon=e,same_discrete_regime=same,gradient=g.tolist(),
                    relative_error=float(np.linalg.norm(g-ng)/max(np.linalg.norm(ng),1e-10))))
            error = float(np.linalg.norm(ng-rg)/max(np.linalg.norm(rg),1e-10))
            cosine = float(np.dot(ng,rg)/max(np.linalg.norm(ng)*np.linalg.norm(rg),1e-16))
            rows.append(dict(id=int(i),horizon_s=h,neural_value_m_s=float(nv.detach()),teacher_value_m_s=float(rv.detach()),
                neural_gradient_per_unit=ng.tolist(),teacher_gradient_per_unit=rg.tolist(),
                relative_error=error,cosine=cosine,self_finite_differences=fd,
                passed=bool(regime_ok and error<=s['max_gradient_relative_error']
                    and cosine>=s['min_gradient_cosine'] and all(r['relative_error']<=.05 for r in fd))))
    # Independent q-dot consistency, evaluated away from time-domain edges.
    times = np.arange(1e-4,s['horizon_s'],.001)
    uu = tensor(units(data['designs'][ids]),device,torch.float64)
    tt = tensor(times,device,torch.float64)
    with torch.no_grad():
        x = predict_trajectory(model,uu,tt)
        plus = predict_trajectory(model,uu,tt+1e-6)
        minus = predict_trajectory(model,uu,tt-1e-6)
    errors = ((plus[...,:6]-minus[...,:6])/(2e-6)-x[...,6:12])/x.new_tensor(STATE_SCALES[6:12])
    rms = torch.sqrt(errors.square().mean((1,2))).cpu().numpy()
    kinematics = [dict(id=int(i),normalized_velocity_rmse=float(e),passed=bool(e<=s['max_kinematic_rmse'])) for i,e in zip(ids,rms)]
    return dict(cases=rows,kinematics=kinematics,
        gradient_gate_passed=bool(rows and all(r['passed'] for r in rows)),
        kinematic_gate_passed=bool(kinematics and all(r['passed'] for r in kinematics)))


def train(model,data,s,config,device,output,provenance):
    ids = np.flatnonzero((data['split']==0)&data['passed'])
    if len(ids)<6:
        raise ValueError('Insufficient physically admissible training designs.')
    u = tensor(units(data['designs'][ids]),device)
    times = tensor(data['times'],device)
    x = tensor(data['states'][ids],device)
    sx = tensor(data['sensitivities'][ids],device)
    with torch.no_grad():
        initial,_ = model.initial(u)
        scale = model.time_scale(times).clamp_min(1e-6)
        z = (x[:,1:]-initial[:,None])/scale[None,1:]
        model.offset_mean.copy_(z.mean((0,1)))
        floors = z.new_tensor([.01,1e-5,.001,.001,.001,.001,.01,.01,1.,1.,1.,1.,1e-10,1e-10])
        model.offset_scale.copy_(torch.maximum(z.std((0,1)),floors))
        scaled_sx = sx[:,1:]/scale[None,1:,:,None]
        sensitivity_scale = torch.maximum(torch.sqrt(scaled_sx.square().mean((0,1,3))),floors)
    optimizer = torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=1e-6)
    rng = np.random.default_rng(s['seed']+1)
    log,best = [],float('inf')
    started = time.perf_counter()
    for iteration in range(1,s['iterations']+1):
        model.train()
        batch = s['batch_size']
        ii = torch.tensor(rng.integers(0,len(ids),batch),device=device)
        jj = rng.integers(1,len(times),batch)
        # Half the samples emphasize early transient dynamics (first .2 s).
        jj[:batch//2] = rng.integers(1,round(.2/s['output_timestep_s'])+1,batch//2)
        jj = torch.tensor(jj,device=device)
        d,t,target = u[ii],times[jj],x[ii,jj]
        ts = model.time_scale(t).clamp_min(1e-6)
        predicted = model(d,t)
        state_loss = (((predicted-target)/ts/model.offset_scale).square()).mean()
        # Shared random direction gives both partial derivatives over batches.
        axis = iteration%2
        delta = d.new_zeros(2);delta[axis] = s['sensitivity_epsilon']
        plus,minus = model(d+delta,t),model(d-delta,t)
        predicted_s = (plus-minus)/(2*s['sensitivity_epsilon'])
        with torch.no_grad():
            pp = parameters_torch(model.physical_designs(d+delta),model.base_design)
            pm = parameters_torch(model.physical_designs(d-delta),model.base_design)
            mask = (pp[:,[22,26,41]]==pm[:,[22,26,41]]).all(-1)&((d-delta>=0)&(d+delta<=1)).all(-1)
        sensitivity_loss = (((predicted_s-sx[ii,jj,:,axis])/ts/sensitivity_scale)[mask].square()).mean()
        p = parameters_torch(model.physical_designs(d),model.base_design)
        _,_,pf = trajectory_observables(predicted[:,None],p,xp=torch)
        _,_,rf = trajectory_observables(target[:,None],p,xp=torch)
        foot_loss = ((pf-rf)/s['max_foot_height_rmse_m']).square().mean()
        # Finite-time derivative avoids unsupported second derivatives through
        # upstream flash attention. The independent final check uses float64.
        kinematic_loss = predicted.new_zeros(())
        if s['kinematic_weight']>0:
            eps_t = 1e-5
            qdot = (model(d,t+eps_t)[...,:6]-model(d,t-eps_t)[...,:6])/(2*eps_t)
            kinematic_loss = ((qdot-predicted[...,6:12])/predicted.new_tensor(STATE_SCALES[6:12])).square().mean()
        loss = state_loss+foot_loss+s['sensitivity_weight']*sensitivity_loss+s['kinematic_weight']*kinematic_loss
        task_loss = loss.new_zeros(())
        if s.get('task_position_scale_m_s',0)>0:
            task_loss = ((predicted[:,0]-target[:,0])/t.clamp_min(.01)/s['task_position_scale_m_s']).square().mean()
            # Explicit late-half objective supervision at the same horizons
            # as validation, exclusively on training designs.
            hs = torch.tensor(rng.choice([.2,1.,s['horizon_s']],batch),device=device,dtype=t.dtype)
            end = (hs/s['output_timestep_s']).round().long()
            begin = end//2
            pa,pb = model(d,hs/2),model(d,hs)
            ca,_,_ = trajectory_observables(pa[:,None],p,xp=torch)
            cb,_,_ = trajectory_observables(pb[:,None],p,xp=torch)
            ra,_,_ = trajectory_observables(x[ii,begin,None],p,xp=torch)
            rb,_,_ = trajectory_observables(x[ii,end,None],p,xp=torch)
            objective_loss = (((cb[:,0,0]-ca[:,0,0]-rb[:,0,0]+ra[:,0,0])/(hs/2))/s['max_speed_error_m_s']).square().mean()
            task_loss = task_loss+s.get('objective_weight',.1)*objective_loss
            loss = loss+task_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss; no state repair applied.')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if iteration%s['validation_every']==0 or iteration==s['iterations']:
            model.eval()
            validation,_ = evaluate(model,data,np.flatnonzero(data['split']==1),s,config,device)
            if s.get('selection_gradient_weight',0)>0:
                validation['derivatives'] = derivatives(model,data,np.flatnonzero(data['split']==1),s,device)
                mean_error = np.mean([c['relative_error'] for c in validation['derivatives']['cases']])
                validation['selection_score'] += s['selection_gradient_weight']*mean_error/s['max_gradient_relative_error']
            row = dict(iteration=iteration,loss=float(loss.detach()),state_loss=float(state_loss.detach()),
                sensitivity_loss=float(sensitivity_loss.detach()),kinematic_loss=float(kinematic_loss.detach()),
                task_loss=float(task_loss.detach()),
                elapsed_s=time.perf_counter()-started,validation=validation)
            log.append(row)
            if validation['selection_score']<best:
                best = validation['selection_score']
                torch.save(dict(state_dict=model.state_dict(),settings=s,configuration=asdict(config),
                    iteration=iteration,provenance=provenance,upstream=model.upstream),output/'model.pt')
            write_json(output/'training.json',dict(log=log,selected_score=best,provenance=provenance))
            print(f"Iteration {iteration}: loss {float(loss.detach()):.5g}, validation score {validation['selection_score']:.4g}, best {best:.4g}, elapsed {row['elapsed_s']:.1f}s",flush=True)
    saved = torch.load(output/'model.pt',map_location=device,weights_only=False)
    model.load_state_dict(saved['state_dict']);model.eval()
    return saved


def plot(data,ids,prediction,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,len(ids),figsize=(5*len(ids),9),squeeze=False,constrained_layout=True)
    for col,i in enumerate(ids):
        for states,label in [(data['states'][i:i+1],'Smooth reference'),(prediction[col:col+1],'NeRD trajectory decoder')]:
            c,v,f = trajectory_observables(states,data['parameters'][i:i+1])
            axes[0,col].plot(data['times'],c[0,:,0],label=label)
            # A small final-time zoom shows actual foot motion, not just a band.
            axes[1,col].plot(data['times'][-201:],v[0,-201:,0],label=label)
            axes[2,col].plot(data['times'][-201:],1000*f[0,-201:,0],label=label+' front')
            axes[2,col].plot(data['times'][-201:],1000*f[0,-201:,1],alpha=.6,label=label+' hind')
        axes[0,col].set_title(f'Held-out design {i}: {data["designs"][i,0]:.3f} um, {data["designs"][i,1]:.2f} deg')
        for row,label in enumerate(['COM x (m)','COM vx (m/s)','Foot height (mm)']):
            axes[row,col].set_ylabel(label);axes[row,col].set_xlabel('Time (s)')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=7)
        axes[2,col].axhline(-.25,color='red',ls='--')
    fig.savefig(output/'comparison.png',dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/nerd_trajectory.yaml'))
    parser.add_argument('--data',type=Path,default=Path('outputs/nerd_trajectory/data'))
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_trajectory/training'))
    parser.add_argument('--nerd-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--initialize-from',type=Path,help='Same-data checkpoint; starts a separate recorded refinement run.')
    parser.add_argument('--animate',action='store_true')
    args = parser.parse_args()
    s,config,designs,split = load(args.study)
    for name in ['iterations','validation_every','batch_size','width','layers']:
        if type(s[name]) is not int or s[name]<1:
            raise ValueError(f'{name} must be a positive integer.')
    for name in ['sensitivity_epsilon','learning_rate','max_speed_error_m_s','max_state_rmse',
                 'max_foot_height_rmse_m','max_kinematic_rmse','max_gradient_relative_error']:
        if not np.isfinite(s[name]) or s[name]<=0:
            raise ValueError(f'{name} must be positive and finite.')
    args.output.mkdir(parents=True,exist_ok=True)
    datafile = args.data/'dataset.npz'
    source_report = json.loads((args.data/'report.json').read_text())
    data_sha = hashlib.sha256(datafile.read_bytes()).hexdigest()
    if data_sha!=source_report['dataset_sha256']:
        raise ValueError('Dataset digest mismatch.')
    with np.load(datafile,allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    if not np.array_equal(data['designs'],designs) or not np.array_equal(data['split'],split):
        raise ValueError('Dataset design/split mismatch.')
    expected = dict(configuration=asdict(config),settings={k:s[k] for k in source_report['fingerprint']['settings']})
    if any(expected[k]!=source_report['fingerprint'][k] for k in expected):
        raise ValueError('Teacher configuration mismatch.')
    provenance = dict(dataset_sha256=data_sha,teacher=source_report['fingerprint'],
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'src/walking_robot/trajectory_nerd.py').read_bytes()).hexdigest(),
        torch_version=torch.__version__,device=args.device,cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name() if args.device=='cuda' else None,
        adapter_sources={n:hashlib.sha256((Path(__file__).resolve().parents[1]/'src/walking_robot'/n).read_bytes()).hexdigest()
                         for n in ['learning.py','trajectory_nerd.py']})
    torch.manual_seed(s['seed']);torch.set_num_threads(4)
    model = TrajectoryNeRD(args.nerd_root,config.design,horizon=s['horizon_s'],width=s['width'],layers=s['layers'],
                           clearance_um=config.simulation.initial_clearance_um,
                           consistent_velocity=s.get('consistent_velocity',False)).to(args.device)
    if args.initialize_from:
        if args.evaluate_only:
            raise ValueError('Initialization and evaluation-only are mutually exclusive.')
        initial_checkpoint = torch.load(args.initialize_from,map_location=args.device,weights_only=False)
        if initial_checkpoint['provenance']['dataset_sha256']!=data_sha or initial_checkpoint['configuration']!=asdict(config):
            raise ValueError('Initialization requires the same dataset and physical configuration.')
        if initial_checkpoint['upstream']['sha256']!=model.upstream['sha256']:
            raise ValueError('Initialization upstream source mismatch.')
        model.load_state_dict(initial_checkpoint['state_dict'])
        provenance['initialization'] = dict(path=str(args.initialize_from),iteration=initial_checkpoint['iteration'],
            sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest())
    if args.evaluate_only:
        saved = torch.load(args.output/'model.pt',map_location=args.device,weights_only=False)
        if saved['provenance']['dataset_sha256']!=data_sha or saved['settings']!=s:
            raise ValueError('Checkpoint data/settings mismatch.')
        if saved['upstream']['sha256']!=model.upstream['sha256']:
            raise ValueError('Upstream source mismatch.')
        if saved['provenance'].get('adapter_sources')!=provenance['adapter_sources']:
            raise ValueError('Checkpoint adapter source mismatch; use the recorded source snapshot.')
        model.load_state_dict(saved['state_dict']);model.eval()
    else:
        if (args.output/'model.pt').exists():
            raise ValueError('Training output exists; use a new directory or --evaluate-only.')
        saved = train(model,data,s,config,args.device,args.output,provenance)
    reports = {}
    # Full double-precision verification occurs only after selecting weights.
    final = copy.deepcopy(model).double().eval()
    for name,part in [('validation',1),('test',2)]:
        ids = np.flatnonzero(data['split']==part)
        start = time.perf_counter()
        report,predicted = evaluate(final,data,ids,s,config,args.device,dense=True)
        report['derivatives'] = derivatives(final,data,ids,s,args.device)
        report['evaluation_elapsed_s'] = time.perf_counter()-start
        reports[name] = report
        write_json(args.output/(name+'.json'),report)
        np.savez_compressed(args.output/(name+'_prediction.npz'),ids=ids,states=predicted,times=data['times'],
                            designs=data['designs'][ids],parameters=data['parameters'][ids])
        if name=='test':
            plot(data,ids,predicted,args.output)
            if args.animate:
                from walking_robot.visualization import render_trajectory
                for label,x in [('neural',predicted[0]),('reference',data['states'][ids[0]])]:
                    render_trajectory(data['times'],x,data['parameters'][ids[0]],args.output/label,
                        config.physical_limits,max_frames=150,formats=['mp4','html'],stop_at_violation=False)
    ready = all(r['trajectory_gate_passed'] and r['derivatives']['gradient_gate_passed']
                and r['derivatives']['kinematic_gate_passed'] for r in reports.values())
    report = dict(settings=s,selected_iteration=saved['iteration'],provenance=saved['provenance'],
        evaluation_provenance=provenance,
        checkpoint_sha256=hashlib.sha256((args.output/'model.pt').read_bytes()).hexdigest(),
        upstream=model.upstream,results=reports,surrogate_gates_passed=bool(ready),
        optimization_enabled=False,
        limitation='Canonical supported start, fixed configured voltage and initial actuator phase; not an arbitrary-state simulator.',
        next_step='Independent original-contact resimulation of neural proposals is required before design acceptance.' if ready
                  else 'Failed surrogate gates prohibit neural design optimization.')
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(selected_iteration=saved['iteration'],surrogate_gates_passed=bool(ready),optimization_enabled=False)),flush=True)


if __name__=='__main__':
    main()
