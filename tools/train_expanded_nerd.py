"""Train a three-variable, fixed-actuation NeRD trajectory adapter.

Disjoint design splits, training-only normalization, validation-only selection.
Test results never choose weights. Neural optimization remains experimental;
every proposed design requires independent hard-contact solver verification.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR','/tmp/walking-robot-mpl')
import numpy as np
import torch
from generate_expanded_data import load_training
from walking_robot.expanded_learning import ExpandedNeRD,expanded_parameters_torch,coordinates,expanded_trajectory
from walking_robot.geometry import points,point_velocities
from walking_robot.physical_search import write_json
from walking_robot.validation import inspect_trajectory


def observables(x,p):
    weights = p[...,:4]/p[...,:4].sum(-1,keepdim=True)
    v = (point_velocities(x,p,xp=torch)['centers']*weights[...,None]).sum(-2)
    feet = points(x,p,xp=torch)['feet'][...,1]
    return v,feet


def speed(model,u):
    t = u.new_tensor([model.horizon/2,model.horizon])
    x = expanded_trajectory(model,u,t)
    p = expanded_parameters_torch(model.physical_designs(u),model.base)[:,None]
    c = coordinates(x,p)
    return (c[:,1,0]-c[:,0,0])/(model.horizon/2)


def assess(model,data,ids,config,device,*,stride=10,save=None):
    u = torch.as_tensor(data['unit'][ids],dtype=torch.float32,device=device)
    tt = torch.as_tensor(data['times'][::stride],dtype=torch.float32,device=device)
    with torch.no_grad():
        prediction = expanded_trajectory(model,u,tt).cpu()
        p = torch.as_tensor(data['parameters'][ids,None],dtype=torch.float32)
        ref = torch.as_tensor(data['states'][ids,::stride],dtype=torch.float32)
        pc,rc = coordinates(prediction,p),coordinates(ref,p)
        pv,pf = observables(prediction,p);rv,rf = observables(ref,p)
    rows = []
    for k,i in enumerate(ids):
        v,w = [(c[k,-1,0]-c[k,len(tt)//2,0]).item()/(float(tt[-1])/2) for c in [pc,rc]]
        physical = inspect_trajectory(tt.cpu().numpy(),prediction[k].numpy(),data['parameters'][i],config.physical_limits)
        rows.append(dict(id=int(i),design=data['designs'][i].tolist(),teacher_passed=bool(data['passed'][i]),
            neural_speed_m_s=v,teacher_speed_m_s=w,speed_error_m_s=abs(v-w),
            com_velocity_rmse_m_s=float(torch.mean((pv[k]-rv[k])**2).sqrt()),
            foot_height_rmse_m=float(torch.mean((pf[k]-rf[k])**2).sqrt()),physical=physical))
    score = float(np.mean([r['speed_error_m_s']/.0025+r['com_velocity_rmse_m_s']/.02+r['foot_height_rmse_m']/.0001 for r in rows]))
    if save is not None:
        np.savez_compressed(save,ids=ids,times=tt.cpu().numpy(),states=prediction.numpy(),parameters=data['parameters'][ids])
    return dict(cases=rows,selection_score=score,
        trajectory_gate_passed=all(r['speed_error_m_s']<=.0025 and r['com_velocity_rmse_m_s']<=.02
            and r['foot_height_rmse_m']<=.0001 and r['physical']['passed']==r['teacher_passed'] for r in rows))


def propose(model,settings,seed):
    """Bounded input optimization; no physics acceptance follows from this loss."""
    for p in model.parameters():p.requires_grad_(False)
    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    rows = []
    for target in settings['targets_m_s']:
        u = torch.rand((settings['optimization_starts'],3),generator=g,device=g.device,requires_grad=True)
        opt = torch.optim.Adam([u],lr=settings['optimization_learning_rate'])
        best = None
        for iteration in range(settings['optimization_iterations']):
            opt.zero_grad();v = speed(model,u);loss = (v-target).square()
            k = int(loss.argmin())
            candidate = dict(iteration=iteration,loss=float(loss[k].detach()),unit=u[k].detach().cpu().tolist(),
                design=model.physical_designs(u)[k].detach().cpu().tolist(),neural_speed_m_s=float(v[k].detach()))
            if best is None or candidate['loss']<best['loss']:best=candidate
            loss.mean().backward();opt.step()
            with torch.no_grad():u.clamp_(0,1)
        rows.append(dict(target_m_s=target,**best))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/expanded_nerd.yaml'))
    parser.add_argument('--data',type=Path,default=Path('outputs/expanded_nerd/data/dataset.npz'))
    parser.add_argument('--output',type=Path,default=Path('outputs/expanded_nerd/training'))
    parser.add_argument('--nerd-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--evaluate-only',action='store_true')
    args = parser.parse_args()
    s,config,base,designs,split = load_training(args.study)
    torch.manual_seed(s['seed']);torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is unavailable.')
    with np.load(args.data,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    np.testing.assert_array_equal(data['designs'],designs);np.testing.assert_array_equal(data['split'],split)
    bounds = np.array(s['bounds']);data['unit'] = (designs-bounds[:,0])/(bounds[:,1]-bounds[:,0])
    train,val,test = [np.flatnonzero(split==k) for k in range(3)]
    model = ExpandedNeRD(args.nerd_root,base,s['bounds'],horizon=s['horizon_s'],width=s['width'],layers=s['layers'],
        clearance_um=config.simulation.initial_clearance_um).to(args.device)
    sources = ['expanded_learning','expanded_design','trajectory_nerd','learning','geometry']
    source_dir = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    provenance = dict(dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((source_dir/(n+'.py')).read_bytes()).hexdigest() for n in sources},
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,
        device=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',upstream=model.upstream)
    args.output.mkdir(parents=True,exist_ok=True)
    tensor = lambda x:torch.as_tensor(x,dtype=torch.float32,device=args.device)
    if not args.evaluate_only:
        if (args.output/'model.pt').exists():raise ValueError('Use a fresh output directory; recorded checkpoints are immutable.')
        u,tt = tensor(data['unit'][train]),tensor(data['times'])
        ref,p = tensor(data['states'][train]),tensor(data['parameters'][train])
        with torch.no_grad():
            c = coordinates(ref,p[:,None]);v,feet = observables(ref,p[:,None]);initial,_ = model.initial(u)
            initial_c = coordinates(initial,p)
            scale = model.time_scale(tt)
            # Exclude the initial transient from normalizer fitting only.
            normalized = ((c-initial_c[:,None])/scale.clamp_min(1e-6)[None])[:,tt>.02]
            model.offset_mean.copy_(normalized.mean((0,1)))
            model.offset_scale.copy_(normalized.std((0,1)).clamp_min(1e-6))
            target_speed = (c[:,-1,0]-c[:,len(tt)//2,0])/(s['horizon_s']/2)
        opt = torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=1e-5)
        best,history = float('inf'),[];start = time.monotonic()
        for iteration in range(1,s['iterations']+1):
            model.train();opt.zero_grad(set_to_none=True)
            di = torch.randint(len(train),(s['batch_size'],),device=args.device)
            ti = torch.randint(1,len(tt),(s['batch_size'],),device=args.device)
            x = model(u[di],tt[ti]);cc = coordinates(x,p[di]);vv,ff = observables(x,p[di])
            # All scales fixed using training designs; objectives have physical units.
            coord_loss = ((cc-c[di,ti])/(model.offset_scale*scale[ti]).clamp_min(1e-6)).square().mean()
            velocity_loss = ((vv-v[di,ti])/s['com_velocity_scale_m_s']).square().mean()
            foot_loss = ((ff-feet[di,ti])/s['max_foot_height_rmse_m']).square().mean()
            mean_loss = ((speed(model,u)-target_speed)/s['mean_speed_scale_m_s']).square().mean()
            loss = coord_loss+velocity_loss+foot_loss+mean_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; no repaired trajectory is accepted.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            if iteration==1 or iteration%s['validation_every']==0:
                model.eval();report = assess(model,data,val,config,args.device)
                row = dict(iteration=iteration,loss=float(loss.detach()),validation_score=report['selection_score'],elapsed_s=time.monotonic()-start)
                history.append(row);write_json(args.output/'history.json',history);print(json.dumps(row),flush=True)
                if row['validation_score']<best:
                    best = row['validation_score']
                    torch.save(dict(state_dict=model.state_dict(),settings=s,base=asdict(base),iteration=iteration,provenance=provenance),args.output/'model.pt')
    saved = torch.load(args.output/'model.pt',map_location=args.device,weights_only=False)
    if saved['provenance']!=provenance or saved['settings']!=s:raise ValueError('Checkpoint provenance mismatch.')
    model.load_state_dict(saved['state_dict']);model.eval()
    results = {name:assess(model,data,ids,config,args.device,stride=1,save=args.output/(name+'_prediction.npz'))
        for name,ids in [('validation',val),('test',test)]}
    proposals = propose(model,s,s['seed']+1)
    report = dict(settings=s,selected_iteration=saved['iteration'],provenance=provenance,
        checkpoint_sha256=hashlib.sha256((args.output/'model.pt').read_bytes()).hexdigest(),
        **results,proposals=proposals,optimization_enabled=False,surrogate_only_optimization_enabled=False,
        limitation='No independent three-variable teacher gradient validation yet. Every neural proposal needs hard-contact solver verification.')
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(selected_iteration=saved['iteration'],proposals=proposals)),flush=True)


if __name__=='__main__':main()
