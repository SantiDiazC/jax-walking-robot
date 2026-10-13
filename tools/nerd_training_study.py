"""Train unchanged planar NeRD with varied physical data and equal-step losses."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax
import jax.numpy as jnp
import numpy as np
import torch

from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.geometry import points
from walking_robot.learning import (PlanarNeRD,STATE_SCALES,DESIGN_LOWER,DESIGN_UPPER,
    neural_rollout,parameters_torch)
from walking_robot.learning_study import (StudySettings,generate_dataset,write_json,
    trajectory_observables,sequence_loss,ranking_agreement)
from walking_robot.validation import inspect_trajectory


def evaluate(model,data,ids,settings,configuration,device):
    d = torch.as_tensor(data["designs"][ids],dtype=torch.float32,device=device)
    initial = torch.as_tensor(data["states"][ids,0],dtype=torch.float32,device=device)
    start = torch.as_tensor(data["start_times"][ids],dtype=torch.float32,device=device)
    with torch.no_grad():
        prediction = neural_rollout(model,d,settings.neural_timestep_s,len(data["times"])-1,
                                    initial=initial,start_times=start).cpu().numpy()
    reference = data["states"][ids]
    pc,_,pf = trajectory_observables(prediction,data["parameters"][ids])
    tc,_,tf = trajectory_observables(reference,data["parameters"][ids])
    halfway = (len(data["times"])-1)//2
    duration = data["times"][-1]-data["times"][halfway]
    ps = (pc[:,-1,0]-pc[:,halfway,0])/duration
    ts = (tc[:,-1,0]-tc[:,halfway,0])/duration
    cases = []
    for k,i in enumerate(ids):
        finite = np.isfinite(prediction[k]).all()
        cases.append(dict(id=int(i),design_id=int(data["design_id"][i]),variant=int(data["variant"][i]),
            reference_passed=bool(data["passed"][i]),
            neural_validation=inspect_trajectory(data["times"],prediction[k],data["parameters"][i],configuration.physical_limits),
            state_rmse=float(np.sqrt(np.mean(((prediction[k]-reference[k])/STATE_SCALES)**2))) if finite else float("inf"),
            speed_error_m_s=float(abs(ps[k]-ts[k])) if finite else float("inf"),
            reference_speed_m_s=float(ts[k]),predicted_speed_m_s=float(ps[k]),
            foot_height_rmse_m=float(np.sqrt(np.mean((pf[k]-tf[k])**2))) if finite else float("inf")))
    valid = [r for r in cases if r["reference_passed"]]
    canonical = [r for r in valid if r["variant"]==0]
    ranking = ranking_agreement([r["reference_speed_m_s"] for r in canonical],
                                [r["predicted_speed_m_s"] for r in canonical],settings.ranking_tolerance_m_s)
    false_accept = [r["id"] for r in cases if not r["reference_passed"] and r["neural_validation"]["passed"]]
    # A fixed composite criterion rewards task fidelity as well as state fidelity.
    score = np.mean([r["state_rmse"]+r["speed_error_m_s"]/settings.max_speed_error_m_s+
                     r["foot_height_rmse_m"]/settings.max_foot_height_rmse_m for r in valid]) if valid else float("inf")
    passed = (len(canonical)>=3 and bool(valid) and all(r["neural_validation"]["passed"] and r["state_rmse"]<=.1
              and r["speed_error_m_s"]<=settings.max_speed_error_m_s
              and r["foot_height_rmse_m"]<=settings.max_foot_height_rmse_m for r in valid)
              and not false_accept and ranking["eligible_pairs"]>=3 and ranking["agreement"] is not None
              and ranking["agreement"]>=settings.min_ranking_agreement)
    return dict(cases=cases,selection_score=float(score),trajectory_gate_passed=bool(passed),
                canonical_ranking=ranking,false_accept_ids=false_accept),prediction


def train(settings,configuration,data,root,output,device):
    model = PlanarNeRD(root,base_design=configuration.design,
                       initial_clearance_um=configuration.simulation.initial_clearance_um).to(device)
    train_ids = np.flatnonzero((data["split"]==0)&data["passed"])
    valid_ids = np.flatnonzero(data["split"]==1)
    if len(train_ids)<3 or sum(data["passed"][valid_ids])<3:
        raise ValueError("Insufficient complete physically admissible training/validation trajectories.")
    deltas = np.diff(data["states"][train_ids,:,6:],axis=1).reshape(-1,8)
    with torch.no_grad():
        model.delta_mean.copy_(torch.as_tensor(deltas.mean(0),device=device,dtype=torch.float32))
        model.delta_scale.copy_(torch.as_tensor(np.maximum(deltas.std(0),[1e-5,1e-5,.01,.01,.01,.01,1e-12,1e-12]),
                                              device=device,dtype=torch.float32))
    x = torch.as_tensor(data["states"],device=device,dtype=torch.float32)
    d = torch.as_tensor(data["designs"],device=device,dtype=torch.float32)
    p = torch.as_tensor(data["parameters"],device=device,dtype=torch.float32)
    start_times = torch.as_tensor(data["start_times"],device=device,dtype=torch.float32)
    # Half of windows are anchored around contact transitions, using training only.
    feet = points(data["states"][train_ids],data["parameters"][train_ids,None,:])["feet"][...,1]<0
    events = {int(i):np.flatnonzero(np.any(feet[k,1:]!=feet[k,:-1],axis=-1)) for k,i in enumerate(train_ids)}
    rng = np.random.default_rng(settings.seed+1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings.learning_rate,weight_decay=1e-5)
    total = sum(s["iterations"] for s in settings.curriculum)
    log,best,iteration = [],float("inf"),0
    started = time.perf_counter()
    for stage in settings.curriculum:
        k = stage["steps"]
        for _ in range(stage["iterations"]):
            model.train()
            ids = rng.choice(train_ids,settings.batch_size)
            last = len(data["times"])-k-1
            positions = rng.integers(0,last+1,len(ids))
            for b in range(len(ids)//2):
                if len(events[int(ids[b])]):
                    positions[b] = np.clip(rng.choice(events[int(ids[b])])-rng.integers(0,k),0,last)
            ids = torch.as_tensor(ids,device=device)
            positions = torch.as_tensor(positions,device=device)
            indices = torch.clamp(positions[:,None]+torch.arange(-model.history+1,1,device=device),min=0)
            history = x[ids[:,None],indices]
            times = indices.to(x.dtype)*settings.neural_timestep_s+start_times[ids,None]
            target_indices = positions[:,None]+torch.arange(1,k+1,device=device)
            targets = x[ids[:,None],target_indices]
            initial = history[:,-1]
            predicted = []
            for step in range(k):
                nxt = model.predict_next(history,times,d[ids],settings.neural_timestep_s)
                predicted.append(nxt)
                history = torch.cat([history[:,1:],nxt[:,None]],dim=1)
                next_time = start_times[ids]+(positions+step+1)*settings.neural_timestep_s
                times = torch.cat([times[:,1:],next_time[:,None]],dim=1)
            loss = sequence_loss(torch.stack(predicted,dim=1),targets,initial,p[ids],model.delta_scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; no state repair applied.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            iteration += 1
            # Smooth LR decay while retaining the same architecture and optimizer.
            for group in optimizer.param_groups:
                group["lr"] = settings.learning_rate*(1-.8*iteration/total)
            if iteration%100==0 or iteration==total:
                record = dict(iteration=iteration,window_steps=k,loss=float(loss.detach()),grad_norm=float(grad_norm))
                if iteration%settings.validation_every==0 or iteration==total:
                    model.eval()
                    report,_ = evaluate(model,data,valid_ids,settings,configuration,device)
                    score = report["selection_score"]
                    record.update(validation_score=score,validation_speed_error_m_s=float(np.mean([
                        r["speed_error_m_s"] for r in report["cases"] if r["reference_passed"]])))
                    if np.isfinite(score) and score<best:
                        best = score
                        torch.save(model.state_dict(),output/"model.pt")
                        write_json(output/"selected_validation.json",report)
                log.append(record)
                write_json(output/"training_progress.json",dict(log=log,elapsed_s=time.perf_counter()-started))
                print(json.dumps(record),flush=True)
    if not np.isfinite(best):
        raise RuntimeError("No finite held-out checkpoint; diagnostics retained.")
    model.load_state_dict(torch.load(output/"model.pt",map_location=device,weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    write_json(output/"training.json",dict(log=log,elapsed_s=time.perf_counter()-started,settings=asdict(settings),
        architecture=dict(history=8,width=64,layers=2,head_outputs=8),upstream=model.upstream,
        train_trajectory_ids=train_ids.tolist(),normalization_uses="Admissible training trajectories only",
        checkpoint_selection="Full validation trajectories: state RMSE + scaled late COM speed error + scaled foot-height RMSE; test excluded.",
        best_selection_score=best,device=str(device),torch_version=torch.__version__,jax_version=jax.__version__))
    return model


def check_gradients(model,data,settings,configuration,device,output,*,design_ids=None):
    """Neural autodiff vs physical central differences at two steps and scales.

    Geometry-dependent supported initial states are rebuilt for every design.
    Physical switches can make reference finite differences scale-dependent;
    such cases explicitly fail certification rather than supplying fake targets.
    """
    base = configuration.design
    macro_steps = len(data["times"])-1
    span = np.asarray(DESIGN_UPPER)-DESIGN_LOWER
    candidates = np.flatnonzero((data["split"]==1)&(data["variant"]==0)&data["passed"])
    unit_designs = (data["designs"][candidates]-DESIGN_LOWER)/span
    margin = np.minimum(unit_designs,1-unit_designs).min(axis=1)
    interior = margin>=max(settings.gradient_unit_epsilons)
    boundary_ids = data["design_id"][candidates[~interior]].tolist()
    ids = candidates[interior]
    if design_ids is not None:
        ids = ids[np.isin(data["design_id"][ids],design_ids)]
    ids = ids[:settings.gradient_design_count]
    simulators = {}
    for dt in [configuration.simulation.timestep_s,configuration.simulation.timestep_s/2]:
        count = round(settings.horizon_s/dt)
        def simulate(d,dt=dt,count=count):
            p = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
            initial = solve_initial_pose(p).at[1].add(configuration.simulation.initial_clearance_um*1e-6)
            t,x = rollout_scan(initial,p,dt=dt,steps=count)
            return t,x,p
        simulators[dt] = jax.jit(jax.vmap(simulate))
    cases = []
    for index in ids:
        center = data["designs"][index]
        probes = [center]
        for epsilon in settings.gradient_unit_epsilons:
            for axis in range(2):
                for sign in [-1,1]:
                    probe = center.copy()
                    probe[axis] += sign*epsilon*span[axis]
                    probes.append(probe)
        probes = np.asarray(probes)
        if not np.all((probes>=DESIGN_LOWER)&(probes<=DESIGN_UPPER)):
            cases.append(dict(design_id=int(data["design_id"][index]),passed=False,reason="Finite-difference probes leave the design domain."))
            continue
        physical = []
        for dt,simulate in simulators.items():
            count = round(settings.horizon_s/dt)
            t,x,p = simulate(jnp.asarray(probes))
            t,x,p = np.asarray(t),np.asarray(x),np.asarray(p)
            com,_,_ = trajectory_observables(x,p)
            speed = (com[:,-1,0]-com[:,count//2,0])/(t[0,-1]-t[0,count//2])
            reports = [inspect_trajectory(tt,xx,pp,configuration.physical_limits) for tt,xx,pp in zip(t,x,p)]
            physical.append(dict(dt_s=dt,values_m_s=speed.tolist(),all_probes_physically_valid=all(r["passed"] for r in reports),
                                 physical_reports=reports))
        def objective(d):
            states = neural_rollout(model,d,settings.neural_timestep_s,macro_steps)
            com,_,_ = trajectory_observables(states,parameters_torch(d,base),xp=torch)
            return (com[:,-1,0]-com[:,macro_steps//2,0])/((macro_steps-macro_steps//2)*settings.neural_timestep_s)
        d32 = torch.tensor(center[None],dtype=torch.float32,device=device,requires_grad=True)
        value32 = objective(d32)
        derivative32, = torch.autograd.grad(value32.sum(),d32)
        gradient32 = derivative32.detach().cpu().numpy()[0]*span
        # Double precision resolves small FD shifts; also check the inference dtype.
        model.double()
        d = torch.tensor(center[None],dtype=torch.float64,device=device,requires_grad=True)
        value = objective(d)
        derivative, = torch.autograd.grad(value.sum(),d)
        gradient = derivative.detach().cpu().numpy()[0]*span
        with torch.no_grad():
            neural_values = objective(torch.tensor(probes,dtype=torch.float64,device=device)).cpu().numpy()
        model.float()
        precision_stable = bool(np.allclose(gradient,gradient32,rtol=.1,atol=1e-5))
        scales = []
        gradients = []
        for e,epsilon in enumerate(settings.gradient_unit_epsilons):
            begin = 1+4*e
            fd = [[(r["values_m_s"][begin+2*a+1]-r["values_m_s"][begin+2*a])/(2*epsilon) for a in range(2)] for r in physical]
            neural_fd = np.asarray([(neural_values[begin+2*a+1]-neural_values[begin+2*a])/(2*epsilon) for a in range(2)])
            fd = np.asarray(fd)
            denom = max(np.linalg.norm(fd[1]),1e-8)
            cosine = float(np.dot(gradient,fd[1])/max(np.linalg.norm(gradient)*np.linalg.norm(fd[1]),1e-16))
            stable = bool(np.linalg.norm(fd[0]-fd[1])/denom<=.25)
            agrees = bool(np.linalg.norm(gradient-fd[1])/denom<=.5 and cosine>=.8)
            neural_self_check = bool(np.allclose(gradient,neural_fd,rtol=.1,atol=1e-5))
            scales.append(dict(unit_epsilon=epsilon,physical_finite_difference_per_unit=fd.tolist(),
                neural_finite_difference_per_unit=neural_fd.tolist(),neural_autodiff_matches_fd=neural_self_check,
                physical_timestep_stable=stable,cosine=cosine,neural_matches_physical=agrees))
            gradients.append(fd[1])
        scale_stable = bool(np.linalg.norm(gradients[0]-gradients[-1])/max(np.linalg.norm(gradients[-1]),1e-8)<=.25)
        passed = (all(r["all_probes_physically_valid"] for r in physical) and scale_stable and precision_stable and
                  all(r["physical_timestep_stable"] and r["neural_matches_physical"] and r["neural_autodiff_matches_fd"] for r in scales))
        cases.append(dict(design_id=int(data["design_id"][index]),design=center.tolist(),
            neural_autodiff_per_unit=gradient.tolist(),neural_float32_autodiff_per_unit=gradient32.tolist(),
            neural_precision_stable=precision_stable,physical_scale_stable=scale_stable,
            checks=scales,physical=physical,passed=bool(passed)))
        print(f"Gradient validation design {data['design_id'][index]}: {passed}",flush=True)
        write_json(output/"gradient_validation.json",dict(cases=cases,status="in_progress"))
    report = dict(cases=cases,gradient_gate_passed=bool(len(cases)==settings.gradient_design_count and all(r["passed"] for r in cases)),
        horizon_s=settings.horizon_s,objective="Late-half mean COM velocity",gradient_units="Per normalized design coordinate (full design range)",
        excluded_boundary_design_ids=boundary_ids,
        scope="Interior validation designs allowing the requested central perturbations. Hard-contact finite differences checked across perturbation scales and two physical timesteps.")
    write_json(output/"gradient_validation.json",report)
    return report


def plot_comparison(data,prediction,ids,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for col,k in enumerate([0,len(ids)//2]):
        i = ids[k]
        for states,label in [(data["states"][i:i+1],"JAX"),(prediction[k:k+1],"NeRD")]:
            com,velocity,feet = trajectory_observables(states,data["parameters"][i:i+1])
            axes[0,col].plot(data["times"],1000*(com[0,:,0]-com[0,0,0]),label=label)
            axes[1,col].plot(data["times"],velocity[0,:,0],label=label)
            axes[2,col].plot(data["times"],1000*feet[0,:,0],label=label+" front")
            axes[2,col].plot(data["times"],1000*feet[0,:,1],alpha=.6,label=label+" hind")
        axes[0,col].set_title(f"Design {data['design_id'][i]}, initial-state variant {data['variant'][i]}")
        for row,label in enumerate(["COM displacement (mm)","COM velocity (m/s)","Foot height (mm)"]):
            axes[row,col].set_ylabel(label)
            axes[row,col].set_xlabel("Elapsed time (s)")
            axes[row,col].legend()
            axes[row,col].grid(alpha=.2)
        axes[2,col].axhline(0,color="black",lw=.7)
    fig.savefig(output/"comparison.png",dpi=150)
    plt.close(fig)


def export_animation(data,prediction,index,output,configuration):
    from walking_robot.visualization import render_trajectory
    output.mkdir(parents=True,exist_ok=True)
    # Both full trajectories remain visible, including violations and drift.
    for label,states in [("reference",data["states"][index]),("neural",prediction)]:
        target = output/label
        target.mkdir(parents=True,exist_ok=True)
        render_trajectory(data["times"],states,data["parameters"][index],target,configuration.physical_limits,
                          formats=("mp4",),max_frames=240,fps=30,playback_speed=.025,stop_at_violation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path("examples/learning/nerd_curriculum.yaml"))
    parser.add_argument("--nerd-root",type=Path,help="Required for neural training; unnecessary with --screen-only.")
    parser.add_argument("--output",type=Path,default=Path("outputs/nerd_curriculum"))
    parser.add_argument("--screen-only",action="store_true")
    parser.add_argument("--evaluate-only",action="store_true",help="Evaluate the selected checkpoint from this exact study.")
    parser.add_argument("--compare-checkpoint",type=Path,help="Report a compatible previous planar checkpoint on the same validation set.")
    args = parser.parse_args()
    if not args.screen_only and args.nerd_root is None:
        parser.error('--nerd-root is required unless --screen-only is selected.')
    settings,configuration = StudySettings.from_file(args.study)
    jax.config.update("jax_enable_x64",True)
    torch.set_num_threads(4)
    torch.manual_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch {device}; JAX {jax.devices()}",flush=True)
    data,metadata = generate_dataset(settings,configuration,args.output)
    print(f"Physical screening: {metadata['full_horizon_pass_count']}/{metadata['total_trajectories']}",flush=True)
    if args.screen_only:
        return
    if args.compare_checkpoint:
        previous = PlanarNeRD(args.nerd_root,base_design=configuration.design,
                             initial_clearance_um=configuration.simulation.initial_clearance_um).to(device).eval()
        previous.load_state_dict(torch.load(args.compare_checkpoint,map_location=device,weights_only=True))
        report,_ = evaluate(previous,data,np.flatnonzero(data["split"]==1),settings,configuration,device)
        report["checkpoint"] = str(args.compare_checkpoint.resolve())
        write_json(args.output/"previous_checkpoint_validation.json",report)
        del previous
    if args.evaluate_only:
        training = json.loads((args.output/"training.json").read_text())
        if training["settings"] != json.loads(json.dumps(asdict(settings))):
            raise ValueError("Checkpoint study settings differ.")
        model = PlanarNeRD(args.nerd_root,base_design=configuration.design,
                          initial_clearance_um=configuration.simulation.initial_clearance_um).to(device).eval()
        model.load_state_dict(torch.load(args.output/"model.pt",map_location=device,weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        model = train(settings,configuration,data,args.nerd_root,args.output,device)
    reports = {}
    for name,split in [("base",0),("validation",1),("test",2)]:
        ids = np.array([0]) if name=="base" else np.flatnonzero(data["split"]==split)
        report,prediction = evaluate(model,data,ids,settings,configuration,device)
        reports[name] = report
        write_json(args.output/f"{name}_validation.json",report)
        np.savez_compressed(args.output/f"{name}_rollouts.npz",ids=ids,x=prediction,t=data["times"])
        if name=="validation":
            plot_comparison(data,prediction,ids,args.output)
            export_animation(data,prediction[0],ids[0],args.output/"animation",configuration)
    gradients = check_gradients(model,data,settings,configuration,device,args.output)
    baseline = reports["base"]["cases"][0]
    base_passed = bool(baseline["reference_passed"] and baseline["neural_validation"]["passed"] and
        baseline["state_rmse"]<=.1 and baseline["speed_error_m_s"]<=settings.max_speed_error_m_s and
        baseline["foot_height_rmse_m"]<=settings.max_foot_height_rmse_m)
    result = dict(status="experimental",optimization_attempted=False,accepted_design=None,
        full_horizon_pass_count=metadata["full_horizon_pass_count"],total_trajectories=metadata["total_trajectories"],
        validation_trajectory_gate_passed=reports["validation"]["trajectory_gate_passed"],
        validation_gradient_gate_passed=gradients["gradient_gate_passed"],
        base_fidelity_gate_passed=base_passed,
        ready_for_optimization=bool(base_passed and reports["validation"]["trajectory_gate_passed"] and gradients["gradient_gate_passed"]),
        test_used_for_selection=False)
    write_json(args.output/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
