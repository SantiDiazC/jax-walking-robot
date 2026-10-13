"""Generate JAX data, train a design-conditioned NeRD and gate optimization.

Example: python tools/nerd_design_experiment.py --nerd-root ../neural-robot-dynamics
No candidate is accepted from neural predictions alone. Reports distinguish
physical failures, surrogate error and proposals confirmed by the JAX solver.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

from walking_robot import calculate_P_from_meta_jax, solve_initial_pose, rollout_scan, rollout_batched
from walking_robot.geometry import points
from walking_robot.design import DesignRun
from walking_robot.workflow import run_design
from walking_robot.validation import PhysicalLimits, inspect_trajectory, assess_surrogate, assess_candidate
from walking_robot.learning import (DESIGN_LOWER, DESIGN_UPPER, STATE_SCALES, PlanarNeRD,
                                    parameters_torch, neural_rollout, design_objective)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def generate(args, limits):
    cache = args.output/"dataset.npz"
    metadata_path = args.output/"dataset_report.json"
    config = {"seed":args.seed,"design_count":args.design_count,"horizon":args.horizon,
              "dt":args.dt,"substeps":args.substeps,"limits":asdict(limits),
              "bounds":[DESIGN_LOWER,DESIGN_UPPER]}
    config["base_configuration"] = asdict(args.configuration)
    source_root = Path(__file__).resolve().parents[1]/"src/walking_robot"
    config["source_sha256"] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(source_root.glob("*.py"))}
    # JSON round trip gives stable tuple/list representation for cache checks.
    config = json.loads(json.dumps(config))
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["configuration"] != config:
            raise ValueError("Cached dataset settings differ; choose a new output directory.")
        with np.load(cache, allow_pickle=False) as saved:
            return {key:saved[key] for key in saved.files},metadata
    rng = np.random.default_rng(args.seed)
    designs = rng.uniform(DESIGN_LOWER,DESIGN_UPPER,(args.design_count,2))
    base = args.configuration.design
    designs[0] = [base.adhesive_thickness_um,base.front_angle_deg]
    make = jax.jit(jax.vmap(lambda d: base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])))
    parameters = make(jnp.asarray(designs))
    initial = jax.jit(jax.vmap(solve_initial_pose))(parameters)
    initial = initial.at[:,1].add(args.configuration.simulation.initial_clearance_um*1e-6)
    steps = round(args.horizon/args.dt)*args.substeps
    run = jax.jit(lambda x,p: rollout_batched(x,p,dt=args.dt/args.substeps,steps=steps))
    print(f"Generating {len(designs)} designs, {steps} physical substeps each...",flush=True)
    start = time.perf_counter()
    times, states = run(initial,parameters)
    states.block_until_ready()
    times, states, parameters = np.asarray(times),np.asarray(states),np.asarray(parameters)
    reports = [inspect_trajectory(times,x,p,limits) for x,p in zip(states,parameters)]
    prefix = []
    for r in reports:
        bad = r["first_violation_index"]
        # A macro interval touching a failed substep is excluded from training.
        prefix.append((bad-1)//args.substeps+1 if bad is not None else len(times[::args.substeps]))
    split = np.zeros(len(designs),dtype=np.int64)
    split[int(len(designs)*2/3):int(len(designs)*5/6)] = 1
    split[int(len(designs)*5/6):] = 2
    dataset = dict(designs=designs,parameters=parameters,times=times[::args.substeps],
                   states=states[:,::args.substeps,:],valid_prefix=np.asarray(prefix),split=split)
    metadata = {"configuration":config,"generation_s":time.perf_counter()-start,
                "full_horizon_pass_count":sum(r["passed"] for r in reports),
                "training_policy":"Only intervals before the first physical violation; full trajectories retained for diagnosis.",
                "split_policy":"Disjoint design IDs: train=0, validation=1, test=2. No windows shared across designs.",
                "designs":[dict(design=d.tolist(),split=int(s),validation=r) for d,s,r in zip(designs,split,reports)]}
    np.savez_compressed(cache,**dataset)
    write_json(metadata_path,metadata)
    print(f"Physical screening: {metadata['full_horizon_pass_count']}/{len(designs)} designs pass the full horizon.",flush=True)
    return dataset,metadata


def fit(args, data, device):
    model = PlanarNeRD(args.nerd_root,history=8,base_design=args.configuration.design,
                       initial_clearance_um=args.configuration.simulation.initial_clearance_um).to(device)
    x = torch.tensor(data["states"],dtype=torch.float32,device=device)
    d = torch.tensor(data["designs"],dtype=torch.float32,device=device)
    valid_ids = {s:np.flatnonzero((data["split"]==s)&(data["valid_prefix"]>=model.history+args.rollout_steps+1)) for s in [0,1,2]}
    if any(len(ids)<2 for ids in valid_ids.values()):
        raise ValueError("Not enough physically admissible prefixes in each design split to train and validate.")
    deltas = np.concatenate([np.diff(data["states"][i,:data["valid_prefix"][i],6:],axis=0) for i in valid_ids[0]])
    floor = np.array([1e-5,1e-5,.01,.01,.01,.01,1e-12,1e-12])
    with torch.no_grad():
        model.delta_mean.copy_(torch.as_tensor(deltas.mean(axis=0),device=device,dtype=torch.float32))
        model.delta_scale.copy_(torch.as_tensor(np.maximum(deltas.std(axis=0),floor),device=device,dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    if args.refine_from:
        source_report = args.refine_from.parent/"dataset_report.json"
        current = json.loads((args.output/"dataset_report.json").read_text())["configuration"]
        previous = json.loads(source_report.read_text())["configuration"]
        if current != previous:
            raise ValueError("Refinement checkpoint must use exactly the same dataset configuration and source hashes.")
        model.load_state_dict(torch.load(args.refine_from,map_location=device,weights_only=True))
    rng = np.random.default_rng(args.seed+1)
    qscale = x.new_tensor(STATE_SCALES[:6])

    def batch(split,batch_size,k):
        ids = rng.choice(valid_ids[split],batch_size)
        positions = np.array([rng.integers(0,int(data["valid_prefix"][i])-k) for i in ids])
        ids = torch.as_tensor(ids,device=device)
        positions = torch.as_tensor(positions,device=device)
        indices = torch.clamp(positions[:,None]+torch.arange(-model.history+1,1,device=device),min=0)
        history = x[ids[:,None],indices]
        times = indices.to(x.dtype)*args.dt
        targets = x[ids[:,None],positions[:,None]+torch.arange(1,k+1,device=device)]
        return history,times,d[ids],targets,positions

    def loss_for_batch(split,k):
        history,times,designs,targets,positions = batch(split,128,k)
        loss = 0.
        for step in range(k):
            predicted = model.predict_next(history,times,designs,args.dt)
            error = (predicted[:,6:]-targets[:,step,6:])/model.delta_scale
            loss = loss+.8**step*(error.square().mean()+.1*((predicted[:,:6]-targets[:,step,:6])/qscale).square().mean())
            history = torch.cat([history[:,1:],predicted[:,None]],dim=1)
            next_time = (positions.to(x.dtype)+step+1)*args.dt
            times = torch.cat([times[:,1:],next_time[:,None]],dim=1)
        return loss/k

    log,best = [],float("inf")
    checkpoint = args.output/"model.pt"
    start = time.perf_counter()
    def save_if_improved(record):
        nonlocal best
        model.eval()
        ids = valid_ids[1]
        with torch.no_grad():
            predicted = neural_rollout(model,d[ids],args.dt,len(data["times"])-1)
            score = float(torch.sqrt(((predicted-x[ids])/x.new_tensor(STATE_SCALES)).square().mean()))
        record["validation_free_rollout_rmse"] = score if np.isfinite(score) else None
        if np.isfinite(score) and score < best:
            best = score
            torch.save(model.state_dict(),checkpoint)
    for iteration in range(args.train_steps):
        model.train()
        k = 1 if iteration < args.train_steps//3 else (4 if iteration < 2*args.train_steps//3 else args.rollout_steps)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(0,k)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; experiment stopped.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if iteration%100==0 or iteration==args.train_steps-1:
            model.eval()
            with torch.no_grad():
                validation = float(torch.stack([loss_for_batch(1,1) for _ in range(4)]).mean())
            record = dict(iteration=iteration,k_rollout=k,training_loss=float(loss.detach()),validation_one_step_loss=validation)
            if iteration%300==0 or iteration==args.train_steps-1:
                # Select on complete held-out trajectories, rather than a
                # small teacher-forced loss that can hide accumulated drift.
                save_if_improved(record)
            log.append(record)
            print(json.dumps(record),flush=True)
    if args.refine_steps:
        if np.isfinite(best):
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        record = dict(phase="before_full_rollout_refinement")
        save_if_improved(record)
        log.append(record)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
        full_ids = np.flatnonzero((data["split"]==0)&(data["valid_prefix"]==len(data["times"])))
        if len(full_ids)<2:
            raise ValueError("Full-rollout refinement needs at least two fully admissible training designs.")
        total_steps = len(data["times"])-1
        halfway = total_steps//2
        scale = x.new_tensor(STATE_SCALES)
        def com_x(states, parameters):
            centers = points(states,parameters[:,None,:],xp=torch)["centers"][...,0]
            return (centers*parameters[:,None,:4]).sum(-1)/parameters[:,:4].sum(-1)[:,None]
        for iteration in range(args.refine_steps):
            model.train()
            ids = rng.choice(full_ids,4,replace=True)
            ids = torch.as_tensor(ids,device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = neural_rollout(model,d[ids],args.dt,total_steps)
            parameters = parameters_torch(d[ids],args.configuration.design)
            pc,tc = com_x(predicted,parameters),com_x(x[ids],parameters)
            speed_error = ((pc[:,-1]-pc[:,halfway])-(tc[:,-1]-tc[:,halfway]))/((total_steps-halfway)*args.dt)
            loss = ((predicted-x[ids])/scale).square().mean()+((pc-tc)/.001).square().mean()+(speed_error/.05).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite full-rollout refinement loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if iteration%20==0 or iteration==args.refine_steps-1:
                record = dict(phase="full_rollout_refinement",iteration=iteration,training_loss=float(loss.detach()),
                              training_late_mean_speed_rmse_m_s=float(speed_error.detach().square().mean().sqrt()))
                save_if_improved(record)
                log.append(record)
                print(json.dumps(record),flush=True)
    if not np.isfinite(best):
        raise RuntimeError("No checkpoint had a finite held-out free rollout.")
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    write_json(args.output/"training.json",dict(log=log,elapsed_s=time.perf_counter()-start,
               best_validation_free_rollout_rmse=best,checkpoint_selection="Full validation free-rollout normalized state RMSE; test split excluded.",upstream=model.upstream,
               seed=args.seed,train_steps=args.train_steps,rollout_steps=args.rollout_steps,
               refine_steps=args.refine_steps,refinement_source_sha256=hashlib.sha256(args.refine_from.read_bytes()).hexdigest() if args.refine_from else None,
               base_configuration=asdict(args.configuration),
               torch_version=torch.__version__,jax_version=jax.__version__,device=str(device)))
    return model


def evaluate(args, model, data, metadata, limits, device):
    output = []
    for split in [1,2]:
        ids = np.flatnonzero(data["split"]==split)
        designs = torch.as_tensor(data["designs"][ids],dtype=torch.float32,device=device)
        with torch.no_grad():
            predicted = neural_rollout(model,designs,args.dt,len(data["times"])-1).cpu().numpy()
        for index,states in zip(ids,predicted):
            reference = data["states"][index]
            physical = inspect_trajectory(data["times"],states,data["parameters"][index],limits)
            ref_report = metadata["designs"][index]["validation"]
            finite = np.isfinite(states).all() and np.isfinite(reference).all()
            rmse = float(np.sqrt(np.mean(((states-reference)/np.asarray(STATE_SCALES))**2))) if finite else None
            rmse = rmse if rmse is not None and np.isfinite(rmse) else None
            ref_speed = ref_report["metrics"].get("late_mean_com_speed_m_s")
            speed = physical["metrics"].get("late_mean_com_speed_m_s")
            speed_error = abs(speed-ref_speed) if speed is not None and ref_speed is not None else None
            record = dict(id=int(index),split="validation" if split==1 else "test",design=data["designs"][index].tolist(),
                          normalized_state_rmse=rmse,late_mean_speed_error_m_s=speed_error,
                          reference_physical_validation=ref_report,neural_physical_validation=physical)
            output.append(record)
            np.savez_compressed(args.output/f"neural_design_{index}.npz",t=data["times"],x=states,P=data["parameters"][index])
    report = dict(cases=output,**assess_surrogate(output))
    write_json(args.output/"surrogate_validation.json",report)
    return report


def optimize(args, model, data, metadata, limits, device):
    base = args.configuration.design
    baseline = np.array([base.adhesive_thickness_um,base.front_angle_deg])
    design = torch.tensor([baseline.tolist()],device=device,requires_grad=True)
    lo,hi = design.new_tensor(DESIGN_LOWER),design.new_tensor(DESIGN_UPPER)
    unit = torch.zeros((1,2),device=device,requires_grad=True)
    # Baseline normalized into the same bounded design domain.
    with torch.no_grad():
        unit.copy_(2*(design.detach()-lo)/(hi-lo)-1)
    optimizer = torch.optim.Adam([unit],lr=.05)
    trace = []
    steps = len(data["times"])-1
    for iteration in range(args.optimize_steps):
        optimizer.zero_grad(set_to_none=True)
        candidate = lo+(unit+1)/2*(hi-lo)
        loss = design_objective(model,candidate,args.dt,steps,limits=limits).mean()
        if not torch.isfinite(loss):
            return dict(accepted=False,reason="nonfinite_neural_objective",trace=trace)
        loss.backward()
        if unit.grad is None or not torch.isfinite(unit.grad).all():
            return dict(accepted=False,reason="nonfinite_design_gradient",trace=trace)
        trace.append(dict(iteration=iteration,design=candidate.detach().cpu().numpy()[0].tolist(),objective=float(loss.detach())))
        optimizer.step()
        with torch.no_grad():
            unit.clamp_(-1,1)
    candidate = (lo+(unit.detach()+1)/2*(hi-lo)).cpu().numpy()[0]
    reference = metadata["designs"][0]["validation"]
    checks=[]
    for subdt in [args.dt/args.substeps,args.dt/(2*args.substeps)]:
        for label,dd in [("baseline",baseline),("candidate",candidate)]:
            p = base.parameters(adhesive_thickness_um=float(dd[0]),front_angle_deg=float(dd[1]))
            x0 = jax.jit(solve_initial_pose)(p)
            x0 = x0.at[1].add(args.configuration.simulation.initial_clearance_um*1e-6)
            # Extend beyond training to expose delayed physical failures.
            horizon = max(.2,args.horizon)
            count = round(horizon/subdt)
            t,x = jax.jit(lambda x,p: rollout_scan(x,p,dt=subdt,steps=count))(x0,p)
            x.block_until_ready()
            report = inspect_trajectory(t,x,p,limits)
            checks.append(dict(label=label,dt_s=subdt,validation=report))
            np.savez_compressed(args.output/f"optimized_{label}_{subdt:g}.npz",t=t,x=x,P=p)
            if label == "candidate":
                configuration = replace(args.configuration,design=replace(base,adhesive_thickness_um=float(dd[0]),front_angle_deg=float(dd[1])),
                                        simulation=replace(args.configuration.simulation,duration_s=horizon,timestep_s=subdt))
                (args.output/"proposed_design.yaml").write_text(yaml.safe_dump(asdict(configuration),sort_keys=False))
    accepted = assess_candidate(checks)
    if accepted:
        # The finer-step physical check becomes the reproducible final design.
        (args.output/"accepted_design.yaml").write_text(yaml.safe_dump(asdict(configuration),sort_keys=False))
        run_design(configuration,args.output/"accepted_design")
    return dict(accepted=bool(accepted),candidate_design=candidate.tolist(),trace=trace,physical_checks=checks,
                baseline_training_horizon_passed=reference["passed"],
                reason="confirmed_by_physical_solver" if accepted else "physical_constraints_or_improvement_not_confirmed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nerd-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("outputs/nerd_pilot"))
    parser.add_argument("--design",type=Path,help="Base design, initial clearance and physical limits in YAML.")
    parser.add_argument("--limits",type=Path)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--design-count",type=int,default=36)
    parser.add_argument("--horizon",type=float,default=.05)
    parser.add_argument("--dt",type=float,default=1e-4,help="Neural interval; physical solver uses dt/substeps.")
    parser.add_argument("--substeps",type=int,help="Defaults to dt divided by the YAML physical timestep.")
    parser.add_argument("--train-steps",type=int,default=1200)
    parser.add_argument("--refine-steps",type=int,default=0,help="Optional full free-rollout supervision with COM displacement/speed losses.")
    parser.add_argument("--refine-from",type=Path,help="Checkpoint with matching dataset settings and source hashes.")
    parser.add_argument("--optimize-steps",type=int,default=20)
    parser.add_argument("--rollout-steps",type=int,default=16,help="Final multistep training window.")
    parser.add_argument("--screen-only",action="store_true",help="Generate and screen designs without training.")
    args = parser.parse_args()
    args.configuration = DesignRun.from_file(args.design) if args.design else DesignRun()
    if args.substeps is None:
        ratio = args.dt/args.configuration.simulation.timestep_s
        args.substeps = round(ratio)
        if not np.isclose(args.substeps,ratio,atol=1e-10,rtol=0):
            parser.error("Neural dt must be an integer multiple of the YAML physical timestep.")
    if (args.design_count<18 or args.train_steps<0 or args.refine_steps<0 or
            (args.train_steps<2 and not (args.refine_from and args.refine_steps>0)) or
            args.dt<=0 or args.horizon<=0 or args.substeps<1 or args.optimize_steps<1):
        parser.error("Need >=18 designs, >=2 training steps (or a refinement checkpoint) and positive timestep/horizon/iteration settings.")
    if args.rollout_steps < 4 or args.rollout_steps >= round(args.horizon/args.dt):
        parser.error("Rollout window must be >=4 and shorter than the simulation.")
    base = args.configuration.design
    if not np.all((np.array([base.adhesive_thickness_um,base.front_angle_deg])>=DESIGN_LOWER)&
                  (np.array([base.adhesive_thickness_um,base.front_angle_deg])<=DESIGN_UPPER)):
        parser.error("Base thickness/angle must lie inside the experimental design bounds.")
    if not np.isclose(round(args.horizon/args.dt)*args.dt,args.horizon,atol=1e-12,rtol=0):
        parser.error("Horizon must be an integer multiple of dt.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    jax.config.update("jax_enable_x64",True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limits = PhysicalLimits(**yaml.safe_load(args.limits.read_text())) if args.limits else args.configuration.physical_limits
    args.configuration = replace(args.configuration,physical_limits=limits)
    args.output.mkdir(parents=True,exist_ok=True)
    print(f"Torch: {device}; JAX: {jax.devices()}",flush=True)
    data,metadata = generate(args,limits)
    if args.screen_only:
        write_json(args.output/"result.json",dict(status="physical_screening_only",optimization_attempted=False,
                   accepted_design=None,full_horizon_pass_count=metadata["full_horizon_pass_count"]))
        return
    model = fit(args,data,device)
    validation = evaluate(args,model,data,metadata,limits,device)
    result = dict(status="experimental",surrogate_validation_passed=validation["validation_gate_passed"],
                  optimization_attempted=False,accepted_design=None)
    if validation["validation_gate_passed"]:
        result["optimization_attempted"] = True
        result["optimization"] = optimize(args,model,data,metadata,limits,device)
        if result["optimization"]["accepted"]:
            result["accepted_design"] = result["optimization"]["candidate_design"]
    else:
        result["reason"] = "surrogate_validation_or_physical_training_domain_failed"
    write_json(args.output/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
