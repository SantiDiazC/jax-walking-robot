"""Explore target-speed gradient losses over the existing two-design domain.

Projected Adam and a grid of initial candidates explore the neural objective.
The best neural proposal is independently screened with original contact at
two resolutions. This is exploratory: neither global feasibility nor design
acceptance follows from the neural search, and discrete rounding is retained.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import numpy as np
import torch
import yaml
from generate_smooth_training_data import load
from walking_robot import solve_initial_pose,rollout_scan
from walking_robot.learning import parameters_torch
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.learning_study import trajectory_observables
from walking_robot.physical_search import step_count,write_json
from walking_robot.trajectory_nerd import TrajectoryNeRD,predict_trajectory
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/learning/nerd_target_speed.yaml'))
    parser.add_argument('--training',type=Path,default=Path('outputs/nerd_trajectory/consistent'))
    parser.add_argument('--nerd-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_target_speed'))
    args = parser.parse_args()
    protocol = yaml.safe_load(args.config.read_text())
    s,c,_,_ = load(args.config.resolve().parent/protocol['training_study'])
    if min(protocol['targets_m_s'])<=0 or not np.isfinite(protocol['targets_m_s']).all():
        raise ValueError('Targets must be positive and finite.')
    for name in ['grid_count','starts','iterations']:
        if type(protocol[name]) is not int or protocol[name]<2:
            raise ValueError(f'{name} must be an integer of at least two.')
    if not np.isfinite(protocol['learning_rate']) or protocol['learning_rate']<=0:
        raise ValueError('learning_rate must be positive.')
    if (args.output/'report.json').exists():
        raise ValueError('Use a fresh output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    raw = (args.training/'report.json').read_bytes()
    previous = json.loads(raw)
    checkpoint = args.training/'model.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=previous['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch.')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['settings']!=s or saved['configuration']!=asdict(c):
        raise ValueError('Checkpoint configuration mismatch.')
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    for n,h in saved['provenance']['adapter_sources'].items():
        if hashlib.sha256((source/n).read_bytes()).hexdigest()!=h:
            raise ValueError('Neural source mismatch.')
    torch.set_num_threads(2)
    torch.manual_seed(protocol['seed'])
    model = TrajectoryNeRD(args.nerd_root,c.design,horizon=s['horizon_s'],width=s['width'],layers=s['layers'],
        clearance_um=c.simulation.initial_clearance_um,consistent_velocity=s.get('consistent_velocity',False)).double().eval()
    if model.upstream['sha256']!=saved['upstream']['sha256']:
        raise ValueError('Upstream source mismatch.')
    model.load_state_dict(saved['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    horizon = protocol['horizon_s']
    if horizon not in [1.,s['horizon_s']]:
        raise ValueError('Use a previously assessed long-horizon objective.')
    def speed(unit):
        t = unit.new_tensor([horizon/2,horizon])
        x = predict_trajectory(model,unit,t)
        p = parameters_torch(model.physical_designs(unit),model.base_design)
        com,_,_ = trajectory_observables(x,p,xp=torch)
        return (com[:,-1,0]-com[:,0,0])/(horizon/2)
    axes = np.linspace(0,1,protocol['grid_count'])
    grid = torch.tensor(np.array(np.meshgrid(axes,axes)).reshape(2,-1).T,dtype=torch.float64)
    with torch.no_grad():
        values = torch.cat([speed(grid[k:k+512]) for k in range(0,len(grid),512)])
    top = torch.argsort(values,descending=True)
    best_unit,best_speed = grid[top[0]].numpy(),float(values[top[0]])
    print(f'Neural grid range: {float(values.min()):.6f} to {float(values.max()):.6f} m/s',flush=True)
    experiments = []
    for target in protocol['targets_m_s']:
        # Include the supported baseline, promising grid points and random starts.
        count = protocol['starts']
        u = torch.rand(count,2,dtype=torch.float64)
        u[:count//2] = grid[top[:count//2]]
        u[-1] = u.new_tensor([(1.-.5)/1.5,.5])
        u.requires_grad_(True)
        optimizer = torch.optim.Adam([u],lr=protocol['learning_rate'])
        history = []
        for i in range(protocol['iterations']):
            velocity = speed(u)
            loss = (velocity-target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural objective.')
            k = int(torch.argmax(velocity))
            if float(velocity[k].detach())>best_speed:
                best_speed = float(velocity[k].detach());best_unit = u[k].detach().numpy().copy()
            history.append(dict(iteration=i,best_speed_m_s=float(velocity.max().detach()),loss=float(loss.detach())))
            optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step()
            # Projection applies only to design bounds; no states are repaired.
            with torch.no_grad():
                u.clamp_(0,1)
        with torch.no_grad():
            velocity = speed(u)
        experiments.append(dict(target_m_s=target,history=history,final_speeds_m_s=velocity.tolist(),
            final_designs=model.physical_designs(u).detach().numpy().tolist()))
        print(f'Target {target:g}: best neural candidate so far {best_speed:.6f} m/s',flush=True)
    best_design = np.array(DESIGN_LOWER)+best_unit*(np.array(DESIGN_UPPER)-DESIGN_LOWER)
    jax.config.update('jax_enable_x64',True)
    p = c.design.parameters(adhesive_thickness_um=best_design[0],front_angle_deg=best_design[1])
    initial = jax.jit(solve_initial_pose)(p).at[1].add(c.simulation.initial_clearance_um*1e-6)
    checks = []
    for dt in protocol['verification_timesteps_s']:
        count = step_count(horizon,dt)
        print(f'Original contact: candidate={best_design.tolist()}, dt={dt:g}',flush=True)
        run = jax.jit(lambda x,p:rollout_scan(x,p,dt,count))
        times,states = jax.tree.map(np.asarray,run(initial,p))
        check = inspect_trajectory(times,states,np.asarray(p),c.physical_limits)
        checks.append(dict(timestep_s=dt,validation=check))
    physical = checks[-1]['validation']['metrics']['late_mean_com_speed_m_s']
    stable = bool(all(r['validation']['passed'] for r in checks) and len(checks)>=2 and
        abs(checks[0]['validation']['metrics']['late_mean_com_speed_m_s']-physical)<=.001)
    report = dict(protocol=protocol,method='Projected Adam of squared target-speed loss; grid and multiple initial designs.',
        neural_grid_speed_range_m_s=[float(values.min()),float(values.max())],experiments=experiments,
        best_neural_design=best_design.tolist(),best_neural_speed_m_s=best_speed,
        physical_checks=checks,physically_admissible_and_resolution_stable=stable,
        measured_speed_m_s=physical,neural_speed_error_m_s=float(abs(best_speed-physical)),
        targets=[dict(target_m_s=t,met_by_screened_candidate=bool(stable and physical>=t)) for t in protocol['targets_m_s']],
        accepted_design=None,global_optimum_or_infeasibility_proven=False,
        scope='Exploratory proposals over the original two-parameter domain with fixed voltage. One candidate screened canonically; no robust final design acceptance.',
        provenance=dict(training_report_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=previous['checkpoint_sha256'],
            torch_version=torch.__version__,jax_version=jax.__version__,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            numerical_sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in
                               ['dynamics','contact','_generated','geometry','parameters','initial_pose','integrators','validation']}))
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(candidate=best_design.tolist(),neural_speed_m_s=best_speed,physical_speed_m_s=physical,
                         physical_passed=stable,targets=report['targets'])),flush=True)


if __name__=='__main__':
    main()
