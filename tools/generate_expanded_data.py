"""Generate fixed-actuation three-design-variable trajectories for NeRD.

The supported baseline and the promising physical-screen design are test
anchors, never training targets. Numerical failures explicitly block collection;
finite physically inadmissible trajectories remain labeled negative examples.
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
import jax.numpy as jnp
import numpy as np
import yaml
from walking_robot import solve_initial_pose
from walking_robot.design import DesignRun
from walking_robot.expanded_design import ExpandedRobotDesign,sample_expanded_designs
from walking_robot.adaptive_jax import JaxAdaptiveDynamics
from walking_robot.physical_search import write_json


def load_training(path):
    s = yaml.safe_load(Path(path).read_text())
    space_file = Path(path).resolve().parent/s['design_space']
    space = yaml.safe_load(space_file.read_text())
    config = DesignRun.from_file(space_file.parent/space['base_design'])
    base = ExpandedRobotDesign(**dict(asdict(config.design),voltage_v=space['voltage_v']),drive_frequency_hz=space['drive_frequency_hz'])
    n = s['design_count']
    if type(n) is not int or n<24 or n%6:
        raise ValueError('Need at least 24 designs divisible by six.')
    designs = sample_expanded_designs(dict(sample_count=n,seed=s['seed'],bounds=s['bounds']),base)
    split = np.repeat([0,1,2],[n*2//3,n//6,n//6])
    # Freeze these anchors before training; the model never sees them in a loss.
    designs[0],designs[n*5//6] = designs[n*5//6].copy(),designs[0].copy()
    designs[n*5//6+1] = [2.6510431237373395,109.0817706014036,99.56825408934093]
    bounds = np.array(s['bounds'])
    if not np.all((designs>=bounds[:,0])&(designs<=bounds[:,1])):
        raise ValueError('Dataset anchors leave the configured training domain.')
    return s,config,base,designs,split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/expanded_nerd.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/expanded_nerd/data'))
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--collect',action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    s,c,base,designs,split = load_training(args.study)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    numerical = ['expanded_design','adaptive_jax','adaptive','contact_residual','parameters','dynamics','contact','_generated','geometry','initial_pose','validation']
    fp = json.loads(json.dumps(dict(settings={k:s[k] for k in ['seed','design_count','bounds','horizon_s','output_timestep_s','integration_timestep_s','reference_rtol']},
        base_design=asdict(base),configuration=asdict(c),jax_version=jax.__version__,backend=jax.default_backend(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in numerical})))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.collect:
        rows,states,p = [],[],[]
        for i in range(len(designs)):
            r = json.loads((args.output/f'{i:03d}.json').read_text())
            if r['fingerprint']!=fp or not r['solver']['success']:
                raise ValueError('Incomplete or mismatched reference data.')
            with np.load(args.output/f'{i:03d}.npz',allow_pickle=False) as f:
                states.append(f['states']);p.append(f['parameters']);times = f['times']
            rows.append(r)
        dataset = args.output/'dataset.npz'
        np.savez_compressed(dataset,designs=designs,split=split,states=np.array(states),parameters=np.array(p),times=times,
            passed=np.array([r['solver']['accepted_step_physical_validation']['passed'] for r in rows]))
        write_json(args.output/'report.json',dict(fingerprint=fp,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),rows=rows,
            policy='All complete finite trajectories; physical failures retain labels as negative constraint examples. Disjoint design splits.'))
        print('Collected',len(rows),'trajectories.',flush=True);return
    if not 0<=args.worker<args.workers:
        raise ValueError('Invalid worker partition.')
    make = jax.jit(lambda d:base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1],hind_angle_deg=d[2]))
    initial = jax.jit(solve_initial_pose)
    solver = JaxAdaptiveDynamics('smooth')
    for i in range(args.worker,len(designs),args.workers):
        path = args.output/f'{i:03d}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['fingerprint']!=fp:raise ValueError('Source/protocol mismatch; use a fresh directory.')
            if old['solver']['success']:continue
        print(f'Expanded teacher {i}, split {split[i]}: {designs[i].tolist()}',flush=True)
        p = make(jnp.array(designs[i]));x = initial(p).at[1].add(c.simulation.initial_clearance_um*1e-6)
        info,result = solver.integrate(x,p,s['horizon_s'],rtol=s['reference_rtol'],sample_dt=s['integration_timestep_s'],physical_limits=c.physical_limits)
        if result is not None:
            stride = round(s['output_timestep_s']/s['integration_timestep_s'])
            np.savez_compressed(args.output/f'{i:03d}.npz',times=result['times'][::stride],states=result['states'][::stride],parameters=np.asarray(p))
        write_json(path,dict(fingerprint=fp,id=i,design=designs[i].tolist(),split=int(split[i]),solver=info))
        print(f"Completed {i}: numerical={info['success']}, physical={info.get('accepted_step_physical_validation',{}).get('passed')}",flush=True)


if __name__=='__main__':main()
