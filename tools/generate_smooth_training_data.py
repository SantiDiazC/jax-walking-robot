"""Generate disjoint-design, canonical trajectories and variational targets.

Independent --worker partitions write independent files. --collect verifies
all fingerprints and complete trajectories before creating the training data.
No neural code runs here; the compiled adaptive teacher always uses float64.
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
from walking_robot.adaptive_jax import JaxAdaptiveDynamics
from walking_robot.contact_residual import ContactSmoothing
from walking_robot.design import DesignRun
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.physical_search import write_json


def load(path):
    settings = yaml.safe_load(Path(path).read_text())
    config = DesignRun.from_file(Path(path).resolve().parent/settings['design'])
    count = settings['design_count']
    if type(count) is not int or count<18 or count%6:
        raise ValueError('Use at least 18 designs, divisible by six.')
    for name in ['horizon_s','output_timestep_s','integration_timestep_s','reference_rtol',
                 'gap_width_um','velocity_width_m_s','force_width_uN']:
        if not np.isfinite(settings[name]) or settings[name]<=0:
            raise ValueError(f'{name} must be positive and finite.')
    if not np.isclose(settings['output_timestep_s']/settings['integration_timestep_s'],
                      round(settings['output_timestep_s']/settings['integration_timestep_s'])):
        raise ValueError('Output timestep must be an integer multiple of integration timestep.')
    rng = np.random.default_rng(settings['seed'])
    designs = rng.uniform(DESIGN_LOWER,DESIGN_UPPER,(count,2))
    # These anchors are held out from every training loss and normalization.
    designs[count*2//3] = [1.,95.]
    designs[count*5//6] = [1.375,96.66666666666667]
    designs[count*5//6+1] = [1.5237432559624633,91.3975248360931]
    split = np.repeat([0,1,2],[count*2//3,count//6,count//6])
    return settings,config,designs,split


def fingerprint(settings,config):
    numerical = ['adaptive_jax','contact_residual','contact','dynamics','_generated',
                 'geometry','parameters','initial_pose','design','validation','learning_spec']
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    data_keys = ['design_count','seed','horizon_s','output_timestep_s','integration_timestep_s',
                 'reference_rtol','gap_width_um','velocity_width_m_s','force_width_uN']
    return json.loads(json.dumps(dict(settings={k:settings[k] for k in data_keys},
        configuration=asdict(config),jax_version=jax.__version__,backend=jax.default_backend(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in numerical})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/nerd_trajectory.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_trajectory/data'))
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--collect',action='store_true')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    settings,config,designs,split = load(args.study)
    if not 0<=args.worker<args.workers:
        raise ValueError('Invalid worker partition.')
    args.output.mkdir(parents=True,exist_ok=True)
    fp = fingerprint(settings,config)
    if args.collect:
        rows,states,sensitivities,parameters = [],[],[],[]
        for i in range(len(designs)):
            row = json.loads((args.output/f'{i:03d}.json').read_text())
            if row['fingerprint']!=fp or row['design']!=designs[i].tolist():
                raise ValueError(f'Data provenance mismatch: design {i}.')
            if not row['solver']['success']:
                raise ValueError(f'Incomplete teacher trajectory {i}.')
            with np.load(args.output/f'{i:03d}.npz',allow_pickle=False) as data:
                states.append(data['states']);sensitivities.append(data['sensitivities'])
                parameters.append(data['parameters']);times = data['times']
            rows.append(row)
        dataset = args.output/'dataset.npz'
        np.savez_compressed(dataset,designs=designs,split=split,states=np.array(states),
            sensitivities=np.array(sensitivities),parameters=np.array(parameters),times=times,
            passed=np.array([r['solver']['accepted_step_physical_validation']['passed'] for r in rows]))
        write_json(args.output/'report.json',dict(fingerprint=fp,rows=rows,
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            split_policy='Disjoint designs, 2/3 train, 1/6 validation, 1/6 test; canonical supported start only.',
            gradient_coordinates='Normalized full design ranges; rounded physical parameters have zero local derivative.'))
        print(f'Collected {len(rows)} trajectories.',flush=True)
        return
    span = jnp.array(DESIGN_UPPER)-jnp.array(DESIGN_LOWER)
    def prepare(unit):
        d = jnp.array(DESIGN_LOWER)+unit*span
        p = config.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        x = solve_initial_pose(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
        return p,x
    prepare = jax.jit(prepare)
    derivative = jax.jit(jax.jacfwd(prepare))
    solver = JaxAdaptiveDynamics('smooth',ContactSmoothing(settings['gap_width_um']*1e-6,
        settings['velocity_width_m_s'],settings['force_width_uN']*1e-6))
    for i in range(args.worker,len(designs),args.workers):
        path = args.output/f'{i:03d}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row['fingerprint']!=fp or row['design']!=designs[i].tolist():
                raise ValueError('Existing data differs; choose a fresh output directory.')
            if row['solver']['success'] and (args.output/f'{i:03d}.npz').exists():
                print(f'Design {i}: reusing verified source/protocol match.',flush=True)
                continue
        unit = (jnp.array(designs[i])-jnp.array(DESIGN_LOWER))/span
        p,x = prepare(unit);dp,dx = derivative(unit)
        print(f'Design {i}, split {split[i]}: {designs[i].tolist()}',flush=True)
        info,result = solver.integrate(x,p,settings['horizon_s'],rtol=settings['reference_rtol'],
            sample_dt=settings['integration_timestep_s'],max_step=2e-4,max_rhs_calls=12000000,
            initial_sensitivity=dx,parameter_sensitivity=dp,physical_limits=config.physical_limits)
        if result is not None:
            stride = round(settings['output_timestep_s']/settings['integration_timestep_s'])
            np.savez_compressed(args.output/f'{i:03d}.npz',times=result['times'][::stride],
                states=result['states'][::stride],sensitivities=result['sensitivities'][::stride],parameters=np.asarray(p))
        write_json(path,dict(fingerprint=fp,design=designs[i].tolist(),split=int(split[i]),solver=info))
        print(f"Completed {i}: success={info['success']}, physical={info.get('accepted_step_physical_validation',{}).get('passed')}",flush=True)


if __name__=='__main__':
    main()
