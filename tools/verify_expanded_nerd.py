"""Independently check frozen NeRD proposals with original hard-contact dynamics.

This canonical two-resolution check is exploratory: it does not replace the
separate perturbed-start acceptance protocol or validate a global neural model.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from generate_expanded_data import load_training
from walking_robot import solve_initial_pose,dynamics_step
from walking_robot.physical_search import step_count,paired_acceptance,write_json
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural',type=Path,default=Path('outputs/expanded_nerd/training'))
    parser.add_argument('--study',type=Path,default=Path('examples/learning/expanded_nerd.yaml'))
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--collect',action='store_true')
    args = parser.parse_args();jax.config.update('jax_enable_x64',True)
    if not 0<=args.worker<args.workers:raise ValueError('Invalid worker partition.')
    settings,config,base,_,_ = load_training(args.study)
    space = yaml.safe_load((args.study.resolve().parent/settings['design_space']).read_text())
    neural = json.loads((args.neural/'report.json').read_text())
    designs = [[base.adhesive_thickness_um,base.front_angle_deg,base.hind_angle_deg]]+[r['design'] for r in neural['proposals']]
    bounds = np.array(settings['bounds'])
    if not np.all((np.array(designs)>=bounds[:,0])&(np.array(designs)<=bounds[:,1])):raise ValueError('Proposal outside trained domain.')
    output = args.neural/'physical_verification';output.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    fp = dict(neural_report_sha256=hashlib.sha256((args.neural/'report.json').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),settings=settings,base=asdict(base),
        configuration=asdict(config),jax_version=jax.__version__,backend=jax.default_backend(),
        timesteps_s=space['verification_timesteps_s'],sources={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest()
            for n in ['expanded_design','parameters','dynamics','contact','_generated','geometry','initial_pose','integrators','validation','physical_search']})
    fp = json.loads(json.dumps(fp))
    if args.collect:
        cases = [json.loads((output/f'{i:03d}.json').read_text()) for i in range(len(designs))]
        if any(c['fingerprint']!=fp for c in cases):raise ValueError('Case source/protocol mismatch.')
        rows = []
        for i,proposal in enumerate(neural['proposals'],1):
            pair = paired_acceptance([dict(horizon_s=settings['horizon_s'],condition_id=0,required_timesteps_s=fp['timesteps_s'],
                runs=[dict(r,label=label) for label,k in [('baseline',0),('candidate',i)] for r in cases[k]['runs']])],SimpleNamespace(**space))
            fine = min(cases[i]['runs'],key=lambda r:r['dt_s'])['validation']['metrics']['late_mean_com_speed_m_s']
            error = None if fine is None else abs(fine-proposal['neural_speed_m_s'])
            rows.append(dict(proposal=proposal,physical_speed_m_s=fine,neural_speed_error_m_s=error,
                neural_forecast_passed=bool(error is not None and error<=settings['mean_speed_scale_m_s']),paired_check=pair,
                target_met=bool(pair['passed'] and fine>=proposal['target_m_s'])))
        write_json(output/'report.json',dict(fingerprint=fp,cases=cases,proposals=rows,optimization_enabled=False,
            scope='Canonical initial state only, original hard contact, two timesteps. No perturbed-start or global gradient certification.'))
        print(json.dumps(rows),flush=True);return
    get = jax.jit(lambda d:base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1],hind_angle_deg=d[2]))
    initial = jax.jit(solve_initial_pose);compiled={}
    for i in range(args.worker,len(designs),args.workers):
        path = output/f'{i:03d}.json'
        if path.exists():
            if json.loads(path.read_text())['fingerprint']!=fp:raise ValueError('Existing case fingerprint mismatch.')
            continue
        p = get(jnp.array(designs[i]));x = initial(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
        runs=[]
        for dt in fp['timesteps_s']:
            count = step_count(settings['horizon_s'],dt)
            if dt not in compiled:
                def integrate(x,p,dt=dt,count=count):
                    def step(state,index):
                        nxt = dynamics_step(state,index*dt,p,dt=dt);return nxt,nxt
                    _,tail = jax.lax.scan(step,x,jnp.arange(count))
                    return jnp.concatenate([x[None],tail])
                compiled[dt] = jax.jit(integrate)
            print(f'NeRD proposal {i}, dt={dt}',flush=True)
            times = np.arange(count+1)*dt;states = np.asarray(compiled[dt](x,p))
            validation = inspect_trajectory(times,states,np.asarray(p),config.physical_limits)
            runs.append(dict(dt_s=dt,horizon_s=settings['horizon_s'],condition_id=0,validation=validation))
            if dt==min(fp['timesteps_s']):
                stride = max(1,round(1e-4/dt))
                np.savez_compressed(output/f'{i:03d}.npz',times=times[::stride],states=states[::stride],parameters=np.asarray(p))
        write_json(path,dict(fingerprint=fp,design=designs[i],runs=runs))


if __name__=='__main__':main()
