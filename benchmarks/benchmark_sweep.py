"""Measure matched CPU/GPU sweep workloads; set JAX_PLATFORMS before launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from walking_robot import solve_initial_pose,rollout_batched
from walking_robot.sweep import load_sweep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/sweeps/matlab_25_cases.yaml'))
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1,25])
    parser.add_argument('--output',type=Path,default=Path('outputs/sweep_benchmark'))
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    if args.steps<1 or args.repeats<2:raise ValueError('Positive steps and at least two repeats required.')
    s,base,designs=load_sweep(args.config)
    rows=[]
    for batch in args.batch_sizes:
        if not 1<=batch<=len(designs):raise ValueError('Batch size leaves the configured design grid.')
        p=jnp.stack([d.parameters() for d in designs[:batch]])
        x=jax.jit(jax.vmap(solve_initial_pose))(p).at[:,1].add(base.simulation.initial_clearance_um*1e-6);x.block_until_ready()
        run=jax.jit(lambda x,p:rollout_batched(x,p,dt=base.simulation.timestep_s,steps=args.steps))
        times=[]
        for repeat in range(args.repeats):
            start=time.perf_counter();_,states=run(x,p);states.block_until_ready();times.append(time.perf_counter()-start)
        rows.append(dict(batch_size=batch,compile_and_first_execution_s=times[0],warm_execution_s=times[1:],
            mean_warm_execution_s=float(np.mean(times[1:])),all_states_finite=bool(np.isfinite(np.asarray(states)).all())))
        print(json.dumps(rows[-1]),flush=True)
    source=Path(__file__).resolve().parents[1]/'src'/'walking_robot'
    report=dict(settings=s,backend=jax.default_backend(),hardware=[dict(device=str(d),kind=d.device_kind) for d in jax.devices()],
        jax_version=jax.__version__,precision='float64',steps=args.steps,timestep_s=base.simulation.timestep_s,
        duration_per_design_s=args.steps*base.simulation.timestep_s,measurements=rows,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        omp_num_threads=os.environ.get('OMP_NUM_THREADS'),openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((source/(n+'.py')).read_bytes()).hexdigest() for n in ['sweep','expanded_design','parameters','integrators','dynamics','contact','_generated','initial_pose']},
        scope='Matched short CPU/GPU workloads; excludes initial pose, plotting and host transfer. No MATLAB or long-horizon speedup claim.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/(jax.default_backend()+'.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
