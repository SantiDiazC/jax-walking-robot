"""Screen supported/drop initial conditions and timestep sensitivity from YAML."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from walking_robot import solve_initial_pose, rollout_scan
from walking_robot.design import DesignRun
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=Path('examples/designs/matlab_supported.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/reconciliation_timestep'))
    parser.add_argument('--duration',type=float,default=.2)
    parser.add_argument('--timesteps',type=float,nargs='+',default=[1e-5,5e-6,2.5e-6])
    args = parser.parse_args()
    configuration = DesignRun.from_file(args.design)
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('Duration must be finite and positive.')
    for dt in args.timesteps:
        if not np.isfinite(dt) or dt <= 0 or not np.isclose(round(args.duration/dt)*dt,args.duration,atol=1e-12,rtol=0):
            parser.error('Timesteps must be positive and divide duration exactly.')
    jax.config.update('jax_enable_x64',True)
    args.output.mkdir(parents=True,exist_ok=True)
    p = configuration.design.parameters()
    x0 = jax.jit(solve_initial_pose)(p).at[1].add(configuration.simulation.initial_clearance_um*1e-6)
    simulate = jax.jit(rollout_scan,static_argnames=('dt','steps','method'))
    cases = []
    for dt in args.timesteps:
        start = time.perf_counter()
        t,x = simulate(x0,p,dt=dt,steps=round(args.duration/dt))
        x.block_until_ready()
        elapsed = time.perf_counter()-start
        report = inspect_trajectory(t,x,p,configuration.physical_limits)
        cases.append(dict(dt_s=dt,horizon_s=args.duration,compile_and_run_s=elapsed,validation=report))
        np.savez_compressed(args.output/f'dt_{dt:g}.npz',t=t,x=x,P=p)
        print(json.dumps(cases[-1]),flush=True)
    reference = min(cases,key=lambda c:c['dt_s'])
    comparisons = []
    for case in cases:
        differences = []
        for name in ['late_mean_com_speed_m_s','max_foot_penetration_m']:
            a,b = case['validation']['metrics'].get(name),reference['validation']['metrics'].get(name)
            differences.append(abs(a-b) if a is not None and b is not None and np.isfinite([a,b]).all() else None)
        speed,penetration = differences
        comparisons.append(dict(dt_s=case['dt_s'],late_mean_speed_difference_m_s=speed,max_penetration_difference_m=penetration,
                                passed_aggregate_check=bool(speed is not None and penetration is not None and speed<=.01 and penetration<=2e-5)))
    result = dict(configuration=asdict(configuration),cases=cases,comparisons_to_finest=comparisons,
                  all_physical_checks_passed=all(c['validation']['passed'] for c in cases),
                  all_aggregate_timestep_checks_passed=all(c['passed_aggregate_check'] for c in comparisons),
                  scope='Specified design and horizons only; aggregate agreement does not validate gradients or a real robot.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
