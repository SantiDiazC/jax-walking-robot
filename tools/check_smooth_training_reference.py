"""Refine held-out teacher sensitivities at a tenfold tighter tolerance."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_SKIP_CUDA_CONSTRAINTS_CHECK','true')
import jax
import jax.numpy as jnp
import numpy as np
from generate_smooth_training_data import load,fingerprint
from validate_long_horizon_gradients import observations
from walking_robot import solve_initial_pose
from walking_robot.adaptive_jax import JaxAdaptiveDynamics
from walking_robot.contact_residual import ContactSmoothing
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.physical_search import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/nerd_trajectory.yaml'))
    parser.add_argument('--data',type=Path,default=Path('outputs/nerd_trajectory/data'))
    parser.add_argument('--output',type=Path,default=Path('outputs/nerd_trajectory/teacher_checks'))
    parser.add_argument('--id',type=int,action='append')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    s,c,designs,split = load(args.study)
    source = json.loads((args.data/'report.json').read_text())
    if source['fingerprint']!=fingerprint(s,c):
        raise ValueError('Numerical reference sources/settings differ.')
    ids = np.flatnonzero(split!=0) if args.id is None else args.id
    if any(i not in np.flatnonzero(split!=0) for i in ids):
        raise ValueError('Reference refinement is restricted to held-out designs.')
    args.output.mkdir(parents=True,exist_ok=True)
    span = jnp.array(DESIGN_UPPER)-jnp.array(DESIGN_LOWER)
    def prepare(u):
        d = jnp.array(DESIGN_LOWER)+u*span
        p = c.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        return p,solve_initial_pose(p).at[1].add(c.simulation.initial_clearance_um*1e-6)
    get,derivative = jax.jit(prepare),jax.jit(jax.jacfwd(prepare))
    solver = JaxAdaptiveDynamics('smooth',ContactSmoothing(s['gap_width_um']*1e-6,s['velocity_width_m_s'],s['force_width_uN']*1e-6))
    for i in ids:
        path = args.output/f'{i:03d}.json'
        if path.exists():
            raise ValueError('Use a fresh output directory for repeated verification.')
        u = (jnp.array(designs[i])-jnp.array(DESIGN_LOWER))/span
        p,x = get(u);dp,dx = derivative(u)
        print(f'Refining held-out teacher design {i}.',flush=True)
        info,result = solver.integrate(x,p,s['horizon_s'],rtol=s['reference_rtol']/10,
            sample_dt=s['integration_timestep_s'],max_step=2e-4,max_rhs_calls=12000000,
            initial_sensitivity=dx,parameter_sensitivity=dp,physical_limits=c.physical_limits)
        row = dict(id=int(i),design=designs[i].tolist(),teacher_fingerprint=source['fingerprint'],
            refined_rtol=s['reference_rtol']/10,solver=info,passed=False)
        if result is not None:
            with np.load(args.data/f'{i:03d}.npz',allow_pickle=False) as f:
                original = {k:f[k] for k in ['times','states','sensitivities']}
            a = observations(original,np.asarray(p),[.2,1.,s['horizon_s']],np.asarray(dp))
            b = observations(result,np.asarray(p),[.2,1.,s['horizon_s']],np.asarray(dp))
            checks = []
            for h in a:
                ga,gb = np.array(a[h]['gradient_per_unit']),np.array(b[h]['gradient_per_unit'])
                error = float(np.linalg.norm(ga-gb)/max(np.linalg.norm(gb),1e-10))
                ve = abs(a[h]['speed_m_s']-b[h]['speed_m_s'])
                checks.append(dict(horizon_s=float(h),original=a[h],refined=b[h],gradient_relative_change=error,
                    speed_change_m_s=ve,passed=bool(error<=.05 and ve<=2.5e-5)))
            row.update(checks=checks,passed=bool(info['accepted_step_physical_validation']['passed'] and all(c['passed'] for c in checks)))
        write_json(path,row)
        print(f"Teacher {i}: {row['passed']}",flush=True)


if __name__=='__main__':
    main()
