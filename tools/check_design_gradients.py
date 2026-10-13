"""Exploratory local JAX design derivatives compared with finite differences."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from walking_robot import solve_initial_pose, rollout_scan
from walking_robot.design import DesignRun
from walking_robot.geometry import points
from walking_robot.validation import inspect_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=Path('examples/designs/matlab_supported.yaml'))
    parser.add_argument('--horizon',type=float,default=.005)
    parser.add_argument('--epsilons',type=float,nargs='+',default=[1e-3,1e-4,1e-5])
    parser.add_argument('--output',type=Path,default=Path('outputs/local_physical_gradients.json'))
    args = parser.parse_args()
    configuration = DesignRun.from_file(args.design)
    settings = replace(configuration.simulation,duration_s=args.horizon)
    if settings.steps < 2:
        parser.error('Need at least two steps.')
    base = configuration.design
    for epsilon in args.epsilons:
        if not np.isfinite(epsilon) or epsilon <= 0:
            parser.error('Epsilons must be finite and positive.')
        for sign in [-1,1]:
            replace(base,adhesive_thickness_um=base.adhesive_thickness_um+sign*epsilon)
            replace(base,front_angle_deg=base.front_angle_deg+sign*epsilon)
    jax.config.update('jax_enable_x64',True)
    def trajectory(d):
        p = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        x0 = solve_initial_pose(p).at[1].add(settings.initial_clearance_um*1e-6)
        t,x = rollout_scan(x0,p,dt=settings.timestep_s,steps=settings.steps)
        return t,x,p
    def objective(d):
        t,x,p = trajectory(d)
        centers = points(x,p,xp=jnp)['centers'][...,0]
        com = (centers*p[:4]).sum(-1)/p[:4].sum()
        halfway = settings.steps//2
        return (com[-1]-com[halfway])/(t[-1]-t[halfway])
    d = jnp.array([base.adhesive_thickness_um,base.front_angle_deg])
    value = jax.jit(objective)
    actual = np.asarray(jax.jit(jax.jacfwd(objective))(d))
    cases = []
    for epsilon in args.epsilons:
        finite_difference = []
        for i in range(2):
            shift = jnp.zeros(2).at[i].set(epsilon)
            finite_difference.append(float((value(d+shift)-value(d-shift))/(2*epsilon)))
        errors = np.abs(actual-finite_difference)
        cases.append(dict(epsilon=epsilon,finite_difference=finite_difference,absolute_difference=errors.tolist(),
                          matches=bool(np.isfinite(actual).all() and np.isfinite(finite_difference).all()
                                       and np.allclose(actual,finite_difference,rtol=.005,atol=1e-8))))
    t,x,p = jax.jit(trajectory)(d)
    report = dict(configuration=asdict(configuration),horizon_s=args.horizon,dt_s=settings.timestep_s,
                  design_variables=['adhesive_thickness_um','front_angle_deg'],
                  objective='Late-half mean COM velocity in m/s',autodiff=actual.tolist(),finite_difference_checks=cases,
                  physical_validation=inspect_trajectory(t,x,p,configuration.physical_limits),
                  scope='Exploratory local derivatives through initial pose and hard contact. No long-horizon or global gradient guarantee.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def serializable(value):
        if isinstance(value,float) and not np.isfinite(value):
            return None
        if isinstance(value,list):
            return [serializable(item) for item in value]
        if isinstance(value,dict):
            return {key:serializable(item) for key,item in value.items()}
        return value
    report = serializable(report)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
