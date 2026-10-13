"""Compare the reconciled JAX RHS against derivation from the original script."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat
from walking_robot import computeM, computeF, computeSdot, solve_initial_pose, rollout_scan
from walking_robot.design import DesignRun


def quote(path):
    return "'"+str(path).replace("'", "''")+"'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matlab', default='matlab')
    parser.add_argument('--output', type=Path, default=Path('outputs/original_matlab_validation.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jax.config.update('jax_enable_x64', True)
    configuration = DesignRun.from_file(root/'examples/designs/matlab_reference.yaml')
    p = configuration.design.parameters()
    pose = jax.jit(solve_initial_pose)(p)
    # Flight, normal contact, forward/backward slip, stick, saturated memory,
    # and receding feet with zero normal load. Avoid the exact y=0 boundary:
    # original MATLAB heaviside(0)=1/2 whereas the port uses strict y<0.
    states = np.tile(np.asarray(pose), (10,1))
    states[:,1] += [.01]+[-1e-4]*9
    states[:,6] = [0,0,.02,-.02,.0002,-.0002,.0002,-.0002,.02,-.02]
    states[6:8,12:] = [[.001,-.001],[-.001,.001]]
    states[8:,7] = 1.
    times = np.linspace(.0001,.003,len(states))
    evaluate = jax.jit(lambda t,x,p: (computeM(t,x,p),computeF(t,x,p),computeSdot(t,x,p)))
    values = [evaluate(t,jnp.asarray(x),p) for t,x in zip(times,states)]
    mass, force, memory = [np.stack(v) for v in zip(*values)]
    x0 = pose.at[1].add(configuration.simulation.initial_clearance_um*1e-6)
    # Includes first impact (~10 ms), rather than only free fall.
    steps, dt = 2000, 1e-5
    _, trajectory = jax.jit(lambda x,p: rollout_scan(x,p,dt=dt,steps=steps))(x0,p)
    trajectory.block_until_ready()
    with tempfile.TemporaryDirectory(prefix='walking-original-') as folder:
        input_file, output_file = Path(folder)/'input.mat',Path(folder)/'output.mat'
        savemat(input_file,dict(states=states,times=times,x0=np.asarray(x0),dt=dt,steps=steps))
        command = f"addpath({quote(root/'tools')}); matlab_original_reference({quote(input_file)},{quote(output_file)})"
        subprocess.run([args.matlab,'-batch',command],check=True)
        expected = loadmat(output_file)
    errors = {}
    for name, actual in dict(mass=mass,force=force,memory=memory,trajectory=trajectory).items():
        errors[name] = float(np.max(np.abs(np.asarray(actual)-expected[name])))
        np.testing.assert_allclose(actual,expected[name],rtol=1e-7,atol=1e-9)
        print(f'{name}: max absolute difference {errors[name]:.6g}',flush=True)
    report = dict(passed=True,profile='matlab',equation_sample_count=len(states),dt_s=dt,steps=steps,
                  initial_clearance_um=configuration.simulation.initial_clearance_um,max_absolute_errors=errors,
                  scope='Original symbolic balances and contact, matched initial state, trace regularization and RK4; not ode45 or real-robot validation. Exact contact boundary excluded.',
                  reference_tool_sha256=hashlib.sha256((root/'tools/matlab_original_reference.m').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
