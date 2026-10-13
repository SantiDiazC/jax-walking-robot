"""Double-check an arbitrary MATLAB-profile design against original MATLAB balances.

Default 0.02-second matched RK4 check includes startup contact. This is separate
from the sweep's full-horizon physical screening and timestep verification.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import numpy as np
from scipy.io import loadmat,savemat
from walking_robot import solve_initial_pose,rollout_scan,computeM,computeF,computeSdot
from walking_robot.expanded_design import ExpandedDesignRun
from walking_robot.design import SimulationSettings
from walking_robot.physical_search import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--matlab',default='matlab')
    parser.add_argument('--duration',type=float,default=.02)
    parser.add_argument('--timestep',type=float,default=1.25e-6)
    parser.add_argument('--output',type=Path,default=Path('outputs/sweep_matlab_check.json'))
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    c=ExpandedDesignRun.from_file(args.design)
    if c.design.profile!='matlab':raise ValueError('Original reference check requires profile: matlab.')
    simulation=SimulationSettings(args.duration,args.timestep,c.simulation.initial_clearance_um)
    p=c.design.parameters();pose=jax.jit(solve_initial_pose)(p)
    x0=pose.at[1].add(c.simulation.initial_clearance_um*1e-6)
    samples=np.tile(np.asarray(pose),(10,1));samples[:,1]+=[.01]+[-1e-4]*9
    samples[:,6]=[0,0,.02,-.02,.0002,-.0002,.0002,-.0002,.02,-.02]
    samples[6:8,12:]=[[.001,-.001],[-.001,.001]];samples[8:,7]=1
    times=np.linspace(.0001,.003,len(samples))
    evaluate=jax.jit(lambda t,x,p:(computeM(t,x,p),computeF(t,x,p),computeSdot(t,x,p)))
    values=[evaluate(t,x,p) for t,x in zip(times,samples)]
    mass,force,memory=[np.stack(v) for v in zip(*values)]
    run=jax.jit(lambda x,p:rollout_scan(x,p,dt=args.timestep,steps=simulation.steps))
    timings=[]
    for _ in range(2):
        start=time.perf_counter();_,trajectory=run(x0,p);trajectory.block_until_ready();timings.append(time.perf_counter()-start)
    root=Path(__file__).resolve().parents[1];quote=lambda path:"'"+str(path).replace("'","''")+"'"
    design={k:np.nan if v is None else v for k,v in asdict(c.design).items()}
    with tempfile.TemporaryDirectory(prefix='walking-sweep-matlab-') as folder:
        inp,out=Path(folder)/'input.mat',Path(folder)/'output.mat'
        savemat(inp,dict(design=design,states=samples,times=times,x0=np.asarray(x0),dt=args.timestep,steps=simulation.steps))
        command=f"addpath({quote(root/'tools')}); matlab_sweep_reference({quote(inp)},{quote(out)})"
        start=time.perf_counter();subprocess.run([args.matlab,'-batch',command],check=True);matlab_wall=time.perf_counter()-start
        expected=loadmat(out)
    errors={};passed=True
    for name,actual in dict(mass=mass,force=force,memory=memory,trajectory=np.asarray(trajectory)).items():
        errors[name]=float(np.max(np.abs(actual-expected[name])))
        passed &= bool(np.allclose(actual,expected[name],rtol=1e-7,atol=1e-9))
    report=dict(passed=passed,design=asdict(c.design),duration_s=args.duration,timestep_s=args.timestep,steps=simulation.steps,
        max_absolute_errors=errors,jax_compile_and_first_execution_s=timings[0],jax_warm_execution_s=timings[1],
        comparison_tolerances=dict(rtol=1e-7,atol=1e-9),
        trajectory_max_absolute_errors_by_state=np.max(np.abs(np.asarray(trajectory)-expected['trajectory']),axis=0).tolist(),
        matlab_reference_setup_s=float(expected['setup_s'].item()),matlab_execution_s=float(expected['execution_s'].item()),matlab_process_wall_s=matlab_wall,
        jax_version=jax.__version__,jax_backend=jax.default_backend(),
        reference_tool_sha256=hashlib.sha256((root/'tools/matlab_sweep_reference.m').read_bytes()).hexdigest(),
        comparison_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_source_sha256=hashlib.sha256((root/'reference/matlab/original/RobotDynamics_mulbo_legdirstiff_stribeck_karnopp_iCVD_dam_thick.m').read_bytes()).hexdigest(),
        scope='Original MATLAB symbolic balances, selected design, same initial state, RK4, precision and regularizer. Short contact check, not the full original ode45 study or hardware validation.')
    write_json(args.output,report);print(report,flush=True)
    if not passed:raise RuntimeError('MATLAB/JAX check failed; see the saved diagnostic report.')


if __name__=='__main__':main()
