"""Commit compact sweep, CPU/GPU and MATLAB evidence without large trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from walking_robot.physical_search import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dropped',type=Path,default=Path('outputs/matlab_25_sweep_v2/report.json'))
    parser.add_argument('--supported',type=Path,default=Path('outputs/supported_25_sweep/report.json'))
    parser.add_argument('--benchmark',type=Path,default=Path('outputs/sweep_benchmark_isolated'))
    parser.add_argument('--output',type=Path,default=Path('reference/validation/sweep_results.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report={}
    for label,path in [('original_drop',args.dropped),('supported_start',args.supported)]:
        row=json.loads(path.read_text())
        if row['fingerprint']['tool_sha256']!=digest(root/'examples/run_sweep.py'):raise ValueError('Sweep runner source mismatch.')
        for name,value in row['fingerprint']['sources'].items():
            if value!=digest(root/'src/walking_robot'/f'{name}.py'):raise ValueError('Numerical source mismatch.')
        report[label]=row
    benchmarks=[json.loads((args.benchmark/(backend+'.json')).read_text()) for backend in ['cpu','gpu']]
    if any(b['benchmark_sha256']!=digest(root/'benchmarks/benchmark_sweep.py') for b in benchmarks):raise ValueError('Benchmark source mismatch.')
    for b in benchmarks:
        for name,value in b['source_sha256'].items():
            if value!=digest(root/'src/walking_robot'/f'{name}.py'):raise ValueError('Benchmark numerical source mismatch.')
    if any(benchmarks[0][key]!=benchmarks[1][key] for key in ['settings','steps','timestep_s','duration_per_design_s','precision','jax_version']):
        raise ValueError('CPU/GPU workloads are not matched.')
    report['backend_comparison']=benchmarks
    checks=[]
    for name in ['sweep_matlab_check','sweep_matlab_check_refined','sweep_matlab_check_fine']:
        path=root/'outputs'/f'{name}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['reference_tool_sha256']!=digest(root/'tools/matlab_sweep_reference.m'):raise ValueError('MATLAB reference source mismatch.')
            if row['original_source_sha256']!=digest(root/'reference/matlab/original/RobotDynamics_mulbo_legdirstiff_stribeck_karnopp_iCVD_dam_thick.m'):raise ValueError('MATLAB archive mismatch.')
            checks.append(row)
    report['matlab_checks']=checks
    report['scope']='Measured sweeps and short matched CPU/GPU/MATLAB checks. Preserves failed physical screens and strict MATLAB comparisons; no full ode45 speedup or global-optimum certification.'
    write_json(args.output,report)
    print(args.output)


if __name__=='__main__':main()
