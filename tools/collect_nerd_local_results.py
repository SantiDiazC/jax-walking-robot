"""Retain local-optimization evidence without committing trajectories/videos."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('outputs/nerd_local_optimization'))
    parser.add_argument('--output',type=Path,default=Path('reference/validation/nerd_local_optimization_results.json'))
    parser.add_argument('--target-run',type=Path,default=Path('outputs/nerd_target_speed'))
    args = parser.parse_args()
    report_file,measurements_file = args.run/'report.json',args.run/'measurements.json'
    report = json.loads(report_file.read_text())
    measurements = json.loads(measurements_file.read_text())
    if report['fingerprint']!=measurements['fingerprint']:
        raise ValueError('Optimization and physical measurements have different provenance.')
    rows = measurements['measurements']
    accepted_rows = [r for r in report['search']['history'] if r['accepted']]
    speeds = [report['search']['initial_speed_m_s']]+[r['physical_speed_m_s'] for r in accepted_rows]
    if not np.all(np.diff(speeds)>0):
        raise ValueError('Accepted physical updates are not monotonically improving.')
    if report['accepted_design'] is not None and not all(
        c['passed'] for c in report['canonical_verification']+report['robustness_verification']):
        raise ValueError('A final accepted design lacks passing independent verification.')
    metrics = [r['validation']['metrics'] for r in rows]
    names = sorted({n for m in metrics for n in m if n.startswith('max_')})
    finite_values = {n:[m[n] for m in metrics if m.get(n) is not None and np.isfinite(m[n])] for n in names}
    peaks = {n:max(v) if v else None for n,v in finite_values.items()}
    regimes = []
    for row in rows:
        with np.load(args.run/row['display_trajectory'],allow_pickle=False) as path:
            regimes.append(path['parameters'][[22,26,41]])
    if not np.all(np.array(regimes)==regimes[0]):
        raise ValueError('Recorded runs cross the fixed rounded-parameter regime.')
    result = dict(report,raw_report_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest(),
        physical_measurements=dict(sha256=hashlib.sha256(measurements_file.read_bytes()).hexdigest(),
            count=len(rows),physically_passed=sum(r['validation']['passed'] for r in rows),peak_metrics=peaks,
            nonfinite_or_missing_metric_counts={n:len(rows)-len(v) for n,v in finite_values.items()},
            rounded_parameters=dict(drive_frequency_hz=float(regimes[0][0]),front_normal_stiffness_n_m=float(regimes[0][1]),
                                    hind_normal_stiffness_n_m=float(regimes[0][2]))))
    target_file = args.target_run/'report.json'
    if target_file.exists():
        target = json.loads(target_file.read_text())
        if target['provenance']['checkpoint_sha256']!=report['fingerprint']['checkpoint_sha256']:
            raise ValueError('Target study uses a different neural checkpoint.')
        for trial in target['experiments']:
            history = trial.pop('history')
            trial['recorded_iterations'] = len(history)
            trial['minimum_recorded_loss'] = min(row['loss'] for row in history)
        result['target_speed_exploration'] = dict(target,raw_report_sha256=hashlib.sha256(target_file.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(optimization_enabled=result['optimization_enabled'],accepted_design=result['accepted_design'],
                         physical_improvement_m_s=result['search']['improvement_m_s'])))


if __name__=='__main__':
    main()
