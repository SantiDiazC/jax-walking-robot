"""Collect fixed-actuation physical evidence and optional experimental NeRD results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/walking-robot-mpl')
import numpy as np
from walking_robot.geometry import points,point_velocities
from walking_robot.physical_search import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical',type=Path,default=Path('outputs/expanded_design_search'))
    parser.add_argument('--neural',type=Path,default=Path('outputs/expanded_nerd/training'))
    parser.add_argument('--output',type=Path,default=Path('reference/validation/expanded_design_results.json'))
    args = parser.parse_args()
    report = json.loads((args.physical/'report.json').read_text());chosen=report['accepted_id']
    if chosen is None:raise ValueError('No accepted physical design.')
    conditions = []
    for condition in range(1+len(report['robustness'])):
        folder = args.physical/('verification' if condition==0 else f'robustness_{condition}')
        pair = {label:json.loads((folder/f'{i:03d}.json').read_text()) for label,i in [('baseline',0),('candidate',chosen)]}
        if any(r['fingerprint']!=report['fingerprint'] for r in pair.values()):raise ValueError('Physical source/protocol mismatch.')
        rows = {label:sorted(r['runs'],key=lambda x:x['dt_s']) for label,r in pair.items()}
        conditions.append(dict(condition_id=condition,**{label:rows[label][0]['validation']['metrics'] for label in rows},
            physical_passed=all(x['validation']['passed'] for r in rows.values() for x in r)))
    summary = dict(physical=report,conditions=conditions,
        min_candidate_speed_m_s=min(r['candidate']['late_mean_com_speed_m_s'] for r in conditions),
        max_candidate_penetration_m=max(r['candidate']['max_foot_penetration_m'] for r in conditions),
        target_scope='Mean COM speed during seconds 1–2 of a two-second simulation. Four tested starts, not a hardware guarantee.',
        neural_found_physical_candidate=False)
    neural = args.neural/'report.json'
    if neural.exists():
        nr = json.loads(neural.read_text())
        if hashlib.sha256((args.neural/'model.pt').read_bytes()).hexdigest()!=nr['checkpoint_sha256']:
            raise ValueError('Neural checkpoint/report mismatch.')
        for name in ['expanded_design','geometry']:
            if nr['provenance']['sources'][name]!=report['fingerprint']['sources'][name]:
                raise ValueError('Neural/physical source mismatch.')
        summary['neural'] = nr
        verified = args.neural/'physical_verification'/'report.json'
        if verified.exists():
            vr = json.loads(verified.read_text())
            if vr['fingerprint']['neural_report_sha256']!=hashlib.sha256(neural.read_bytes()).hexdigest():
                raise ValueError('Neural proposal verification belongs to another report.')
            summary['neural_proposal_verification'] = vr
    write_json(args.output,summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,3,figsize=(13,3.8))
    for label,i in [('Baseline',0),('Candidate',chosen)]:
        with np.load(args.physical/'verification'/f'{i:03d}.npz',allow_pickle=False) as f:
            t,x,p = [f[k] for k in ['times','states','parameters']]
        weights = p[:4]/p[:4].sum()
        c = (points(x,p)['centers']*weights[None,:,None]).sum(-2)
        v = (point_velocities(x,p)['centers']*weights[None,:,None]).sum(-2)
        feet = points(x,p)['feet'][...,1]
        ax[0].plot(t,c[:,0],label=label)
        ax[1].plot(t[::20],v[::20,0],label=label)
        ax[2].plot(t[::20],np.maximum(-feet[::20],0).max(-1)*1e3,label=label)
    ax[0].set(ylabel='Horizontal COM position (m)')
    ax[1].set(ylabel='Horizontal COM velocity (m/s)')
    ax[2].set(ylabel='Foot penetration (mm)');ax[2].axhline(.25,color='black',ls='--',label='Provisional limit')
    for a in ax:a.set_xlabel('Time (s)');a.grid(alpha=.25);a.legend()
    fig.tight_layout();fig.savefig(args.physical/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['min_candidate_speed_m_s','max_candidate_penetration_m']}))


if __name__=='__main__':main()
