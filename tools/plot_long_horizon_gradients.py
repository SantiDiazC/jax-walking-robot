"""Plot the completed multi-design study without repeating integrations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/long_horizon_gradients'))
    parser.add_argument('--verification',type=Path,default=Path('outputs/gradient_proposal_verification/report.json'))
    args = parser.parse_args()
    report = json.loads((args.output/'report.json').read_text())
    if report['status']!='complete':
        raise ValueError('Collect all completed design reports before plotting.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    designs = report['designs']
    refinements = []
    if args.verification.exists():
        verified = json.loads(args.verification.read_text())
        if verified['original_report_sha256']!=hashlib.sha256((args.output/'report.json').read_bytes()).hexdigest():
            raise ValueError('Proposal refinement belongs to a different original study.')
        refinements = verified['checks']
    horizons = report['protocol']['horizons_s']
    errors = np.full((len(designs),len(horizons)),np.nan)
    labels,smooth,hard,uncertainty = [],[],[],[]
    for i,design in enumerate(designs):
        for j,case in enumerate(design['cases']):
            check = case['tolerance_checks'][-1]
            if check['complete']:
                errors[i,j] = 100*max(check['relative_errors'])
            labels.append(f"{design['name'].replace('_',' ')}\n{case['horizon_s']:g} s")
            direction,bridge = case['direction_check'],case['hard_bridge']
            extra = next((c for c in refinements if c['name']==design['name'] and c['horizon_s']==case['horizon_s'] and 'refined_check' in c),None)
            if extra:
                refined = extra['refined_check']['checks'][0]
                bridge = dict(gains_m_s=[refined.get('improvement_m_s',np.nan)],
                              estimated_uncertainty_m_s=refined.get('estimated_pair_uncertainty_m_s',np.nan))
            smooth.append(1000*direction['actual_gain_m_s'] if direction and 'actual_gain_m_s' in direction else np.nan)
            hard.append(1000*bridge['gains_m_s'][-1] if bridge and 'gains_m_s' in bridge else np.nan)
            uncertainty.append(1000*report['protocol']['uncertainty_factor']*bridge['estimated_uncertainty_m_s'] if bridge and 'estimated_uncertainty_m_s' in bridge else 0.)
    fig,axes = plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    shown = axes[0].imshow(np.ma.masked_invalid(errors),aspect='auto',cmap='viridis',vmin=0,
                          vmax=max(1.,float(np.nanmax(errors))) if np.isfinite(errors).any() else 1.)
    axes[0].set_xticks(np.arange(len(horizons)),[f'{h:g} s' for h in horizons])
    axes[0].set_yticks(np.arange(len(designs)),[d['name'].replace('_',' ') for d in designs])
    for i in range(len(designs)):
        for j in range(len(horizons)):
            axes[0].text(j,i,f'{errors[i,j]:.2f}%' if np.isfinite(errors[i,j]) else 'incomplete',ha='center',va='center',color='white')
    axes[0].set_title('Worst FD gradient error at the finest tolerance')
    fig.colorbar(shown,ax=axes[0],label='Relative vector error (%)')
    x = np.arange(len(labels))
    axes[1].bar(x-.18,smooth,width=.36,label='Smooth-contact measured gain')
    axes[1].bar(x+.18,hard,width=.36,yerr=uncertainty,capsize=3,label='Hard-contact gain ± empirical resolution screen')
    axes[1].set_xticks(x,labels,rotation=40,ha='right',fontsize=7)
    axes[1].set_ylabel('Late mean speed gain (mm/s)')
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].grid(axis='y',alpha=.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title('Small gradient proposals, independently resimulated')
    fig.suptitle('Local fixed-rounding regimes; no neural or hardware certification')
    fig.savefig(args.output/'gradient_validation.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
