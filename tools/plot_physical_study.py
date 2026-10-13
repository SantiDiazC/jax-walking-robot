"""Plot saved physical-search and integrator results without resimulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from walking_robot.geometry import points,point_velocities


def observables(path):
    with np.load(path,allow_pickle=False) as data:
        t,x,p = data['t'],data['x'],data['P']
    weights = p[:4]/p[:4].sum()
    g,v = points(x,p),point_velocities(x,p)
    return t,(g['centers']*weights[None,:,None]).sum(1),(v['centers']*weights[None,:,None]).sum(1),g['feet'][...,1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search',type=Path,default=Path('outputs/physical_reference'))
    parser.add_argument('--integrators',type=Path,default=Path('outputs/integrator_comparison'))
    parser.add_argument('--gradients',type=Path,default=Path('outputs/adaptive_gradients'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    search = json.loads((args.search/'report.json').read_text())
    measured = json.loads((args.search/'measurements.json').read_text())['measurements']
    candidate = search['accepted_design'] or search['proposed_design']
    if candidate is not None:
        horizon = max(search['fingerprint']['settings']['verification_horizons_s'])
        dt = min(search['fingerprint']['settings']['verification_timesteps_s'])
        fig,axes = plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
        for label,design in [('Baseline',search['baseline']),('Candidate',candidate)]:
            row = next(r for r in measured if r['design']==design and r['horizon_s']==horizon and r['dt_s']==dt and r['condition_id']==0)
            t,com,velocity,feet = observables(args.search/row['display_trajectory'])
            axes[0].plot(t,1000*(com[:,0]-com[0,0]),label=label)
            axes[1].plot(t,velocity[:,0],label=label)
            axes[2].plot(t,1000*np.minimum(feet.min(1),0),label=label)
        axes[2].axhline(-1000*search['fingerprint']['configuration']['physical_limits']['max_foot_penetration_m'],color='red',ls='--',label='Provisional penetration limit')
        for ax,label in zip(axes,['COM displacement (mm)','COM velocity (m/s)','Lowest foot height (mm)']):
            ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend()
        axes[-1].set_xlabel('Elapsed time (s)')
        fig.suptitle('Physical search: paired fine-step verification; display samples only')
        fig.savefig(args.search/'comparison.png',dpi=150); plt.close(fig)
    if not (args.integrators/'report.json').exists():
        return
    report = json.loads((args.integrators/'report.json').read_text())
    rows = json.loads((args.integrators/'measurements.json').read_text())['measurements']
    comparisons = report['comparisons']
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    labels,errors = [],[]
    for comparison in comparisons:
        d = comparison['design']; mode = comparison['contact_mode']
        for r in comparison['adaptive']:
            label = f'{d[0]:.3f} um / {d[1]:.2f} deg\n{mode} {r["method"]} {r["rtol"]:g}'
            if r['success']:
                labels.append(label); errors.append(1000*r['speed_difference_to_rk4_m_s'])
    if errors:
        axes[0].barh(np.arange(len(errors)),errors)
        axes[0].set_yticks(np.arange(len(errors)),labels,fontsize=7)
    axes[0].set_xlabel('Late speed difference to finest same-contact RK4 (mm/s)')
    axes[0].axvline(0,color='black',lw=.7); axes[0].grid(axis='x',alpha=.2)
    focused_path = args.gradients/'report.json'
    focused = json.loads(focused_path.read_text()) if focused_path.exists() else None
    for c in (focused['cases'] if focused else report['sensitivities']):
        if not c['complete']:
            continue
        fd = np.asarray(c['finite_difference_per_unit'])
        gradient = np.asarray(c['gradient_per_unit'])
        for axis,name in enumerate(['Thickness','Front angle']):
            color = ['tab:blue','tab:orange'][axis]
            x = np.asarray(report['fingerprint']['settings']['gradient_unit_epsilons'])
            axes[1].plot(x,fd[:,axis],'o-',color=color,label=f'{name} FD, rtol={c["rtol"]:g}')
            axes[1].axhline(gradient[axis],color=color,ls='--',alpha=.6,label=f'{name} tangent, rtol={c["rtol"]:g}')
    axes[1].set(xscale='log',xlabel='Finite perturbation / full design range',ylabel='Late speed sensitivity (m/s per unit design)',title='Smooth-contact adaptive sensitivity check')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.2)
    failures = sum(not r['solver']['success'] for r in rows if r['method']!='RK4')
    fig.suptitle(f'Integrator diagnostics; {failures} initial adaptive runs failed and have no plotted score')
    fig.savefig(args.integrators/'comparison.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    main()
