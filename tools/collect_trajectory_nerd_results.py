"""Collect immutable training evidence and plot held-out acceptance metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from walking_robot.geometry import point_velocities
from walking_robot.surrogate_checks import validated_objective_horizons


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/nerd_trajectory'))
    parser.add_argument('--output',type=Path,default=Path('reference/validation/nerd_trajectory_results.json'))
    args = parser.parse_args()
    final_path = args.root/'consistent/report.json'
    final = json.loads(final_path.read_text())
    s = final['settings']
    cases = []
    for name,result in final['results'].items():
        with np.load(args.root/'data/dataset.npz',allow_pickle=False) as f:
            reference_states = f['states']
        with np.load(args.root/f'consistent/{name}_prediction.npz',allow_pickle=False) as f:
            prediction = {k:f[k] for k in f.files}
        p = prediction['parameters'][:,None,:]
        weights = p[...,:4]/p[...,:4].sum(-1,keepdims=True)
        def com_velocity(states):
            return (point_velocities(states,p)['centers']*weights[...,None]).sum(-2)
        velocity_error = com_velocity(prediction['states'])-com_velocity(reference_states[prediction['ids']])
        for c in result['cases']:
            index = list(prediction['ids']).index(c['id'])
            gradients = [dict(id=g['id'],horizon_s=g['horizon_s'],
                neural_gradient_per_unit=g['neural_gradient_per_unit'],teacher_gradient_per_unit=g['teacher_gradient_per_unit'],
                relative_error=g['relative_error'],cosine=g['cosine'],passed=g['passed'],
                max_self_finite_difference_error=max(f['relative_error'] for f in g['self_finite_differences']))
                for g in result['derivatives']['cases'] if g['id']==c['id']]
            kinematic = next(k for k in result['derivatives']['kinematics'] if k['id']==c['id'])
            cases.append(dict(split=name,id=c['id'],design=c['design'],state_rmse=c['state_rmse'],
                foot_height_rmse_m=c['foot_height_rmse_m'],horizons=c['horizons'],
                physical_refinements=[dict(sample_timestep_s=r['sample_timestep_s'],passed=r['validation']['passed'],
                    metrics=r['validation']['metrics']) for r in c['physical_refinements']],
                gradients=gradients,kinematic=kinematic,trajectory_passed=c['passed']))
            cases[-1]['post_evaluation_diagnostics'] = dict(
                instantaneous_com_velocity_vector_rmse_m_s=float(np.sqrt(np.mean(velocity_error[index]**2))),
                instantaneous_com_vx_rmse_m_s=float(np.sqrt(np.mean(velocity_error[index,:,0]**2))))
    stages = []
    for name in ['training','refined','consistent']:
        path = args.root/name/'report.json'
        report = json.loads(path.read_text())
        if report['provenance']['dataset_sha256']!=final['provenance']['dataset_sha256']:
            raise ValueError('Training-stage comparisons require identical data.')
        log = json.loads((args.root/name/'training.json').read_text())['log']
        stages.append(dict(name=name,selected_iteration=report['selected_iteration'],
            training_elapsed_s=log[-1]['elapsed_s'],report_sha256=digest(path),
            checkpoint_sha256=report['checkpoint_sha256'],provenance={k:v for k,v in report['provenance'].items() if k!='teacher'},
            results={k:dict(trajectory_gate_passed=v['trajectory_gate_passed'],
                gradient_gate_passed=v['derivatives']['gradient_gate_passed'],
                kinematic_gate_passed=v['derivatives']['kinematic_gate_passed']) for k,v in report['results'].items()}))
    teacher = [json.loads(p.read_text()) for p in sorted((args.root/'teacher_checks').glob('*.json'))]
    proposals = []
    for p in sorted((args.root/'proposals').glob('*/report.json')):
        raw = json.loads(p.read_text())
        compact = {k:v for k,v in raw.items() if k!='runs'}
        compact['runs'] = [dict(label=r['label'],timestep_s=r['timestep_s'],elapsed_s=r['elapsed_s'],
            passed=r['validation']['passed'],metrics=r['validation']['metrics']) for r in raw['runs']]
        proposals.append(dict(report=compact,report_sha256=digest(p)))
    reload_check = None
    if (args.root/'recheck/report.json').exists():
        recheck = json.loads((args.root/'recheck/report.json').read_text())
        if recheck['provenance']!=final['provenance']:
            raise ValueError('Checkpoint reload lost training provenance.')
        for name in ['validation','test']:
            with np.load(args.root/f'consistent/{name}_prediction.npz') as a, np.load(args.root/f'recheck/{name}_prediction.npz') as b:
                if not np.array_equal(a['states'],b['states']):
                    raise ValueError('Checkpoint reload changed trajectories.')
            if any(recheck['results'][name][k]!=final['results'][name][k] for k in ['cases','derivatives']):
                raise ValueError('Checkpoint reload changed verification metrics.')
        reload_check = dict(report_sha256=digest(args.root/'recheck/report.json'),
            bitwise_equal_trajectories=True,identical_physical_gradient_kinematic_checks=True,training_provenance_preserved=True)
    limits = final['provenance']['teacher']['configuration']['physical_limits']
    count = s['design_count']
    validated_horizons = validated_objective_horizons(final)
    result = dict(settings=s,cases=cases,stages=stages,upstream=final['upstream'],
        teacher_provenance=final['provenance']['teacher'],checkpoint_reload_verification=reload_check,
        final_report_sha256=digest(final_path),surrogate_gates_passed=final['surrogate_gates_passed'],
        objective_gradient_gate_horizons_s=validated_horizons,
        full_dynamics_validation_passed=False,
        optimization_enabled=False,accepted_design=None,
        reference_refinements=[dict(id=r['id'],passed=r['passed'],checks=r.get('checks',[])) for r in teacher],
        previously_verified_anchors=dict(ids=[count*2//3,count*5//6,count*5//6+1],source='long_horizon_gradient_results.json',
            sha256=digest(Path('reference/validation/long_horizon_gradient_results.json'))),
        physical_proposals=proposals,physical_limits=limits,
        executed_proposal_tool_sha256=digest(args.root/'proposals/executed_tool.py') if proposals else None,
        selection_policy='All architecture/loss changes and weight selection used validation only. Stage test comparisons were read after selecting the final checkpoint.',
        aborted_attempt='Native LayerNorm mixed-derivative test failed; its partial run is excluded. The final elementary normalization passes those tests.',
        additional_limitation='Plot inspection revealed excess instantaneous COM velocity oscillations despite accurate displacement, mean speed and foot height. This post-evaluation diagnostic was not used to retune the final model; full dynamics equivalence is not established.',
        precision='Float32 training; float64 final trajectory, kinematic, gradient and proposal inference.',
        scope='18 designs: 12 training, 3 validation, 3 test; canonical supported start, fixed voltage and phase, horizons 0.2/1/2 s. Local continuous regimes only; no arbitrary-state, hardware, fabrication or global-optimum certification.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    labels = [f'{c["split"][:3]} {c["id"]}' for c in cases]
    colors = ['C0' if c['split']=='validation' else 'C1' for c in cases]
    x = np.arange(len(cases))
    quantities = [([max(h['speed_error_m_s'] for h in c['horizons'])*1000 for c in cases],
                   s['max_speed_error_m_s']*1000,'Maximum COM speed error (mm/s)'),
                  ([max(r['metrics']['max_foot_penetration_m'] for r in c['physical_refinements'])*1000 for c in cases],
                   limits['max_foot_penetration_m']*1000,'Maximum foot penetration (mm)'),
                  ([max(g['relative_error'] for g in c['gradients'])*100 for c in cases],
                   s['max_gradient_relative_error']*100,'Maximum gradient vector error (%)'),
                  ([c['state_rmse'] for c in cases],s['max_state_rmse'],'Normalized state RMSE')]
    for ax,(values,limit,label) in zip(axes.flat,quantities):
        ax.bar(x,values,color=colors)
        ax.axhline(limit,color='red',ls='--',label='Acceptance threshold')
        ax.set_xticks(x,labels,rotation=20);ax.set_ylabel(label);ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('NeRD trajectory decoder — held-out canonical designs\n0.2, 1 and 2 s; provisional physical limits')
    fig.savefig(args.root/'validation_summary.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(surrogate_gates_passed=result['surrogate_gates_passed'],
                         physical_proposals_passed=[p['report']['passed'] for p in proposals])))


if __name__=='__main__':
    main()
