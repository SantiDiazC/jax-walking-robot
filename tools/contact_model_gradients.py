"""Check the selected contact network's local design sensitivities.

Model finite differences check autodiff implementation. Independently refined
physical differences check fidelity; unstable physical differences cannot
certify a network, even when its own autodiff agrees with its own rollouts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from walking_robot import solve_initial_pose
from walking_robot.contact_residual import ContactSmoothing,contact_rollout
from walking_robot.design import DesignRun
from walking_robot.geometry import points
from walking_robot.learning_spec import DESIGN_LOWER,DESIGN_UPPER
from walking_robot.validation import inspect_trajectory
from contact_residual_study import load_reference
from contact_sensitivity_study import probes_for,finite_differences,write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('examples/learning/contact_structure.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/contact_structure'))
    args = parser.parse_args()
    settings = yaml.safe_load(args.study.read_text())
    root = args.study.resolve().parent
    config = DesignRun.from_file(root/settings['design'])
    jax.config.update('jax_enable_x64',True)
    data,provenance = load_reference(settings,config,root)
    trained = json.loads((args.output/'training.json').read_text())
    if trained['configuration']!=settings or trained['dataset_sha256']!=provenance['dataset_sha256']:
        raise ValueError('Selected checkpoint settings or dataset mismatch.')
    checkpoint = args.output/'contact_network.npz'
    with np.load(checkpoint,allow_pickle=False) as saved:
        weights = {k:jnp.asarray(saved[k]) for k in saved.files}
    mode = settings['training_mode']
    physical_path = root/settings[mode+'_sensitivity_report']
    physical = json.loads(physical_path.read_text())
    if physical.get('status')!='complete' or physical['dataset_sha256']!=provenance['dataset_sha256'] or physical['horizon_s']!=settings['horizon_s']:
        raise ValueError('Physical sensitivity report is incomplete or mismatched.')
    if physical['unit_epsilons']!=settings['gradient_unit_epsilons']:
        raise ValueError('Physical sensitivity perturbations differ from YAML.')
    dt,substeps = settings['neural_timestep_s'],settings['integration_substeps']
    count = round(settings['horizon_s']/dt)
    if not np.isclose(count*dt,settings['horizon_s'],rtol=0,atol=1e-12):
        raise ValueError('Horizon must be an integer number of macro steps.')
    options = dict(mode=mode,smoothing=ContactSmoothing(settings['gap_width_um']*1e-6,
        settings['velocity_width_m_s'],settings['force_width_uN']*1e-6),max_log_correction=settings['max_log_correction'])
    def trajectory(d,*,full=False):
        p = config.design.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])
        x = solve_initial_pose(p).at[1].add(config.simulation.initial_clearance_um*1e-6)
        t,x = contact_rollout(x,p,dt,count,weights,substeps=substeps,return_substeps=full,**options)
        com = (points(x,p,xp=jnp)['centers'][...,0]*p[:4]).sum(-1)/p[:4].sum()
        speed = (com[-1]-com[(len(t)-1)//2])/(t[-1]-t[(len(t)-1)//2])
        return speed,t,x,p
    value_and_trajectory = jax.jit(jax.vmap(lambda d:trajectory(d,full=True)))
    derivative = jax.jit(jax.jacfwd(lambda d:trajectory(d)[0]))
    span = np.asarray(DESIGN_UPPER)-DESIGN_LOWER
    cases = []
    for design_id in settings['gradient_design_ids']:
        ids = np.flatnonzero((data['design_id']==design_id)&(data['variant']==0))
        if len(ids)!=1:
            raise ValueError('Design must have exactly one canonical trajectory.')
        center = data['designs'][ids[0]]
        probes = probes_for(center,settings['gradient_unit_epsilons'])
        if not np.all((probes>=DESIGN_LOWER)&(probes<=DESIGN_UPPER)):
            raise ValueError('Sensitivity probes leave the design domain.')
        print(f'Contact network gradients: design {design_id}...',flush=True)
        values,t,x,p = jax.tree.map(np.asarray,value_and_trajectory(jnp.asarray(probes)))
        grad = np.asarray(derivative(jnp.asarray(center)))*span
        fd = np.asarray(finite_differences(values,settings['gradient_unit_epsilons']))
        reports = [inspect_trajectory(tt,xx,pp,config.physical_limits) for tt,xx,pp in zip(t,x,p)]
        reference = next(c for c in physical['cases'] if c['design_id']==design_id)
        if reference['design']!=center.tolist():
            raise ValueError('Physical gradient center differs from checkpoint probe.')
        finest = reference['measurements'][-1]
        ref_fd = np.asarray(finest['finite_difference_per_unit'])
        self_error = np.linalg.norm(fd-grad,axis=-1)/np.maximum(np.linalg.norm(fd,axis=-1),1e-8)
        ref_error = np.linalg.norm(grad-ref_fd,axis=-1)/np.maximum(np.linalg.norm(ref_fd,axis=-1),1e-8)
        finite = bool(np.isfinite(grad).all() and np.isfinite(fd).all())
        cases.append(dict(design_id=design_id,design=center.tolist(),autodiff_per_unit=grad.tolist(),
            finite_difference_per_unit=fd.tolist(),model_self_relative_errors=self_error.tolist(),
            model_self_check_passed=bool(finite and np.all(self_error<=.25)),
            physical_finest_timestep_s=finest['dt_s'],physical_finite_difference_per_unit=ref_fd.tolist(),
            relative_errors_to_physical_differences=ref_error.tolist(),
            physical_sensitivity_gate_passed=reference['sensitivity_gate_passed'],
            physical_reports=reports,
            gradient_gate_passed=bool(finite and all(r['passed'] for r in reports) and np.all(self_error<=.25) and
                reference['sensitivity_gate_passed'] and np.all(ref_error<=.25))))
    report = dict(cases=cases,mode=mode,horizon_s=settings['horizon_s'],macro_timestep_s=dt,integration_substeps=substeps,
        unit_epsilons=settings['gradient_unit_epsilons'],selected_iteration=trained['selected_iteration'],
        dataset_sha256=provenance['dataset_sha256'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        physical_report_sha256=hashlib.sha256(physical_path.read_bytes()).hexdigest(),
        gradient_gate_passed=all(c['gradient_gate_passed'] for c in cases),
        scope='Local canonical initial states only; no optimization or accepted design.')
    write(args.output/'gradient_validation.json',report)
    summary_path = args.output/'report.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary['configuration']!=settings or summary['provenance']['dataset_sha256']!=provenance['dataset_sha256']:
            raise ValueError('Existing trajectory report differs from this gradient study.')
        summary['gradient_validation_file'] = 'gradient_validation.json'
        summary['gradient_gate_passed'] = report['gradient_gate_passed']
        summary['optimization_block_reasons'] = (['This diagnostic tool does not implement an optimization loop.']+
            ([] if report['gradient_gate_passed'] else ['Local model/physical design-gradient certification failed.']))
        write(summary_path,summary)
    print(json.dumps(dict(gradient_gate_passed=report['gradient_gate_passed'])),flush=True)


if __name__=='__main__':
    main()
