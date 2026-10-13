from dataclasses import replace
import numpy as np
import jax.numpy as jnp
import pytest

torch = pytest.importorskip("torch")
from walking_robot.design import DesignRun, RobotDesign
from walking_robot.integrators import dynamics_step
from walking_robot.learning import initial_pose_torch, parameters_torch, neural_rollout
from walking_robot.learning_study import (StudySettings, sample_conditions,
    reference_batch, sequence_loss, ranking_agreement)


def test_variants_preserve_design_splits_and_canonical_initial_conditions():
    settings = StudySettings(design_count=18,trajectories_per_design=5)
    configuration = DesignRun(design=RobotDesign(profile="matlab"))
    data = sample_conditions(settings,configuration)
    assert len(data["designs"]) == 90
    for design_id in np.unique(data["design_id"]):
        ids = data["design_id"]==design_id
        assert len(np.unique(data["split"][ids])) == 1
        assert len(np.unique(data["designs"][ids],axis=0)) == 1
    canonical = data["variant"]==0
    np.testing.assert_array_equal(data["initial_offsets"][canonical],0)
    np.testing.assert_array_equal(data["phase_cycles"][canonical],0)
    assert np.ptp(data["phase_cycles"][~canonical])>.5
    np.testing.assert_array_equal(data["designs"],sample_conditions(settings,configuration)["designs"])


def test_physical_actuator_phase_uses_absolute_time():
    d = torch.tensor([[1.,95.]],dtype=torch.float64)
    p = jnp.asarray(parameters_torch(d,RobotDesign(profile="matlab")).numpy())
    x = jnp.asarray(initial_pose_torch(torch.tensor(np.asarray(p))).numpy())
    start,dt = jnp.array([.0007]),1e-5
    trajectory = reference_batch(x,p,start,dt,2)
    first = dynamics_step(x[0],start[0],p[0],dt=dt)
    second = dynamics_step(first,start[0]+dt,p[0],dt=dt)
    np.testing.assert_allclose(trajectory[0],np.stack([x[0],first,second]),rtol=1e-12,atol=1e-12)
    unshifted = reference_batch(x,p,jnp.zeros(1),dt,2)
    assert np.max(np.abs(np.asarray(trajectory-unshifted)))>1e-3


def test_neural_free_rollout_preserves_explicit_state_and_phase_derivatives():
    class Recorder:
        history = 3
        def predict_next(self,history,times,designs,dt):
            self.last_times = times
            return history[:,-1]+times[:,-1,None]*dt
    model = Recorder()
    initial = torch.arange(14,dtype=torch.float64)[None]
    start = torch.tensor([.003],dtype=torch.float64,requires_grad=True)
    design = torch.tensor([[1.,95.]],dtype=torch.float64)
    trajectory = neural_rollout(model,design,.001,3,initial=initial,start_times=start)
    torch.testing.assert_close(trajectory[:,0],initial)
    torch.testing.assert_close(trajectory[:,-1],initial+(.003+.004+.005)*.001)
    derivative, = torch.autograd.grad(trajectory[:,-1,0].sum(),start)
    torch.testing.assert_close(derivative,start.new_tensor([.003]))


def test_loss_weights_later_steps_equally_and_keeps_contact_error_visible():
    d = torch.tensor([[1.,95.]],dtype=torch.float64)
    p = parameters_torch(d,RobotDesign(profile="matlab"))
    initial = initial_pose_torch(p)
    target = initial[:,None].expand(-1,4,-1).clone()
    def error_at(step):
        prediction = target.clone()
        prediction[:,step,6] += .01
        return sequence_loss(prediction,target,initial,p,torch.ones(8,dtype=torch.float64))
    torch.testing.assert_close(error_at(1),error_at(3))
    assert error_at(3)>0
    floating = target.clone()
    floating[...,1] += .001
    assert sequence_loss(floating,target,initial,p,torch.ones(8,dtype=torch.float64))>90


def test_ranking_rejects_reversed_predictions_and_reports_no_distinguishable_pairs():
    assert ranking_agreement([.1,.2,.3],[3,2,1],.01)["agreement"] == 0
    assert ranking_agreement([.1,.2,.3],[1,2,3],.01)["agreement"] == 1
    assert ranking_agreement([.1,.10001],[1,2],.001)["agreement"] is None


@pytest.mark.parametrize("changes", [dict(trajectories_per_design=1),dict(design_count=19),
    dict(height_um=-1),dict(random_actuator_phase="true"),dict(curriculum=[dict(iterations=1,steps=2000)])])
def test_study_rejects_ambiguous_or_invalid_settings(changes):
    with pytest.raises(ValueError):
        replace(StudySettings(),**changes).validate()
