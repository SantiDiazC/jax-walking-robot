from pathlib import Path
import os
import numpy as np
import pytest
import jax
from walking_robot import calculate_P_from_meta_jax, solve_initial_pose

torch = pytest.importorskip("torch")
from walking_robot.learning import parameters_torch, initial_pose_torch, features, PlanarNeRD, design_objective
from walking_robot.design import RobotDesign


@pytest.mark.parametrize("design", [[.5,90.],[1.,95.],[2.,100.]])
def test_material_and_initial_pose_match_jax(design):
    d = torch.tensor(design,dtype=torch.float64)
    p = parameters_torch(d)
    expected = calculate_P_from_meta_jax(.01,design[0]*1e-6,200.,front_angle_deg=design[1])
    np.testing.assert_allclose(p.numpy(),np.asarray(expected),rtol=1e-12,atol=1e-12)
    pose = initial_pose_torch(p)
    np.testing.assert_allclose(pose.numpy(),jax.jit(solve_initial_pose)(expected),atol=1e-8)


def test_robot_centric_features_ignore_horizontal_translation():
    d = torch.tensor([[1.,95.]],dtype=torch.float64)
    x = initial_pose_torch(parameters_torch(d))[:,None,:].expand(-1,8,-1).clone()
    t = torch.arange(8,dtype=torch.float64)[None,:]*1e-4
    original = features(x,t,d)
    x[...,0] += .123
    np.testing.assert_allclose(features(x,t,d).numpy(),original.numpy(),atol=1e-12)
    assert original.shape == (1,8,41)


@pytest.mark.parametrize("design", [[.5,90.],[1.,95.],[2.,100.]])
def test_yaml_fixed_inputs_and_reconciled_material_map_match_jax(design):
    base = RobotDesign(profile="matlab",voltage_v=150.,damping_ratio=.02,hind_angle_deg=120.,front_leg_offset_mm=2.7)
    d = torch.tensor(design,dtype=torch.float64,requires_grad=True)
    p = parameters_torch(d,base)
    expected = base.parameters(adhesive_thickness_um=design[0],front_angle_deg=design[1])
    np.testing.assert_allclose(p.detach(),np.asarray(expected),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(initial_pose_torch(p).detach(),jax.jit(solve_initial_pose)(expected),atol=1e-8)
    derivative, = torch.autograd.grad(p[21],d)
    assert torch.isfinite(derivative).all() and derivative[0] != 0


@pytest.mark.parametrize("profile,clearance", [("legacy",0.),("matlab",501.)])
def test_neural_design_gradient_matches_finite_difference(profile,clearance):
    root = Path(os.environ.get("NERD_ROOT",str(Path(__file__).resolve().parents[2]/"neural-robot-dynamics")))
    if not (root/"models/model_transformer.py").is_file():
        pytest.skip("Optional official NeRD checkout unavailable; set NERD_ROOT to test its backbone.")
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = PlanarNeRD(root,width=32,layers=1,base_design=RobotDesign(profile=profile),initial_clearance_um=clearance).double().eval()
    # Nonzero head makes this an actual gradient check, not a constant output.
    with torch.no_grad():
        torch.nn.init.normal_(model.head.weight,std=.01)
        model.delta_scale.copy_(torch.tensor([.01,.01,1.,1.,1.,1.,1e-10,1e-10]))
    for p in model.parameters():
        p.requires_grad_(False)
    d = torch.tensor([[1.,95.]],dtype=torch.float64,requires_grad=True)
    objective = design_objective(model,d,1e-4,4).sum()
    derivative, = torch.autograd.grad(objective,d)
    assert torch.isfinite(derivative).all()
    assert torch.linalg.vector_norm(derivative) > 1e-9
    for i in range(2):
        epsilon = 1e-4
        plus,minus = d.detach().clone(),d.detach().clone()
        plus[0,i] += epsilon
        minus[0,i] -= epsilon
        expected = (design_objective(model,plus,1e-4,4)-design_objective(model,minus,1e-4,4))/(2*epsilon)
        np.testing.assert_allclose(derivative[0,i].detach(),expected.detach()[0],rtol=1e-4,atol=1e-9)
