from pathlib import Path
import jax
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from walking_robot.expanded_design import ExpandedRobotDesign
from walking_robot.expanded_learning import ExpandedNeRD,expanded_parameters_torch,coordinates
from walking_robot.geometry import point_velocities


@pytest.mark.parametrize('d',[[.5,75.,100.],[2.651,109.08,99.568],[4.,125.,135.]])
def test_three_variable_torch_parameters_match_reference(d):
    base = ExpandedRobotDesign(profile='matlab',drive_frequency_hz=363.)
    expected = base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1],hind_angle_deg=d[2])
    actual = expanded_parameters_torch(torch.tensor([d],dtype=torch.float64),base)[0]
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-10,atol=1e-13)


def test_kinematics_and_mixed_design_derivative():
    root = Path(__file__).resolve().parents[2]/'neural-robot-dynamics'
    if not root.exists():pytest.skip('Optional official NeRD source not installed')
    torch.manual_seed(19)
    base = ExpandedRobotDesign(profile='matlab',drive_frequency_hz=363.)
    model = ExpandedNeRD(root,base,[[.5,4],[70,130],[95,135]],width=32,layers=1).double()
    with torch.no_grad():
        model.offset_mean.copy_(torch.tensor([.8,0,0,0,0,0,0,0]))
        model.offset_scale.fill_(.01)
        model.posture.weight.normal_(0,.02);model.drift[-1].weight.normal_(0,.02)
    u = torch.tensor([[.4,.5,.2]],dtype=torch.float64,requires_grad=True)
    t = torch.tensor([.13],dtype=torch.float64)
    x = model(u,t)
    e = 1e-7
    fd = (model._raw(u,t+e)-model._raw(u,t-e))/(2*e)
    torch.testing.assert_close(x[:,6:12],fd[:,:6],rtol=1e-5,atol=1e-7)
    p = expanded_parameters_torch(model.physical_designs(u),base)
    v = (point_velocities(x,p,xp=torch)['centers']*(p[:,:4]/p[:,:4].sum(-1,keepdim=True))[...,None]).sum(-2)
    def objective(z):
        xx = model(z,t);pp = expanded_parameters_torch(model.physical_designs(z),base)
        return (point_velocities(xx,pp,xp=torch)['centers']*(pp[:,:4]/pp[:,:4].sum(-1,keepdim=True))[...,None]).sum(-2)[0,0]
    grad, = torch.autograd.grad(objective(u),u)
    for k in range(3):
        step = torch.zeros_like(u);step[0,k] = 1e-5
        numerical = (objective(u+step)-objective(u-step))/(2e-5)
        torch.testing.assert_close(grad[0,k],numerical,rtol=1e-4,atol=1e-7)
    initial,_ = model.initial(u)
    torch.testing.assert_close(model(u,torch.zeros_like(t)),initial,rtol=1e-9,atol=1e-12)
    comfd = (coordinates(model(u,t+e),p)-coordinates(model(u,t-e),p))/(2*e)
    torch.testing.assert_close(v,comfd[:,:2],rtol=1e-5,atol=1e-7)
