"""Check the adapted decoder's initial state and design derivative contract."""
import os
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from walking_robot.design import RobotDesign
from walking_robot.trajectory_nerd import TrajectoryNeRD,predict_trajectory


def model(consistent=False):
    root = Path(os.environ.get('NERD_ROOT',str(Path(__file__).resolve().parents[2]/'neural-robot-dynamics')))
    if not (root/'models/model_transformer.py').is_file():
        pytest.skip('Optional official NeRD checkout unavailable.')
    torch.set_num_threads(2)
    torch.manual_seed(9)
    m = TrajectoryNeRD(root,RobotDesign(profile='matlab'),width=32,layers=1,clearance_um=12.,consistent_velocity=consistent).double().eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.head.weight,std=.02)
        m.offset_scale.copy_(m.offset_scale.new_tensor([.1,.0001,.1,.1,.1,.1,1,1,500,500,500,500,1e-10,1e-10]))
    return m


@pytest.mark.parametrize('consistent',[False,True])
def test_exact_supported_initial_state_and_query_independence(consistent):
    m = model(consistent)
    u = torch.tensor([[.4,.6],[.7,.2]],dtype=torch.float64)
    t = torch.tensor([0.,.037,.2],dtype=torch.float64)
    full = predict_trajectory(m,u,t)
    chunked = predict_trajectory(m,u,t,chunk=1)
    initial,_ = m.initial(u)
    torch.testing.assert_close(full[:,0],initial,rtol=0,atol=0)
    torch.testing.assert_close(full,chunked,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(predict_trajectory(m,u,t.flip(0)).flip(1),full,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('consistent',[False,True])
def test_normalized_design_gradient_includes_geometry_and_matches_fd(consistent):
    m = model(consistent)
    u = torch.tensor([[.4,.6]],dtype=torch.float64,requires_grad=True)
    t = torch.tensor([.1,.2],dtype=torch.float64)
    def value(d):
        x = predict_trajectory(m,d,t)
        return (x[:,1,0]-x[:,0,0])/.1+x[:,1,1]+x[:,1,3]+.01*x[:,1,6]
    gradient, = torch.autograd.grad(value(u).sum(),u)
    assert torch.isfinite(gradient).all() and torch.linalg.vector_norm(gradient)>1e-8
    for axis in range(2):
        plus,minus = u.detach().clone(),u.detach().clone()
        plus[:,axis]+=1e-5;minus[:,axis]-=1e-5
        fd = (value(plus)-value(minus))/(2e-5)
        np.testing.assert_allclose(gradient[:,axis].detach(),fd.detach(),rtol=1e-5,atol=1e-8)


def test_kinematic_decoder_velocity_is_position_time_derivative():
    m = model(True)
    u = torch.tensor([[.4,.6]],dtype=torch.float64)
    t = torch.tensor([.037],dtype=torch.float64)
    x = m(u,t)
    fd = (m(u,t+1e-7)[:,:6]-m(u,t-1e-7)[:,:6])/(2e-7)
    torch.testing.assert_close(x[:,6:12],fd,rtol=1e-5,atol=1e-6)
