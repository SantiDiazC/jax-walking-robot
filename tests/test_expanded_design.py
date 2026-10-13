from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from walking_robot.design import RobotDesign
from walking_robot.expanded_design import ExpandedRobotDesign,ExpandedDesignRun,sample_expanded_designs
from walking_robot import solve_initial_pose
from walking_robot.geometry import points


def test_unspecified_drive_frequency_preserves_original_parameter_map():
    a = RobotDesign(profile='matlab')
    b = ExpandedRobotDesign(**asdict(a))
    np.testing.assert_array_equal(a.parameters(),b.parameters())


def test_fixed_actuation_remains_constant_while_all_three_design_variables_change():
    base = ExpandedRobotDesign(profile='matlab',voltage_v=200.,drive_frequency_hz=363.)
    def parameters(d):
        return base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1],hind_angle_deg=d[2])
    a,b = jax.vmap(parameters)(jnp.array([[.5,75.,100.],[12.,125.,150.]]))
    assert a[22]==b[22]==363.
    assert b[0]>a[0] and b[19]!=a[19] and b[16]!=a[16] and b[41]!=a[41]
    jac = jax.jacfwd(parameters)(jnp.array([1.,95.,125.]))
    np.testing.assert_array_equal(jac[22],np.zeros(3))
    assert jac[0,0]>0 and jac[15,1]==pytest.approx(np.pi/180) and jac[16,2]==pytest.approx(np.pi/180)


@pytest.mark.parametrize('front,hind',[(75.,100.),(100.,150.),(125.,125.)])
def test_expanded_supported_pose_still_places_both_feet_on_ground(front,hind):
    p = ExpandedRobotDesign(profile='matlab',front_angle_deg=front,hind_angle_deg=hind,drive_frequency_hz=363.).parameters()
    x = jax.jit(solve_initial_pose)(p)
    np.testing.assert_allclose(points(x,p)['feet'][:,1],0,atol=1e-8)


def test_extended_yaml_roundtrip_and_sampling(tmp_path):
    run = ExpandedDesignRun(design=ExpandedRobotDesign(profile='matlab',drive_frequency_hz=363.))
    path = tmp_path/'design.yaml'
    path.write_text(yaml.safe_dump(asdict(run)))
    loaded = ExpandedDesignRun.from_file(path)
    assert loaded.design==run.design and loaded.simulation==run.simulation and loaded.physical_limits==run.physical_limits
    assert tuple(loaded.visualization.formats)==tuple(run.visualization.formats)
    s = dict(sample_count=96,seed=3,bounds=[[.5,12],[70,130],[95,155]])
    a = sample_expanded_designs(s,run.design)
    np.testing.assert_array_equal(a,sample_expanded_designs(s,run.design))
    assert a.shape==(96,3) and len(np.unique(a,axis=0))==96
    np.testing.assert_array_equal(a[0],[1.,95.,125.])
    assert np.all((a>=np.array(s['bounds'])[:,0])&(a<=np.array(s['bounds'])[:,1]))
