"""Extended designs with an explicit fixed drive frequency and traced hind angle.

The original design/parameter interfaces remain unchanged so their recorded
checkpoints and validation fingerprints retain their original meaning.
"""
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
import yaml
from .design import RobotDesign,DesignRun,SimulationSettings,VisualizationSettings
from .parameters import calculate_P_from_meta_jax
from .validation import PhysicalLimits


@dataclass(frozen=True)
class ExpandedRobotDesign(RobotDesign):
    drive_frequency_hz: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.drive_frequency_hz is not None and self.drive_frequency_hz<=0:
            raise ValueError('Drive frequency must be positive.')

    def parameters(self,*,adhesive_thickness_um=None,front_angle_deg=None,hind_angle_deg=None):
        thickness = self.adhesive_thickness_um if adhesive_thickness_um is None else adhesive_thickness_um
        front = self.front_angle_deg if front_angle_deg is None else front_angle_deg
        hind = self.hind_angle_deg if hind_angle_deg is None else hind_angle_deg
        p = calculate_P_from_meta_jax(self.damping_ratio,thickness*1e-6,self.voltage_v,front,hind,profile=self.profile)
        for index,offset in [(17,self.front_leg_offset_mm),(18,self.hind_leg_offset_mm)]:
            if offset is not None:
                p = p.at[index].set(offset*1e-3)
        if self.drive_frequency_hz is not None:
            p = p.at[22].set(self.drive_frequency_hz)
        return p


@dataclass(frozen=True)
class ExpandedDesignRun(DesignRun):
    design: ExpandedRobotDesign = field(default_factory=ExpandedRobotDesign)

    @classmethod
    def from_file(cls,path):
        data = yaml.safe_load(Path(path).read_text())
        constructors = dict(design=ExpandedRobotDesign,simulation=SimulationSettings,
                            visualization=VisualizationSettings,physical_limits=PhysicalLimits)
        if not isinstance(data,dict) or set(data)-set(constructors):
            raise ValueError('Expected design, simulation, visualization and physical_limits sections.')
        return cls(**{key:constructors[key](**value) for key,value in data.items()})


def sample_expanded_designs(settings,base):
    """Deterministic anchors plus stratified samples; thickness sampled in log space."""
    count = settings['sample_count']
    bounds = np.array(settings['bounds'],dtype=float)
    if type(count) is not int or count<18 or bounds.shape!=(3,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError('Need at least 18 samples and increasing bounds for thickness/front/hind.')
    if bounds[0,0]<=0 or min(bounds[1:,0])<=0 or max(bounds[1:,1])>=180:
        raise ValueError('Thickness must be positive and angles strictly between zero and 180.')
    anchors = [[base.adhesive_thickness_um,base.front_angle_deg,base.hind_angle_deg]]
    anchors += [[1.,f,h] for f in [75.,100.,125.] for h in [100.,125.,150.]]
    anchors += [[th,95.,125.] for th in [.75,1.5,2.5]]
    anchors = [d for d in anchors if np.all((np.array(d)>=bounds[:,0])&(np.array(d)<=bounds[:,1]))]
    rng = np.random.default_rng(settings['seed'])
    n = count-len(anchors)
    unit = np.column_stack([(rng.permutation(n)+rng.uniform(size=n))/n for _ in range(3)])
    designs = bounds[:,0]+unit*(bounds[:,1]-bounds[:,0])
    designs[:,0] = np.exp(np.log(bounds[0,0])+unit[:,0]*np.log(bounds[0,1]/bounds[0,0]))
    return np.concatenate([np.array(anchors),designs])
