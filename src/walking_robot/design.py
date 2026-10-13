"""Editable physical designs and settings for a complete simulation run."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import yaml
from .parameters import calculate_P_from_meta_jax
from .validation import PhysicalLimits


@dataclass(frozen=True)
class RobotDesign:
    profile: str = "legacy"
    damping_ratio: float = .01
    adhesive_thickness_um: float = 1.
    voltage_v: float = 200.
    front_angle_deg: float = 95.
    hind_angle_deg: float = 125.
    front_leg_offset_mm: float | None = None
    hind_leg_offset_mm: float | None = None

    def __post_init__(self):
        if self.profile not in {"legacy", "matlab"}:
            raise ValueError("Design profile must be legacy or matlab.")
        if any(not np.isfinite(value) for key, value in asdict(self).items() if key != "profile" and value is not None):
            raise ValueError("Design values must be finite.")
        if self.damping_ratio < 0 or self.adhesive_thickness_um < 0:
            raise ValueError("Damping and adhesive thickness must be nonnegative.")
        if any(not 0 < angle < 180 for angle in [self.front_angle_deg, self.hind_angle_deg]):
            raise ValueError("Leg angles must be between 0 and 180 degrees.")
        if any(not 0 <= offset <= 6.25 for offset in [self.front_leg_offset_mm, self.hind_leg_offset_mm] if offset is not None):
            raise ValueError("Leg offsets must lie within their 6.25 mm body segments.")

    def parameters(self, *, adhesive_thickness_um=None, front_angle_deg=None):
        """Map this design, optionally substituting traced optimization inputs."""
        thickness = self.adhesive_thickness_um if adhesive_thickness_um is None else adhesive_thickness_um
        front = self.front_angle_deg if front_angle_deg is None else front_angle_deg
        p = calculate_P_from_meta_jax(self.damping_ratio, thickness*1e-6,
                                     self.voltage_v, front, self.hind_angle_deg, profile=self.profile)
        for index, offset in [(17, self.front_leg_offset_mm), (18, self.hind_leg_offset_mm)]:
            if offset is not None:
                p = p.at[index].set(offset*1e-3)
        return p


@dataclass(frozen=True)
class SimulationSettings:
    duration_s: float = .1
    timestep_s: float = 1e-5
    initial_clearance_um: float = 0.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in [self.duration_s, self.timestep_s]):
            raise ValueError("Duration and timestep must be finite and positive.")
        if not np.isfinite(self.initial_clearance_um) or self.initial_clearance_um < 0:
            raise ValueError("Initial clearance must be finite and nonnegative.")
        if self.steps < 1 or not np.isclose(self.steps*self.timestep_s, self.duration_s, rtol=0, atol=1e-12):
            raise ValueError("Duration must be a positive integer multiple of timestep.")

    @property
    def steps(self):
        return round(self.duration_s/self.timestep_s)


@dataclass(frozen=True)
class VisualizationSettings:
    formats: tuple = ("gif", "html")
    fps: int = 30
    playback_speed: float = .01  # 0.01 = 100x slow motion before any frame cap.
    max_frames: int = 600
    stop_at_violation: bool = False

    def __post_init__(self):
        if isinstance(self.formats, str) or not self.formats or not set(self.formats) <= {"html", "gif", "mp4"}:
            raise ValueError("Animation formats must be a nonempty list of html, gif or mp4.")
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps < 1:
            raise ValueError("fps must be a positive integer.")
        if not np.isfinite(self.playback_speed) or self.playback_speed <= 0:
            raise ValueError("Playback speed must be finite and positive.")
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames < 2:
            raise ValueError("max_frames must be an integer >= 2.")
        if not isinstance(self.stop_at_violation, bool):
            raise ValueError("stop_at_violation must be true or false.")


@dataclass(frozen=True)
class DesignRun:
    design: RobotDesign = field(default_factory=RobotDesign)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    physical_limits: PhysicalLimits = field(default_factory=PhysicalLimits)

    @classmethod
    def from_file(cls, filename):
        try:
            data = yaml.safe_load(Path(filename).read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML: {error}") from error
        constructors = dict(design=RobotDesign, simulation=SimulationSettings,
                            visualization=VisualizationSettings, physical_limits=PhysicalLimits)
        if not isinstance(data, dict) or set(data)-set(constructors):
            raise ValueError("Expected design, simulation, visualization and physical_limits sections.")
        try:
            return cls(**{key:constructors[key](**value) for key, value in data.items()})
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid design configuration: {error}") from error
