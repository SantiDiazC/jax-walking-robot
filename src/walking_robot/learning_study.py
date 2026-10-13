"""Data and loss helpers for the second, optional NeRD training study.

The architecture and hard-contact reference solver are unchanged. Trajectory
splits are assigned by design, not by windows or randomized initial states.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

from .design import DesignRun
from .geometry import points, point_velocities
from .initial_pose import solve_initial_pose
from .integrators import dynamics_step
from .learning import DESIGN_LOWER, DESIGN_UPPER, STATE_SCALES, parameters_torch
from .validation import inspect_trajectory


@dataclass(frozen=True)
class StudySettings:
    design: str = "../designs/matlab_supported.yaml"
    seed: int = 42
    design_count: int = 36
    trajectories_per_design: int = 4
    horizon_s: float = .2
    neural_timestep_s: float = 1e-4
    generation_batch_size: int = 8
    height_um: float = 30.
    tilt_deg: float = .1
    horizontal_speed_m_s: float = .02
    vertical_speed_m_s: float = .01
    angular_speed_rad_s: float = 2.
    random_actuator_phase: bool = True
    batch_size: int = 64
    learning_rate: float = 3e-4
    validation_every: int = 300
    curriculum: list = field(default_factory=lambda: [dict(iterations=300,steps=1),dict(iterations=300,steps=4),
        dict(iterations=400,steps=16),dict(iterations=500,steps=64),dict(iterations=300,steps=128)])
    max_speed_error_m_s: float = .0025
    max_foot_height_rmse_m: float = 1e-4
    min_ranking_agreement: float = .9
    ranking_tolerance_m_s: float = .001
    gradient_design_count: int = 3
    gradient_unit_epsilons: list = field(default_factory=lambda: [.01,.001])

    @classmethod
    def from_file(cls, path):
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values,dict):
            raise ValueError("Study YAML must contain a mapping.")
        settings = cls(**values)
        settings.validate()
        return settings, DesignRun.from_file(Path(path).resolve().parent/settings.design)

    def validate(self):
        for name in ["seed","design_count","trajectories_per_design","generation_batch_size","batch_size",
                     "validation_every","gradient_design_count"]:
            value = getattr(self,name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a positive integer (seed may be zero).")
        if self.design_count < 18 or self.design_count % 6:
            raise ValueError("Use at least 18 designs and a multiple of six for disjoint splits.")
        if self.trajectories_per_design < 2 or type(self.random_actuator_phase) is not bool:
            raise ValueError("Need at least two trajectories per design and a boolean phase flag.")
        for name in ["horizon_s","neural_timestep_s","learning_rate","max_speed_error_m_s", "max_foot_height_rmse_m",
                     "ranking_tolerance_m_s"]:
            if not np.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ["height_um","tilt_deg","horizontal_speed_m_s","vertical_speed_m_s","angular_speed_rad_s"]:
            if not np.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        count = round(self.horizon_s/self.neural_timestep_s)
        if not np.isclose(count*self.neural_timestep_s,self.horizon_s,rtol=0,atol=1e-12):
            raise ValueError("Horizon must be an integer multiple of neural timestep.")
        if not self.curriculum:
            raise ValueError("Need a nonempty curriculum.")
        for stage in self.curriculum:
            if set(stage) != {"iterations","steps"} or any(type(v) is not int or v<1 for v in stage.values()):
                raise ValueError("Each curriculum stage needs positive integer iterations and steps.")
            if stage["steps"] >= count:
                raise ValueError("Training window must be shorter than the full rollout.")
        if not 0 < self.min_ranking_agreement <= 1 or not self.gradient_unit_epsilons or any(
                not np.isfinite(e) or not 0 < e < .1 for e in self.gradient_unit_epsilons):
            raise ValueError("Invalid ranking threshold or normalized gradient perturbations.")


def write_json(path, value):
    """Preserve failure diagnostics as null, never emit invalid JSON NaNs."""
    def clean(v):
        if isinstance(v,dict):
            return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v,np.integer):
            return int(v)
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+"\n")


def sample_conditions(settings, configuration):
    """Draw designs once, then independent states/phases within each split."""
    rng = np.random.default_rng(settings.seed)
    designs = rng.uniform(DESIGN_LOWER,DESIGN_UPPER,(settings.design_count,2))
    base = configuration.design
    designs[0] = [base.adhesive_thickness_um,base.front_angle_deg]
    if not np.all((designs[0] >= DESIGN_LOWER)&(designs[0] <= DESIGN_UPPER)):
        raise ValueError("Base design must lie inside the experimental design bounds.")
    ids = np.repeat(np.arange(settings.design_count),settings.trajectories_per_design)
    variants = np.tile(np.arange(settings.trajectories_per_design),settings.design_count)
    split_design = np.repeat([0,1,2],[settings.design_count*2//3,settings.design_count//6,settings.design_count//6])
    offsets = np.zeros((len(ids),14))
    offsets[:,1] = rng.uniform(0,settings.height_um*1e-6,len(ids))
    offsets[:,2:6] = np.deg2rad(rng.uniform(-settings.tilt_deg,settings.tilt_deg,len(ids)))[:,None]
    offsets[:,6] = rng.uniform(-settings.horizontal_speed_m_s,settings.horizontal_speed_m_s,len(ids))
    offsets[:,7] = rng.uniform(-settings.vertical_speed_m_s,settings.vertical_speed_m_s,len(ids))
    offsets[:,8:12] = rng.uniform(-settings.angular_speed_rad_s,settings.angular_speed_rad_s,(len(ids),4))
    phases = rng.uniform(0,1,len(ids)) if settings.random_actuator_phase else np.zeros(len(ids))
    offsets[variants==0] = 0
    phases[variants==0] = 0
    return dict(designs=designs[ids],design_id=ids,variant=variants,split=split_design[ids],
                initial_offsets=offsets,phase_cycles=phases)


def reference_batch(initial, parameters, start_times, dt, steps):
    """Elapsed trajectory times, absolute times for periodic physical forcing."""
    def single(x,p,t0):
        def step(state,i):
            nxt = dynamics_step(state,t0+i*dt,p,dt=dt)
            return nxt,nxt
        _,tail = jax.lax.scan(step,x,jnp.arange(steps))
        return jnp.concatenate([x[None],tail])
    return jax.vmap(single)(initial,parameters,start_times)


def generate_dataset(settings, configuration, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    ratio = settings.neural_timestep_s/configuration.simulation.timestep_s
    substeps = round(ratio)
    if substeps < 1 or not np.isclose(ratio,substeps,rtol=0,atol=1e-10):
        raise ValueError("Neural timestep must be an integer multiple of physical timestep.")
    sources = Path(__file__).resolve().parent
    fingerprint = dict(settings=asdict(settings),configuration=asdict(configuration),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources.glob("*.py"))})
    fingerprint = json.loads(json.dumps(fingerprint))
    cache,report_file = output/"dataset.npz",output/"dataset_report.json"
    if cache.exists() and report_file.exists():
        report = json.loads(report_file.read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError("Dataset configuration/source mismatch; use a new output directory.")
        with np.load(cache,allow_pickle=False) as saved:
            return {k:saved[k] for k in saved.files},report
    data = sample_conditions(settings,configuration)
    base = configuration.design
    p = jax.jit(jax.vmap(lambda d:base.parameters(adhesive_thickness_um=d[0],front_angle_deg=d[1])))(jnp.asarray(data["designs"]))
    initial = jax.jit(jax.vmap(solve_initial_pose))(p)
    initial = initial.at[:,1].add(configuration.simulation.initial_clearance_um*1e-6)+jnp.asarray(data["initial_offsets"])
    start_times = jnp.asarray(data["phase_cycles"])/p[:,22]
    count = round(settings.horizon_s/configuration.simulation.timestep_s)
    times = np.arange(count+1)*configuration.simulation.timestep_s
    run = jax.jit(lambda x,p,t:reference_batch(x,p,t,configuration.simulation.timestep_s,count))
    saved_states,reports = [],[]
    for begin in range(0,len(initial),settings.generation_batch_size):
        end = begin+settings.generation_batch_size
        states = np.asarray(run(initial[begin:end],p[begin:end],start_times[begin:end]))
        saved_states.extend(states[:,::substeps])
        reports.extend(inspect_trajectory(times,x,np.asarray(pp),configuration.physical_limits)
                       for x,pp in zip(states,p[begin:end]))
        print(f"Physical data: {min(end,len(initial))}/{len(initial)} trajectories",flush=True)
    data.update(states=np.asarray(saved_states),parameters=np.asarray(p),start_times=np.asarray(start_times),
                times=times[::substeps],passed=np.asarray([r["passed"] for r in reports]))
    report = dict(fingerprint=fingerprint,full_horizon_pass_count=int(data["passed"].sum()),
        total_trajectories=len(reports),training_policy="Only complete physically admissible training trajectories.",
        split_policy="Disjoint design IDs across train/validation/test; all variants stay with their design.",
        trajectories=[dict(id=i,design_id=int(data["design_id"][i]),variant=int(data["variant"][i]),
                           split=int(data["split"][i]),validation=r) for i,r in enumerate(reports)])
    np.savez_compressed(cache,**data)
    write_json(report_file,report)
    return data,report


def trajectory_observables(states,parameters,xp=np):
    """COM position/velocity and both signed foot heights, in SI units."""
    p = parameters[:,None,:]
    g,v = points(states,p,xp=xp),point_velocities(states,p,xp=xp)
    if xp is torch:
        weights = p[...,:4]/p[...,:4].sum(-1,keepdim=True)
        com = (g["centers"]*weights[...,None]).sum(-2)
        velocity = (v["centers"]*weights[...,None]).sum(-2)
    else:
        weights = p[...,:4]/p[...,:4].sum(-1,keepdims=True)
        com = (g["centers"]*weights[...,None]).sum(-2)
        velocity = (v["centers"]*weights[...,None]).sum(-2)
    return com,velocity,g["feet"][...,1]


def sequence_loss(predicted, targets, initial, parameters, delta_scale):
    """Equally weighted rollout supervision plus a normalized first-step loss.

    No clipping, state projection or hidden repair. COM and signed foot heights
    expose floating trajectories that can pass broad physical screening limits.
    """
    scale = predicted.new_tensor(STATE_SCALES)
    state_loss = ((predicted-targets)/scale).square().mean()
    first_step = ((predicted[:,0,6:]-targets[:,0,6:])/delta_scale).square().mean()
    pc,pv,pf = trajectory_observables(predicted,parameters,xp=torch)
    tc,tv,tf = trajectory_observables(targets,parameters,xp=torch)
    origin,_,_ = trajectory_observables(initial[:,None],parameters,xp=torch)
    # Relative displacement makes this loss invariant under horizontal translation.
    displacement = (((pc-origin)-(tc-origin))/.0005).square().mean()
    speed = ((pv-tv)/.1).square().mean()
    feet = ((pf-tf)/.0001).square().mean()
    return .25*first_step+state_loss+displacement+speed+feet


def ranking_agreement(reference,predicted,tolerance):
    """Compare design ordering, ignoring physically indistinguishable pairs."""
    reference,predicted = np.asarray(reference),np.asarray(predicted)
    eligible = correct = 0
    for i in range(len(reference)):
        for j in range(i):
            gap = reference[i]-reference[j]
            if np.isfinite(gap) and abs(gap)>tolerance:
                eligible += 1
                correct += int(np.isfinite(predicted[[i,j]]).all() and gap*(predicted[i]-predicted[j])>0)
    return dict(eligible_pairs=eligible,correct_pairs=correct,agreement=correct/eligible if eligible else None)
