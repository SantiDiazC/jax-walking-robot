"""Experimental planar adapter using the official NeRD GPT backbone.

This module is optional (PyTorch). It does not use the Warp inference wrapper,
so design derivatives survive through geometry, actuation and predicted states.
The upstream source remains in its own checkout and is loaded explicitly.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn
from .geometry import points, point_velocities
from .validation import PhysicalLimits
from .design import RobotDesign

from .learning_spec import DESIGN_LOWER, DESIGN_UPPER, STATE_SCALES


def parameters_torch(design, base_design=None):
    """Differentiable thickness/angle map, with other inputs from a YAML design."""
    base = base_design or RobotDesign()
    had, front = design[..., 0]*1e-6, design[..., 1]
    zero = torch.zeros_like(had)
    constant = lambda v: zero+v
    w, length, seg = .015, .025, .00625
    ha, hp, ya, yp, yad = 80e-6, 75e-6, 2.8e9, 4e9, 61e6
    rho = w*(1853*ha+1461*hp+1000*had)
    ybar = (ya*w*ha*(hp+had+ha/2)+yad*w*had*(hp+had/2)+yp*w*hp*(hp/2))/(ya*w*ha+yad*w*had+yp*w*hp)
    inertia_a = w*ha**3/12+w*ha*(hp+had+ha/2-ybar)**2
    inertia_ad = w*had**3/12+w*had*(hp+had/2-ybar)**2
    inertia_p = w*hp**3/12+w*hp*(hp/2-ybar)**2
    ei = ya*inertia_a+yad*inertia_ad+yp*inertia_p
    omega = 4.73**2*torch.sqrt(ei/(rho*length**4))
    frequency = torch.ceil(22.373/length**2*torch.sqrt(ei/rho)/(2*torch.pi))
    mass = rho*seg
    inertia = mass*seg**2/12
    spring = ei/seg
    damping = 2*base.damping_ratio*spring/omega
    torque = .5*(w*yp*(-22e-12)*base.voltage_v/hp*(2*ybar*hp-2*hp*(hp+ha+had)+hp**2))
    if base.profile == "matlab":
        torque = .5*(w*ya*(-22e-12)*base.voltage_v/ha*(2*ybar*ha-2*ha*(ha+had+hp)+ha**2))
    # Match the existing scalar front/hind averaged ground stiffness model.
    strip = yp*(25e-6)**3/12
    beam = 3*(w*(25e-6)**3/12*yp)/(.0025**3)
    if base.profile == "matlab":
        beam = float(math.ceil(beam))

    def ground(angle, fold):
        den = torch.sin(torch.deg2rad(180-angle+6))**2
        den = torch.clamp_min(den, torch.sin(constant(torch.pi/360))**2)
        stiff_plus = 1/(1/beam+.0025**2/(strip*w/(fold*.5)))/den
        stiff_minus = 1/(1/beam+.0025**2/(strip*w/(fold*2)))/den
        return torch.ceil((stiff_plus+stiff_minus)/2)

    kyf, kyh = ground(front,65e-6), ground(constant(base.hind_angle_deg),40e-6)
    ky = kyf if base.profile == "matlab" else .5*(kyf+kyh)
    front_offset = base.front_leg_offset_mm
    hind_offset = base.hind_leg_offset_mm
    if front_offset is None:
        front_offset = 2.8 if base.profile == "matlab" else .25
    if hind_offset is None:
        hind_offset = 2.5 if base.profile == "matlab" else .25
    values = [mass]*4+[inertia]*4+[constant(9.81)]+[constant(seg)]*4
    values += [constant(.0025),constant(.0025),torch.deg2rad(front),constant(torch.pi*base.hind_angle_deg/180),constant(front_offset*1e-3),constant(hind_offset*1e-3)]
    values += [spring,damping,torque,frequency]+[constant(torch.pi*4/180)]*3
    values += [ky]+[constant(v) for v in [8e-5,.03,.4,.03,.8,1e-3,.05,2.,.2,.1,200.,1000.,1e-4,3e-3]]
    values += [kyh if base.profile == "matlab" else zero]
    return torch.stack(values, dim=-1)


def initial_pose_torch(parameters):
    """Analytic pose for the small-angle design domain; both feet at y=0.

    The relative foot vector rotates rigidly with the base angle. Select the
    branch with base angle in (-pi/2, pi/2). Tests compare it to the JAX solver.
    """
    p = parameters
    zero = torch.zeros_like(p[..., 0])
    relative = torch.stack([zero,p[...,23],p[...,23]+p[...,24],p[...,23]+p[...,24]+p[...,25]],dim=-1)
    raw = torch.cat([torch.stack([zero,zero],-1),relative,torch.zeros((*p.shape[:-1],8),device=p.device,dtype=p.dtype)],dim=-1)
    feet = points(raw,p,xp=torch)["feet"]
    difference = feet[...,1,:]-feet[...,0,:]
    angle = torch.atan(-difference[...,1]/difference[...,0])
    angles = relative+angle[...,None]
    pose = torch.cat([torch.stack([zero,zero],-1),angles,raw[...,6:]],dim=-1)
    front = points(pose,p,xp=torch)["feet"][...,0,:]
    return torch.cat([-front,pose[...,2:]],dim=-1)


def features(states, times, designs, base_design=None):
    """Robot-centric state, recomputed contact and torque features; shape (B,T,41)."""
    p = parameters_torch(designs, base_design)[:,None,:]
    g, velocities = points(states,p,xp=torch), point_velocities(states,p,xp=torch)
    theta = states[...,3]
    c, s = torch.cos(theta),torch.sin(theta)

    def rotate(v):
        return torch.stack([c[...,None]*v[...,0]+s[...,None]*v[...,1],
                            -s[...,None]*v[...,0]+c[...,None]*v[...,1]],dim=-1)

    centers = g["centers"]
    vb = rotate(velocities["centers"])[...,1,:]
    base = torch.cat([centers[...,1,1:2]/.0025,
        (states[...,[2,4,5]]-theta[...,None])/.2,
        vb, states[...,9:10]/500,
        (states[...,[8,10,11]]-states[...,9:10])/500,states[...,12:]/1e-7],dim=-1)
    gravity = torch.stack([-s,-c],dim=-1)
    feet = g["feet"]
    relative = feet-centers[...,1:2,:]
    closest = torch.stack([feet[...,0],torch.zeros_like(feet[...,1])],dim=-1)-centers[...,1:2,:]
    normal = torch.stack([s,c],dim=-1)[...,None,:].expand_as(feet)
    gap = feet[...,1:2]
    # Tangential/normal foot velocities are world x/y for the flat ground.
    contact = torch.cat([rotate(relative)/.0025,rotate(closest)/.0025,normal,
                         gap/.0025,velocities["feet"],torch.sigmoid(-gap/2e-5)],dim=-1).flatten(-2)
    phase = 2*torch.pi*p[...,22]*times
    drive = p[...,21]*torch.sin(phase)
    torque = p[...,19:20]*(torch.diff(states[...,2:6],dim=-1)-p[...,23:26])+p[...,20:21]*torch.diff(states[...,8:12],dim=-1)+drive[...,None]
    lo,hi = designs.new_tensor(DESIGN_LOWER),designs.new_tensor(DESIGN_UPPER)
    d = (2*(designs-lo)/(hi-lo)-1)[:,None,:].expand(-1,states.shape[1],-1)
    return torch.cat([base,gravity,contact,torque/1e-5,torch.stack([torch.sin(phase),torch.cos(phase)],-1),d],dim=-1)


def load_nerd_backbone(nerd_root, *, inputs, history, width, layers):
    """Load the upstream GPT without copying or modifying its source."""
    source = Path(nerd_root).resolve()/"models/model_transformer.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official NeRD GPT source not found: {source}")
    module_name = "walking_robot_external_nerd_gpt"
    spec = importlib.util.spec_from_file_location(module_name,source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    upstream = {"source":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "project":"https://github.com/NVlabs/neural-robot-dynamics"}
    backbone = module.GPT(module.GPTConfig(block_size=history,vocab_size=inputs,n_layer=layers,
        n_head=4,n_embd=width,dropout=0.,bias=True))
    return backbone,upstream


class PlanarNeRD(nn.Module):
    def __init__(self, nerd_root, history=8, width=64, layers=2, *, base_design=None, initial_clearance_um=0.):
        super().__init__()
        self.backbone,self.upstream = load_nerd_backbone(nerd_root,inputs=41,history=history,width=width,layers=layers)
        self.history = history
        self.base_design = base_design or RobotDesign()
        self.initial_clearance_um = initial_clearance_um
        self.head = nn.Linear(width,8)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("delta_mean",torch.zeros(8))
        self.register_buffer("delta_scale",torch.ones(8))

    def predict_next(self, history, times, designs, dt):
        embedding = self.backbone(features(history,times,designs,self.base_design))
        delta = self.head(embedding[:,-1,:])*self.delta_scale+self.delta_mean
        last = history[:,-1,:]
        velocity = last[...,6:12]+delta[...,:6]
        q = last[...,:6]+.5*dt*(last[...,6:12]+velocity)
        return torch.cat([q,velocity,last[...,12:]+delta[...,6:]],dim=-1)


def neural_rollout(model, designs, dt, steps, *, initial=None, start_times=None):
    """Free rollout, optionally from measured states and absolute actuator times.

    Defaults retain the design-dependent supported pose. Explicit initial states
    are used verbatim; callers checking design derivatives must rebuild them for
    each design rather than freezing their geometry dependence.
    """
    if initial is None:
        initial = initial_pose_torch(parameters_torch(designs,model.base_design))
        clearance = initial.new_zeros(14)
        clearance[1] = model.initial_clearance_um*1e-6
        initial = initial+clearance
    if initial.shape != (len(designs),14):
        raise ValueError("Initial states must have shape (batch,14).")
    start_times = designs.new_zeros(len(designs)) if start_times is None else start_times
    if start_times.shape != (len(designs),):
        raise ValueError("Start times must have shape (batch,).")
    history = initial[:,None,:].expand(-1,model.history,-1)
    times = start_times[:,None].expand(-1,model.history)
    result = [initial]
    for i in range(steps):
        nxt = model.predict_next(history,times,designs,dt)
        history = torch.cat([history[:,1:,:],nxt[:,None,:]],dim=1)
        times = torch.cat([times[:,1:],start_times[:,None]+(i+1)*dt],dim=1)
        result.append(nxt)
    return torch.stack(result,dim=1)


def design_objective(model, designs, dt, steps, limits=None):
    """Maximize COM displacement rate, penalizing predicted limit violations."""
    states = neural_rollout(model,designs,dt,steps)
    limits = limits or PhysicalLimits()
    p = parameters_torch(designs,model.base_design)[:,None,:]
    g = points(states,p,xp=torch)
    com = (g["centers"]*p[...,:4,None]).sum(dim=-2)/p[...,:4].sum(dim=-1,keepdim=True)
    begin = steps//2
    speed = (com[:,-1,0]-com[:,begin,0])/((steps-begin)*dt)
    velocities = point_velocities(states,p,xp=torch)
    quantities = [( -g["feet"][...,1], limits.max_foot_penetration_m),
                  ( -g["body"][...,1], limits.max_body_penetration_m),
                  ( g["body"][...,1], limits.max_body_height_m),
                  ( torch.linalg.vector_norm(velocities["centers"],dim=-1), limits.max_center_speed_m_s),
                  ( torch.linalg.vector_norm(velocities["feet"],dim=-1), limits.max_foot_speed_m_s),
                  ( torch.abs(states[...,8:12]), limits.max_angular_speed_rad_s),
                  ( torch.abs(torch.diff(states[...,2:6],dim=-1)), limits.max_relative_angle_rad)]
    penalty = sum((torch.relu(value-limit)/limit).square().mean((1,2)) for value,limit in quantities)
    return -speed+10*penalty
