"""Three-variable NeRD adapter with COM coordinates and fixed actuation.

The learned COM horizontal drift uses slow time features. The NeRD GPT learns
periodic posture/vertical motion. Coordinates are converted analytically to
the original state, and velocities are exact time derivatives of positions.
This prevents internal posture errors from becoming spurious horizontal COM
velocity. No state, contact or neural output is clipped/projected.
"""
import math
import torch
from torch import nn
from .learning import parameters_torch,initial_pose_torch,load_nerd_backbone
from .trajectory_nerd import MixedDerivativeLayerNorm
from .geometry import points


def expanded_parameters_torch(designs,base):
    if base.profile!='matlab':
        raise ValueError('The expanded adapter requires the reconciled MATLAB profile.')
    p = parameters_torch(designs[...,:2],base)
    hind = designs[...,2]
    den = torch.sin(torch.deg2rad(186-hind)).square().clamp_min(math.sin(math.pi/360)**2)
    beam = math.ceil(3*(.015*(25e-6)**3/12*4e9)/.0025**3)
    strip = 4e9*(25e-6)**3/12
    plus = 1/(1/beam+.0025**2/(strip*.015/(40e-6*.5)))
    minus = 1/(1/beam+.0025**2/(strip*.015/(40e-6*2)))
    stiffness = torch.ceil((plus+minus)/2/den)
    p = torch.cat([p[...,:16],torch.deg2rad(hind)[...,None],p[...,17:41],stiffness[...,None]],-1)
    if base.drive_frequency_hz is not None:
        p = torch.cat([p[...,:22],torch.full_like(p[...,22:23],base.drive_frequency_hz),p[...,23:]],-1)
    return p


def coordinates(states,parameters):
    g = points(states,parameters,xp=torch)
    weights = parameters[...,:4]/parameters[...,:4].sum(-1,keepdim=True)
    com = (g['centers']*weights[...,None]).sum(-2)
    return torch.cat([com,states[...,2:6],states[...,12:]],-1)


class ExpandedNeRD(nn.Module):
    def __init__(self,nerd_root,base,bounds,*,horizon=2.,width=96,layers=2,clearance_um=0.):
        super().__init__()
        self.base,self.bounds,self.horizon,self.clearance_um = base,bounds,horizon,clearance_um
        self.backbone,self.upstream = load_nerd_backbone(nerd_root,inputs=27,history=1,width=width,layers=layers)
        self.backbone.transformer.ln_f = MixedDerivativeLayerNorm(self.backbone.transformer.ln_f)
        for b in self.backbone.transformer.h:
            b.ln_1,b.ln_2 = MixedDerivativeLayerNorm(b.ln_1),MixedDerivativeLayerNorm(b.ln_2)
        self.posture = nn.Linear(width,7)
        self.drift = nn.Sequential(nn.Linear(19,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.posture.weight);nn.init.zeros_(self.posture.bias)
        nn.init.zeros_(self.drift[-1].weight);nn.init.zeros_(self.drift[-1].bias)
        self.register_buffer('offset_mean',torch.zeros(8))
        self.register_buffer('offset_scale',torch.ones(8))

    def physical_designs(self,unit):
        bounds = unit.new_tensor(self.bounds)
        return bounds[:,0]+unit*(bounds[:,1]-bounds[:,0])

    def initial(self,unit):
        p = expanded_parameters_torch(self.physical_designs(unit),self.base)
        x = initial_pose_torch(p)
        clearance = x.new_zeros(14);clearance[1] = self.clearance_um*1e-6
        return x+clearance,p

    def time_scale(self,t):
        e = -torch.expm1(-t/.002)
        return torch.cat([(t*e)[:,None],e.square()[:,None].expand(-1,5),e[:,None].expand(-1,2)],-1)

    def _raw(self,unit,t):
        x,p = self.initial(unit)
        initial = coordinates(x,p)
        clock = torch.stack([t/self.horizon,torch.log1p(t/.01)/math.log1p(self.horizon/.01),
                             torch.exp(-t/.01),torch.exp(-t/.1),torch.exp(-t/.5)],-1)
        discrete = torch.stack([(p[:,22]-363)/10,p[:,26]/20,p[:,41]/20],-1)
        slow = torch.stack([f(2*torch.pi*k*t) for k in [.5,1.,2.,4.] for f in [torch.sin,torch.cos]],-1)
        phase = 2*torch.pi*p[:,22]*t
        periodic = torch.stack([f(k*phase) for k in range(1,5) for f in [torch.sin,torch.cos]],-1)
        common = torch.cat([2*unit-1,discrete,clock,slow],-1)
        fast = torch.cat([2*unit-1,discrete,clock,periodic,slow],-1)
        output = torch.cat([self.drift(common),self.posture(self.backbone(fast[:,None])[:,0])],-1)
        predicted = initial+self.time_scale(t)*(self.offset_mean+self.offset_scale*output)
        # With x=y=0 the COM position is a pure internal geometry offset.
        raw = torch.cat([torch.zeros_like(predicted[:,:2]),predicted[:,2:6],x[:,6:12],predicted[:,6:]],-1)
        internal = coordinates(raw,p)[:,:2]
        return torch.cat([predicted[:,:2]-internal,predicted[:,2:]],-1)

    def forward(self,unit,t):
        unit,t = unit.contiguous(),t.contiguous()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            q,velocity = torch.func.jvp(self._raw,(unit,t),(torch.zeros_like(unit),torch.ones_like(t)))
        return torch.cat([q[:,:6],velocity[:,:6],q[:,6:]],-1)


def expanded_trajectory(model,unit,times,chunk=4096):
    b,n = len(unit),len(times)
    u = unit[:,None,:].expand(b,n,3).reshape(-1,3)
    t = times[None,:].expand(b,n).reshape(-1)
    return torch.cat([model(u[k:k+chunk],t[k:k+chunk]) for k in range(0,len(t),chunk)]).reshape(b,n,14)
