"""Experimental design/time trajectory decoder with the official NeRD GPT.

This is a supervised surrogate for canonical supported starts, not an
autoregressive dynamics model or an arbitrary-state simulator. The optional
kinematic architecture derives velocities from the position decoder in time.
No state clipping, contact projection or physical repair is performed.
"""
import math
import torch
from torch import nn
from .learning import load_nerd_backbone,parameters_torch,initial_pose_torch
from .learning_spec import DESIGN_LOWER,DESIGN_UPPER


class MixedDerivativeLayerNorm(nn.Module):
    """Equivalent upstream normalization with reliable mixed derivatives.

    PyTorch 2.9.1 native LayerNorm forward-AD followed by reverse-AD disagrees
    with finite differences for this adapter. Elementary operations preserve
    the same normalization and trainable parameters while passing that check.
    """
    def __init__(self,original):
        super().__init__()
        self.weight,self.bias = original.weight,original.bias

    def forward(self,x):
        centered = x-x.mean(-1,keepdim=True)
        value = centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)*self.weight
        return value if self.bias is None else value+self.bias


class TrajectoryNeRD(nn.Module):
    def __init__(self,nerd_root,base_design,*,horizon=2.,width=96,layers=2,clearance_um=0.,consistent_velocity=False):
        super().__init__()
        if horizon<=0 or width%4:
            raise ValueError('Positive horizon and width divisible by four required.')
        self.base_design,self.horizon,self.clearance_um = base_design,horizon,clearance_um
        self.consistent_velocity = consistent_velocity
        self.backbone,self.upstream = load_nerd_backbone(nerd_root,inputs=26,history=1,width=width,layers=layers)
        if consistent_velocity:
            self.backbone.transformer.ln_f = MixedDerivativeLayerNorm(self.backbone.transformer.ln_f)
            for block in self.backbone.transformer.h:
                block.ln_1,block.ln_2 = MixedDerivativeLayerNorm(block.ln_1),MixedDerivativeLayerNorm(block.ln_2)
        self.head = nn.Linear(width,14)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('offset_mean',torch.zeros(14))
        self.register_buffer('offset_scale',torch.ones(14))

    def physical_designs(self,unit):
        lo,hi = unit.new_tensor(DESIGN_LOWER),unit.new_tensor(DESIGN_UPPER)
        return lo+unit*(hi-lo)

    def initial(self,unit):
        p = parameters_torch(self.physical_designs(unit),self.base_design)
        x = initial_pose_torch(p)
        clearance = x.new_zeros(14);clearance[1] = self.clearance_um*1e-6
        return x+clearance,p

    def time_scale(self,times):
        envelope = -torch.expm1(-times/.002)
        scale = torch.cat([times[:,None],envelope[:,None].expand(-1,13)],-1)
        if self.consistent_velocity:
            # Position offsets and their first time derivatives vanish at zero.
            scale = torch.cat([scale[:,:6]*envelope[:,None],scale[:,6:]],-1)
        return scale

    def forward(self,unit,times):
        if not self.consistent_velocity:
            return self._decode(unit,times)
        # Mathematical SDPA supports forward AD and mixed time/weight/design
        # derivatives. Flash attention does not support these second derivatives.
        unit,times = unit.contiguous(),times.contiguous()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            state,tangent = torch.func.jvp(self._decode,(unit,times),
                (torch.zeros_like(unit),torch.ones_like(times)))
        return torch.cat([state[:,:6],tangent[:,:6],state[:,12:]],-1)

    def _decode(self,unit,times):
        if unit.ndim!=2 or unit.shape[-1]!=2 or times.shape!=unit.shape[:1]:
            raise ValueError('Expected normalized designs (batch,2), times (batch,).')
        initial,p = self.initial(unit)
        t = times
        phase = 2*torch.pi*p[:,22]*t
        clock = torch.stack([t/self.horizon,torch.log1p(t/.01)/math.log1p(self.horizon/.01),
                             torch.exp(-t/.01),torch.exp(-t/.1),torch.exp(-t/.5)],-1)
        discrete = torch.stack([(p[:,22]-365)/5,p[:,26]/20,p[:,41]/20],-1)
        periodic = torch.stack([f(k*phase) for k in range(1,5) for f in [torch.sin,torch.cos]],-1)
        slow = torch.stack([f(2*torch.pi*k*t) for k in [.5,1.,2.,4.] for f in [torch.sin,torch.cos]],-1)
        inputs = torch.cat([2*unit-1,discrete,clock,periodic,slow],-1)
        decoded = self.head(self.backbone(inputs[:,None,:])[:,0])
        return initial+self.time_scale(t)*(self.offset_mean+self.offset_scale*decoded)


def predict_trajectory(model,unit,times,*,chunk=8192):
    """Batched independent queries; gradients are preserved when enabled."""
    b,n = len(unit),len(times)
    d = unit[:,None,:].expand(b,n,2).reshape(-1,2)
    t = times[None,:].expand(b,n).reshape(-1)
    return torch.cat([model(d[k:k+chunk],t[k:k+chunk]) for k in range(0,len(t),chunk)]).reshape(b,n,14)
