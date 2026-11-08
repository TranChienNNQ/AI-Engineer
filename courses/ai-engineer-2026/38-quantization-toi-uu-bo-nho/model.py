import copy
import torch
from torch import nn
from quantization import PackedLinear,fake_activation


class MLP(nn.Module):
    def __init__(self,mean=None,std=None):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(8,32),nn.Linear(32,16),nn.Linear(16,3)])
        self.register_buffer('mean',torch.zeros(8) if mean is None else mean.clone())
        self.register_buffer('std',torch.ones(8) if std is None else std.clone())
    def inputs(self,x):
        x=(x-self.mean)/self.std;values=[]
        for i,layer in enumerate(self.layers):
            values.append(x);x=layer(x)
            if i<2:x=x.relu()
        return values
    def forward(self,x):
        x=(x-self.mean)/self.std
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<2:x=x.relu()
        return x


@torch.no_grad()
def calibrate(model,x):
    scales=[]
    for value in model.inputs(x):
        a=value.abs().max();scales.append(a/127 if a>0 else a.new_tensor(1.))
    return torch.stack(scales)


class QuantMLP(nn.Module):
    def __init__(self,base,bits=4,group_size=None,activation='none',activation_scales=None):
        super().__init__()
        if activation not in ['none','static8','dynamic8']:raise ValueError('activation mode')
        self.layers=nn.ModuleList([PackedLinear(layer,bits,group_size) for layer in base.layers])
        self.register_buffer('mean',base.mean.clone());self.register_buffer('std',base.std.clone())
        scales=(torch.ones(3) if activation_scales is None else activation_scales.clone()) if activation=='static8' else torch.empty(0)
        if activation=='static8' and (scales.shape!=(3,) or not torch.isfinite(scales).all() or (scales<=0).any()):raise ValueError('three positive scales')
        self.register_buffer('activation_scales',scales);self.activation=activation
        self.spec=dict(bits=bits,group_size=group_size,activation=activation,architecture='8-32-16-3-ReLU-v1')
    def forward(self,x,return_stats=False):
        x=(x-self.mean)/self.std;stats=[]
        for i,layer in enumerate(self.layers):
            if self.activation!='none':x,s=fake_activation(x,self.activation,self.activation_scales[i] if self.activation=='static8' else None);stats.append(s)
            x=layer(x)
            if i<2:x=x.relu()
        return (x,stats) if return_stats else x
