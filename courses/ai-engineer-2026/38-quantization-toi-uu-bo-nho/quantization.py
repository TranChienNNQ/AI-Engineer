"""CPU teaching format: narrow symmetric INT8 or packed offset-INT4, FP32 scales.
Not ONNX/GPTQ/AWQ/NF4 or a fused integer kernel. Dequantized FP32 compute.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
FORMAT='symmetric-narrow-offset-nibble-v1'


def pack_nibbles(codes):
    if codes.dtype!=torch.uint8 or codes.ndim!=1 or (codes>15).any():raise ValueError('uint8 nibbles')
    if codes.numel()%2:codes=F.pad(codes,(0,1))
    return codes[::2] | (codes[1::2]<<4)


def unpack_nibbles(packed,count):
    if packed.dtype!=torch.uint8 or packed.ndim!=1 or type(count) is not int or count<0 or packed.numel()!=(count+1)//2:raise ValueError('packed length/dtype')
    if count%2 and packed.numel() and int(packed[-1])>>4:raise ValueError('nonzero tail nibble')
    return torch.stack([packed&15,packed>>4],1).flatten()[:count]


def affine(x,lo,hi,bits=4):
    if bits not in [4,8] or not math.isfinite(lo) or not math.isfinite(hi) or not lo<=0<=hi or hi<=lo or not torch.isfinite(x).all():raise ValueError('finite range including zero')
    qmax=2**bits-1;s=(hi-lo)/qmax;z=max(0,min(qmax,round(-lo/s)))
    q=(torch.round(x/s)+z).clamp(0,qmax).to(torch.uint8)
    return q,(q.float()-z)*s,dict(scale=s,zero_point=z,qmin=0,qmax=qmax)


def quantize_weight(weight,bits=4,group_size=None):
    if weight.ndim!=2 or not weight.numel() or weight.device.type!='cpu' or not torch.isfinite(weight).all():raise ValueError('nonempty finite CPU matrix')
    if type(bits) is not int or bits not in [4,8]:raise ValueError('bits4/8')
    rows,cols=weight.shape;g=cols if group_size is None else group_size
    if type(g) is not int or g<1:raise ValueError('positive group size')
    groups=(cols+g-1)//g;w=F.pad(weight.detach().float(),(0,groups*g-cols)).reshape(rows,groups,g)
    qmax=2**(bits-1)-1;amax=w.abs().amax(-1);scales=torch.where(amax==0,torch.ones_like(amax),amax/qmax)
    q=torch.round(w/scales[:,:,None]).clamp(-qmax,qmax).to(torch.int8).flatten()
    data=pack_nibbles((q.to(torch.int16)+8).to(torch.uint8)) if bits==4 else q
    return data,scales,dict(rows=rows,cols=cols,bits=bits,group_size=g,format=FORMAT)


def dequantize_weight(data,scales,spec):
    if spec.get('format')!=FORMAT or spec.get('bits') not in [4,8]:raise ValueError('format')
    rows,cols,g=[spec[k] for k in ['rows','cols','group_size']]
    if any(type(v) is not int or v<1 for v in [rows,cols,g]):raise ValueError('shape metadata')
    groups=(cols+g-1)//g;count=rows*groups*g
    if scales.shape!=(rows,groups) or scales.dtype!=torch.float32 or not torch.isfinite(scales).all() or (scales<=0).any():raise ValueError('scales')
    if spec['bits']==4:
        codes=unpack_nibbles(data,count)
        if (codes==0).any():raise ValueError('reserved narrow-range code')
        q=codes.to(torch.int16)-8
    else:
        if data.dtype!=torch.int8 or data.ndim!=1 or data.numel()!=count or (data==-128).any():raise ValueError('int8 data')
        q=data
    return (q.float().reshape(rows,groups,g)*scales[:,:,None]).reshape(rows,groups*g)[:,:cols]


class PackedLinear(nn.Module):
    def __init__(self,linear,bits=4,group_size=None):
        super().__init__();self.in_features=linear.in_features;self.out_features=linear.out_features
        data,scales,self.spec=quantize_weight(linear.weight,bits,group_size)
        self.register_buffer('data',data);self.register_buffer('scales',scales)
        self.register_buffer('bias',linear.bias.detach().clone() if linear.bias is not None else None)
    def forward(self,x):return F.linear(x,dequantize_weight(self.data,self.scales,self.spec),self.bias)


def fake_activation(x,mode='dynamic8',scale=None):
    if mode not in ['dynamic8','static8'] or not torch.isfinite(x).all():raise ValueError('activation mode/finite')
    if mode=='dynamic8':
        a=x.abs().amax(-1,keepdim=True);scale=torch.where(a==0,torch.ones_like(a),a/127)
    elif scale is None or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('static activation scale')
    q=torch.round(x/scale).clamp(-127,127)
    clipped=int((x.abs()>127*scale).sum()) if mode=='static8' else 0
    return q*scale,dict(clipped=clipped,total=x.numel())


def tensor_bytes(module):return sum(v.numel()*v.element_size() for v in module.state_dict().values())
