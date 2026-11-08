"""CPU teaching NF4 codebook/block absmax/nibble packing; not bitsandbytes kernels or full QLoRA."""
import math
import torch
from torch import nn
from torch.nn import functional as F

# NF4 levels from QLoRA paper, Appendix E. Local format/blocks are a teaching implementation.
NF4=torch.tensor([-1.0,-.6961928009986877,-.5250730514526367,-.39491748809814453,
                  -.28444138169288635,-.18477343022823334,-.09105003625154495,0.0,
                  .07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,
                  .44070982933044434,.5626170039176941,.7229568362236023,1.0],dtype=torch.float32)


def pack(codes):
    if codes.dtype!=torch.uint8 or codes.ndim!=1 or codes.numel()%2 or (codes>15).any():raise ValueError('nibbles')
    return codes[::2] | (codes[1::2]<<4)


def unpack(packed):
    if packed.dtype!=torch.uint8 or packed.ndim!=1:raise ValueError('packed dtype/shape')
    return torch.stack([packed&15,packed>>4],dim=1).flatten()


def quantize(weight,block_size=32):
    if type(block_size) is not int or block_size<2 or block_size%2:raise ValueError('even block size')
    if weight.ndim!=2 or not torch.isfinite(weight).all() or not weight.numel():raise ValueError('finite 2D weight')
    flat=weight.detach().float().cpu().flatten();padding=(-flat.numel())%block_size
    blocks=F.pad(flat,(0,padding)).reshape(-1,block_size)
    scales=blocks.abs().amax(1);normalized=blocks/scales.clamp_min(torch.finfo(torch.float32).tiny)[:,None]
    codes=(normalized[:,:,None]-NF4).abs().argmin(-1).to(torch.uint8).flatten()
    return pack(codes),scales


def dequantize(packed,scales,shape,block_size=32):
    codes=unpack(packed).long();codebook=NF4.to(scales.device)
    blocks=codebook[codes].reshape(-1,block_size)*scales[:,None]
    return blocks.flatten()[:math.prod(shape)].reshape(shape)


class NF4Linear(nn.Module):
    def __init__(self,linear,block_size=32):
        super().__init__();self.in_features=linear.in_features;self.out_features=linear.out_features;self.block_size=block_size
        packed,scales=quantize(linear.weight,block_size)
        self.register_buffer('packed',packed);self.register_buffer('scales',scales)
        self.register_buffer('bias',linear.bias.detach().clone() if linear.bias is not None else None)
    def weight_value(self):return dequantize(self.packed,self.scales,(self.out_features,self.in_features),self.block_size)
    def forward(self,x):
        # Base weight is frozen; matmul must keep gradient with respect to x for earlier adapters.
        return F.linear(x,self.weight_value().to(x.dtype),self.bias)
