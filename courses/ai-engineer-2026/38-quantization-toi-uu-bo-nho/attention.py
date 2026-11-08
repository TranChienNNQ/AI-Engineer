"""Quantized K/V diagnostic for one attention operation; not a full KV engine or KIVI."""
import math
import torch
from quantization import quantize_weight,dequantize_weight


def run():
    g=torch.Generator().manual_seed(381);q=torch.randn(2,2,3,8,generator=g);k=torch.randn(2,2,17,8,generator=g);v=torch.randn(2,2,17,8,generator=g)
    def attend(kk,vv):
        p=(q@kk.transpose(-2,-1)/math.sqrt(8)).softmax(-1);return p@vv,p
    ref,pref=attend(k,v);rows=[]
    for bits in [8,4]:
        tensors=[];payload=0
        for tensor in [k,v]:
            data,scales,spec=quantize_weight(tensor.reshape(-1,8),bits,8)
            tensors.append(dequantize_weight(data,scales,spec).reshape_as(tensor));payload+=data.numel()*data.element_size()+scales.numel()*scales.element_size()
        out,p=attend(*tensors);rows.append(dict(bits=bits,packed_payload_bytes=payload,max_error=float((out-ref).abs().max()),
                                             rmse=float(((out-ref)**2).mean().sqrt()),attention_kl=float((pref*(pref.log()-p.log())).sum(-1).mean())))
    return dict(shape=list(k.shape),query_shape=list(q.shape),fp32_kv_bytes=(k.numel()+v.numel())*4,rows=rows,
                scope='per-token/head groups of8, K and V both quantized alike; float32 reconstruction/attention, no causal generation or KIVI implementation')
