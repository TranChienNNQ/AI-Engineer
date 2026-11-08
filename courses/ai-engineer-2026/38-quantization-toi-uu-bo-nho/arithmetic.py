import math
import torch
from quantization import affine,quantize_weight,dequantize_weight


def plan():
    # Exact toy ledger, hypothetical fully supported payloads; no capacity guarantee.
    n=7_000_000_000;quantized=n*98//100;rest=n-quantized;group=64
    codes=quantized//2;scales=(quantized+group-1)//group*2;other_weights=rest*2
    weights4=codes+scales+other_weights;weights16=n*2;kv_per_request=2*32*8*128*4096*2
    budget=16*1024**3;workspace=2*1024**3
    return dict(parameters=n,quantized=quantized,fp16_parameters=rest,group_size=group,scale_bytes=2,
                int4_codes_bytes=codes,scales_bytes=scales,unquantized_bytes=other_weights,
                mixed_weight_bytes=weights4,fp16_weight_bytes=weights16,effective_bits=weights4*8/n,
                kv_per_request_bytes=kv_per_request,budget_bytes=budget,other_reserved_bytes=workspace,
                payload_ceiling_fp16=max(0,(budget-workspace-weights16)//kv_per_request),
                payload_ceiling_mixed4=max(0,(budget-workspace-weights4)//kv_per_request),
                scope='hypothetical ledger; no row padding/alignment, temporary dequant tensors, real allocator or hardware measurement')


def hand():
    w=torch.tensor([[-3.5,-1.2,0.,.7,2.4,3.5]])
    data,scales,spec=quantize_weight(w,4)
    reconstructed=dequantize_weight(data,scales,spec)
    x=torch.tensor([-1.,-.35,0.,.65,2.]);codes,rec,meta=affine(x,-1.,2.,4)
    a=torch.tensor([1.,-2.,3.]);b=torch.tensor([.5,.25,-1.]);sx=.5;sw=.25
    qx=torch.round(a/sx).to(torch.int32);qw=torch.round(b/sw).to(torch.int32);acc=int((qx*qw).sum())
    return dict(symmetric=dict(input=w.tolist()[0],scale=scales.item(),signed_codes=(torch.round(w/scales.item())).int().tolist()[0],
                               packed=data.tolist(),reconstructed=reconstructed.tolist()[0],mse=((w-reconstructed)**2).mean().item(),max_error=(w-reconstructed).abs().max().item()),
                affine=dict(input=x.tolist(),codes=codes.tolist(),reconstructed=rec.tolist(),**meta),
                dot=dict(qx=qx.tolist(),qw=qw.tolist(),int32_accumulator=acc,dequantized=acc*sx*sw,float_dot=float(a@b)),memory=plan())
