"""Hand arithmetic and explicit storage assumptions for SFT/LoRA/QLoRA."""
import math
import torch
from data import records,Tokenizer,example,IGNORE
from quantization import quantize,dequantize,unpack


def hand():
    dtype=torch.float64
    w=torch.tensor([[1.,0.,-1.],[0.,2.,1.]],dtype=dtype)
    a=torch.tensor([[1.,0.,1.]],dtype=dtype,requires_grad=True)
    b=torch.tensor([[.2],[-.1]],dtype=dtype,requires_grad=True)
    x=torch.tensor([2.,1.,3.],dtype=dtype);target=torch.tensor([1.,2.],dtype=dtype)
    output=w@x+2*b@(a@x);loss=.5*(output-target).square().sum();loss.backward()
    return dict(base_output=(w@x).tolist(),Ax=(a@x).tolist(),scaled_update=(2*b@(a@x)).detach().tolist(),
                output=output.detach().tolist(),loss=float(loss.detach()),grad_A=a.grad.tolist(),grad_B=b.grad.tolist(),
                params_full=6,params_lora=5,rank=1,alpha=2,scale=2,
                large_matrix=dict(out=4096,inp=4096,rank=8,full_params=4096**2,lora_params=8*(4096+4096)))


def mask_demo():
    rows=records();tok=Tokenizer.fit(rows['train']);r=rows['train'][0];x,y=example(r,tok)
    return dict(prompt=r['prompt'],answer=r['answer'],x=x,y=y,supervised_positions=[i for i,v in enumerate(y) if v!=IGNORE],
                supervised_labels=[v for v in y if v!=IGNORE],uniform_loss=math.log(tok.size),vocab=tok.size)


def quant_demo():
    w=torch.tensor([[-1.,-.5,0.,.5,1.,.2]],dtype=torch.float32)
    packed,scales=quantize(w,block_size=8);restored=dequantize(packed,scales,w.shape,8)
    return dict(kind='local NF4-codebook quantizer, not bitsandbytes QLoRA',weight=w.tolist(),codes=unpack(packed).tolist(),
                packed=packed.tolist(),scales=scales.tolist(),restored=restored.tolist(),
                max_error=float((w-restored).abs().max()),packed_bytes=packed.numel(),scale_bytes=scales.numel()*scales.element_size())


def memory():
    n=7_000_000_000
    return dict(kind='Arithmetic estimates, not measured VRAM or OOM guarantee',parameters=n,
                bf16_weights_bytes=n*2,ideal_4bit_weights_bytes=n//2,
                bf16_weights_gib=n*2/2**30,ideal_4bit_weights_gib=n/2/2**30,
                fp32_adam_training_bytes_per_parameter=16,
                fp32_adam_assumption='weight4 + gradient4 + two FP32 moments8, excluding activations/workspace',
                nf4_scale_overhead_bits_per_weight=32/64,
                ideal_nf4_with_scales_bits=4+32/64,
                ideal_nested_bits=4+8/64+32/(64*256),
                ideal_nested_savings_bits=(4+32/64)-(4+8/64+32/(64*256)))
