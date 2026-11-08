"""Check local representation and quantization semantics with independent NumPy loops."""
import copy
import math
import numpy as np
import torch
from torch import nn
from quantization import pack_nibbles,unpack_nibbles,quantize_weight,dequantize_weight,affine,fake_activation,PackedLinear,tensor_bytes
from model import MLP,QuantMLP,calibrate
from data import dataset,calibration_rows
from arithmetic import hand,plan


def run():
    torch.set_num_threads(1);names=[]
    def check(name,ok):
        if not bool(ok):raise AssertionError(name)
        names.append(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True);return
        raise AssertionError(name+' accepted')
    h=hand();check('hand scale and packing',h['symmetric']['scale']==.5 and h['symmetric']['packed']==[0x61,0x98,0xfd])
    check('hand symmetric q',h['symmetric']['signed_codes']==[-7,-2,0,1,5,7])
    check('hand error',abs(h['symmetric']['mse']-.015)<1e-7 and abs(h['symmetric']['max_error']-.2)<1e-6)
    check('affine hand',h['affine']['codes']==[0,3,5,8,15] and h['affine']['zero_point']==5)
    check('integer accumulator',h['dot']['int32_accumulator']==-24 and h['dot']['float_dot']==h['dot']['dequantized']==-3.)
    check('ties to even',torch.round(torch.tensor([.5,1.5,2.5,-.5,-1.5])).tolist()==[0,2,2,0,-2])
    for size in [0,1,2,3,15,16,17]:
        codes=(torch.arange(size)%16).to(torch.uint8);p=pack_nibbles(codes)
        check(f'nibble roundtrip n{size}',torch.equal(unpack_nibbles(p,size),codes) and len(p)==math.ceil(size/2))
    rejects('invalid nibble range',lambda:pack_nibbles(torch.tensor([16],dtype=torch.uint8)))
    rejects('invalid nibble dtype',lambda:pack_nibbles(torch.tensor([1])))
    rejects('invalid packed length',lambda:unpack_nibbles(torch.tensor([1],dtype=torch.uint8),3))
    rejects('tail nibble rejected',lambda:unpack_nibbles(torch.tensor([0x11],dtype=torch.uint8),1))
    rng=np.random.default_rng(38)
    for bits,shape,g in [(4,(2,5),3),(4,(3,7),8),(8,(2,5),3),(8,(3,7),8),(4,(1,1),1),(8,(1,1),1)]:
        w=rng.normal(size=shape).astype(np.float32);data,s,spec=quantize_weight(torch.tensor(w),bits,g)
        decoded=dequantize_weight(data,s,spec).numpy();expected=np.zeros_like(w);qmax=2**(bits-1)-1
        for row in range(shape[0]):
            for start in range(0,shape[1],g):
                segment=w[row,start:start+g];scale=float(np.max(np.abs(segment)))/qmax if np.any(segment) else 1.
                expected[row,start:start+g]=np.clip(np.rint(segment/scale),-qmax,qmax)*scale
        check(f'NumPy independent quant {bits}/{shape}/{g}',np.allclose(decoded,expected,atol=2e-7))
    for bits in [4,8]:
        w=torch.zeros(3,5);d,s,sp=quantize_weight(w,bits,3)
        check(f'zero matrix {bits}',torch.equal(dequantize_weight(d,s,sp),w) and torch.equal(s,torch.ones_like(s)))
    d,s,sp=quantize_weight(torch.tensor([[1.,2.,3.]]));bad=d.clone();bad[0]&=0xf0
    rejects('reserved code0',lambda:dequantize_weight(bad,s,sp))
    rejects('zero scale',lambda:dequantize_weight(d,torch.zeros_like(s),sp))
    rejects('unknown format',lambda:dequantize_weight(d,s,dict(sp,format='other')))
    rejects('wrong scale shape',lambda:dequantize_weight(d,s.flatten(),sp))
    rejects('invalid bits',lambda:quantize_weight(torch.ones(1,2),3))
    rejects('invalid group',lambda:quantize_weight(torch.ones(1,2),4,0))
    rejects('nonfinite weights',lambda:quantize_weight(torch.tensor([[float('nan')]])))
    rejects('empty weights',lambda:quantize_weight(torch.empty(0,2)))
    q,r,meta=affine(torch.tensor([-10.,0.,10.]),-1.,2.)
    check('affine clamp and exact zero',q.tolist()==[0,5,15] and r[1]==0)
    rejects('bad affine range',lambda:affine(torch.tensor([1.]),1.,2.))
    a,stats=fake_activation(torch.tensor([[-4.,0.,4.]]),'static8',torch.tensor(1/127))
    check('static saturation',a.tolist()==[[-1.,0.,1.]] and stats['clipped']==2)
    a,stats=fake_activation(torch.tensor([[-4.,0.,4.]]),'dynamic8')
    check('dynamic range',torch.allclose(a,torch.tensor([[-4.,0.,4.]])) and stats['clipped']==0)
    a,_=fake_activation(torch.zeros(2,3));check('activation zeros finite',torch.equal(a,torch.zeros_like(a)))
    rejects('zero activation scale',lambda:fake_activation(torch.ones(1,3),'static8',torch.tensor(0.)))
    # A single global scale loses the small row; per-row scales protect it in this example.
    w=torch.tensor([[.1,.2,.3],[10.,20.,30.]])
    global_scale=w.abs().max()/7;global_rec=torch.round(w/global_scale).clamp(-7,7)*global_scale
    d,s,sp=quantize_weight(w);row_rec=dequantize_weight(d,s,sp)
    check('outlier row granularity',((row_rec[0]-w[0])**2).mean()<((global_rec[0]-w[0])**2).mean())
    layer=nn.Linear(7,3);q=PackedLinear(layer,4,3);x=torch.tensor(rng.normal(size=(4,7)),dtype=torch.float32)
    expected=x.numpy()@dequantize_weight(q.data,q.scales,q.spec).numpy().T+q.bias.detach().numpy()
    check('packed linear NumPy matmul',np.allclose(q(x).detach().numpy(),expected,atol=2e-7))
    check('no hidden FP32 weight parameter',not list(q.parameters()) and set(q.state_dict())=={'data','scales','bias'})
    base=MLP();full=tensor_bytes(base);q=QuantMLP(base,4,8)
    check('MLP footprint',full==3532 and tensor_bytes(q)==1084)
    check('group padding cost',tensor_bytes(QuantMLP(base,4,32))==tensor_bytes(QuantMLP(base,8))==1288)
    check('fake activation no persistent saving',tensor_bytes(QuantMLP(base,4,8,'static8'))==1096)
    d=dataset();cal,ix=calibration_rows(d);groups=[set(v['groups']) for v in d.values()]
    check('group disjoint',all(not a&b for i,a in enumerate(groups) for b in groups[:i]))
    check('calibration train rows only',len(ix)==128 and len(set(ix))==128 and max(ix)<504 and torch.equal(cal,d['train']['x'][ix]))
    sc=calibrate(base,cal);check('calibration positive finite',sc.shape==(3,) and torch.isfinite(sc).all() and (sc>0).all())
    p=plan();check('memory ledger',p['mixed_weight_bytes']==3_924_375_000 and p['effective_bits']==4.485)
    check('KV remains after W4',p['kv_per_request_bytes']==512*1024**2 and p['payload_ceiling_fp16']==1 and p['payload_ceiling_mixed4']==20)
    return dict(passed=len(names),checks=len(names),names=names)
