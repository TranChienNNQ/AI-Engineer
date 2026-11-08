"""Synthetic metric vectors with a fixed linear teacher, grouped train/val/test."""
import hashlib
import json
import random
import torch
TEACHER=torch.tensor([[1.2,.8,.4,0,.4,.3,0,-.2],[0,.7,-.2,.1,0,.4,.6,1.2],[-.3,.2,.8,.7,1,0,.3,-.2]])
FEATURES=['latency_z','error_z','queue_z','cpu_z','io_z','retry_z','cache_miss_z','change_z']


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def dataset():
    gen=torch.Generator().manual_seed(38);groups=list(range(64));random.Random(38).shuffle(groups);raw={}
    for group in range(64):
        x=torch.randn(12,8,generator=gen)+.15*torch.randn(1,8,generator=gen)
        raw[group]=x
    result={}
    for name,ids in [('train',groups[:42]),('validation',groups[42:53]),('test',groups[53:])]:
        x=torch.cat([raw[i] for i in ids]);y=(x@TEACHER.T).argmax(-1)
        result[name]=dict(x=x,y=y,groups=ids)
    assert not(set(result['train']['groups'])&set(result['test']['groups']))
    return result


def calibration_rows(data,count=128):
    g=torch.Generator().manual_seed(380);ix=torch.randperm(len(data['train']['x']),generator=g)[:count]
    return data['train']['x'][ix],ix.tolist()
