import copy
import hashlib
import json
import random
import torch
from torch.nn import functional as F
from data import dataset,calibration_rows,digest,TEACHER,FEATURES
from model import MLP,QuantMLP,calibrate
from quantization import tensor_bytes,FORMAT


def state_hash(state):
    h=hashlib.sha256()
    for key,t in sorted(state.items()):
        h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_bundle(path,expected):
    b=torch.load(path,weights_only=True,map_location='cpu')
    if b['metadata']!=expected or b['metadata']['format']!=FORMAT:raise ValueError('metadata mismatch')
    if state_hash(b['state'])!=expected['payload_hash']:raise ValueError('payload mismatch')
    spec=expected['spec']
    if spec['architecture']!='8-32-16-3-ReLU-v1':raise ValueError('architecture mismatch')
    m=QuantMLP(MLP(),spec['bits'],spec['group_size'],spec['activation'])
    m.load_state_dict(b['state']);m.eval()
    return m


@torch.no_grad()
def metrics(model,x,y,baseline):
    if isinstance(model,QuantMLP):logits,stats=model(x,True)
    else:logits=model(x);stats=[]
    ref=baseline(x);probs=ref.softmax(-1);kl=(probs*(ref.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean()
    return dict(correct=int((logits.argmax(-1)==y).sum()),count=len(y),accuracy=float((logits.argmax(-1)==y).float().mean()),
                ce=float(F.cross_entropy(logits,y)),agreement=float((logits.argmax(-1)==ref.argmax(-1)).float().mean()),
                logit_rmse=float(((logits-ref)**2).mean().sqrt()),kl_ref_to_quant=float(kl),
                activation_clipped=sum(s['clipped'] for s in stats),activation_elements=sum(s['total'] for s in stats))


def experiment(out):
    torch.set_num_threads(1);torch.manual_seed(38);torch.use_deterministic_algorithms(True)
    d=dataset();x=d['train']['x'];base=MLP(x.mean(0),x.std(0,unbiased=False).clamp_min(1e-6))
    opt=torch.optim.AdamW(base.parameters(),lr=.01,weight_decay=.001);history=[];best=float('inf');saved=None;best_epoch=0
    for epoch in range(1,61):
        order=list(range(len(x)));random.Random(3800+epoch).shuffle(order);base.train()
        for start in range(0,len(order),64):
            ix=order[start:start+64];loss=F.cross_entropy(base(x[ix]),d['train']['y'][ix])
            opt.zero_grad();loss.backward();opt.step()
        base.eval()
        with torch.no_grad():v=float(F.cross_entropy(base(d['validation']['x']),d['validation']['y']))
        history.append(dict(epoch=epoch,validation_ce=v))
        if v<best:best=v;saved=copy.deepcopy(base.state_dict());best_epoch=epoch
    base.load_state_dict(saved);base.eval().requires_grad_(False)
    cal,indices=calibration_rows(d);scales=calibrate(base,cal)
    cal_hash=digest(dict(train_groups=d['train']['groups'],indices=indices,values_hash=state_hash({'raw_calibration':cal})))
    configs=[('FP32',None,None,'none'),('W8-row',8,None,'none'),('W4-row',4,None,'none'),('W4-g8',4,8,'none'),
             ('W4-g32',4,32,'none'),('W4-g8-A8static',4,8,'static8'),('W4-g8-A8dynamic',4,8,'dynamic8')]
    rows=[];test=d['test'];ood=test['x']*4;ood_y=(ood@TEACHER.T).argmax(-1)
    for name,bits,g,activation in configs:
        model=base if bits is None else QuantMLP(base,bits,g,activation,scales)
        rows.append(dict(name=name,persistent_tensor_bytes=tensor_bytes(model),
                         weight_buffer_bytes=0 if bits is None else sum(layer.data.numel()*layer.data.element_size()+layer.scales.numel()*4 for layer in model.layers),
                         validation=metrics(model,d['validation']['x'],d['validation']['y'],base),
                         test=metrics(model,test['x'],test['y'],base),stress4x=metrics(model,ood,ood_y,base)))
    chosen=QuantMLP(base,4,8,'static8').eval()
    metadata=dict(format=FORMAT,spec=chosen.spec,source_model_hash=state_hash(base.state_dict()),calibration_hash=cal_hash,
                  split_hash=digest({k:v['groups'] for k,v in d.items()}),data_hash=state_hash({f'{k}_{field}':v[field] for k,v in d.items() for field in ['x','y']}),
                  feature_names=FEATURES,payload_hash=state_hash(chosen.state_dict()),seed=38,
                  calibration_rows=len(indices),selection='FP32 min validation CE; seven quant configs fixed before test')
    path=out/'quantized.pt';torch.save(dict(metadata=metadata,state=chosen.state_dict()),path)
    restored=load_bundle(path,metadata)
    with torch.no_grad():roundtrip=torch.equal(chosen(test['x']),restored(test['x']))
    assert roundtrip
    torch.save(dict(metadata=dict(seed=38,kind='FP32 reference'),state=base.state_dict()),out/'fp32.pt')
    return dict(metadata=metadata,parameters=sum(p.numel() for p in base.parameters()),
                splits={k:dict(rows=len(v['x']),groups=len(v['groups'])) for k,v in d.items()},
                training=dict(epochs=60,updates=480,best_epoch=best_epoch,validation_ce=best,history=history),
                activation_scales=scales.tolist(),rows=rows,checkpoint_roundtrip=roundtrip,
                files=dict(fp32_bytes=(out/'fp32.pt').stat().st_size,quantized_bytes=path.stat().st_size),
                rule_accuracy=float(((test['x']@TEACHER.T).argmax(-1)==test['y']).float().mean()),scope='synthetic grouped triage classification; packed weights and fake-quant activations with FP32 compute; no fused low-bit kernel or speed claim')
