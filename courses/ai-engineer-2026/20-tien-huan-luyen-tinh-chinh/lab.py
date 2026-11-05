"""Tiny pretraining -> SFT, scratch and frozen-backbone controls. CPU; no downloads."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
from core import ByteBPE, LanguageModel, Config, PAD, BOS, EOS, IGNORE

EVENTS={'timeout':'NET','connection-reset':'NET','denied':'AUTH','expired-token':'AUTH','deadlock':'DATA','duplicate-key':'DATA'}
LABELS=['AUTH','DATA','NET']

def make_data():
    groups=list(itertools.product(['api','worker','gateway','billing','search','cache','auth','queue'],
                                 ['read','write','list','update','delete','check']))
    random.Random(20).shuffle(groups)
    records={}
    for split,selected in [('train',groups[:32]),('validation',groups[32:40]),('test',groups[40:])]:
        records[split]=[]
        for service,op in selected:
            for event,label in EVENTS.items():
                text=f'service={service} op={op} event={event}.\n'
                records[split].append(dict(group=[service,op],text=text,prompt='Route: '+text+'Team: ',answer=label))
    return records


def batch(records,tok,sft=True):
    examples=[]
    for item in records:
        if sft:
            prefix=tok.encode(item['prompt']); answer=tok.encode(item['answer'])
            # Explicit boundary contract: whitespace/nonwhitespace pre-tokenizer prevents crossing here.
            assert prefix+answer==tok.encode(item['prompt']+item['answer'])
            seq=[BOS]+prefix+answer+[EOS]
        else:
            prefix=[];seq=[BOS]+tok.encode(item['text'])+[EOS]
        x,y=torch.tensor(seq[:-1]),torch.tensor(seq[1:])
        if sft:y[:len(prefix)]=IGNORE
        examples.append((x,y))
    length=max(len(x) for x,y in examples)
    xs=torch.full((len(examples),length),PAD,dtype=torch.long);ys=torch.full_like(xs,IGNORE)
    for i,(x,y) in enumerate(examples):xs[i,:len(x)]=x;ys[i,:len(y)]=y
    return xs,ys

@torch.no_grad()
def evaluate(model,records,tok,sft=True):
    model.eval();total=correct=count=0
    for start in range(0,len(records),16):
        x,y=batch(records[start:start+16],tok,sft);z=model(x);valid=y.ne(IGNORE)
        total+=F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=IGNORE,reduction='sum').item()
        count+=int(valid.sum());correct+=int((z.argmax(-1)[valid]==y[valid]).sum())
    return dict(ce=total/count,ppl=math.exp(total/count),token_accuracy=correct/count,tokens=count)

@torch.no_grad()
def generate(model,tok,prompt,budget=10):
    ids=[BOS]+tok.encode(prompt)
    if budget<1 or len(ids)+budget-1>model.cfg.max_len:raise ValueError('Context budget exceeded')
    output=[];ended=False;model.eval()
    for _ in range(budget):
        logits=model(torch.tensor([ids]))[0,-1].clone();logits[PAD]=logits[BOS]=-float('inf')
        token=int(logits.argmax());output.append(token)
        if token==EOS:ended=True;break
        ids.append(token)
    raw=tok.raw(output)
    try:text=raw.decode('utf-8');valid=True
    except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
    return dict(text=text,ids=output,eos=ended,valid_utf8=valid)


def task_metrics(model,records,tok):
    correct=valid=0;confusion=[[0]*4 for _ in LABELS];samples=[]
    for item in records:
        prediction=generate(model,tok,item['prompt'])
        # Exact includes EOS and canonical IDs; no stripping or repairing prediction.
        hit=prediction['ids']==tok.encode(item['answer'])+[EOS]
        correct+=hit
        label=prediction['text'] if prediction['eos'] and prediction['valid_utf8'] else None
        j=LABELS.index(label) if label in LABELS else 3
        valid+=j<3;confusion[LABELS.index(item['answer'])][j]+=1
        if len(samples)<6:samples.append(dict(prompt=item['prompt'],expected=item['answer'],prediction=prediction))
    return dict(exact=correct/len(records),correct=correct,samples_count=len(records),valid_format=valid/len(records),
                confusion=confusion,columns=LABELS+['INVALID'],samples=samples)


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path,model,tok,stage,epoch,parent_hash):
    torch.save(dict(model=model.state_dict(),config=asdict(model.cfg),tokenizer_hash=tok.fingerprint(),
                    stage=stage,epoch=epoch,parent_hash=parent_hash),path)


def load_checkpoint(path,tok):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp['tokenizer_hash']!=tok.fingerprint():raise ValueError('Tokenizer mismatch')
    model=LanguageModel(tok.size,Config(**cp['config']));model.load_state_dict(cp['model'])
    return model.eval(),cp


def fit_stage(model,records,tok,out,stage,epochs,lr,sft,seed=2026,head_only=False):
    parent_hash=model_hash(model)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    for name,p in model.named_parameters():p.requires_grad_(not head_only or name.startswith('head.'))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    # New optimizer for each new objective/branch: this is adaptation, not exact resume.
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=lr)
    rng=random.Random(seed);rows=[];best=float('inf')
    for epoch in range(1,epochs+1):
        order=list(records['train']);rng.shuffle(order);model.train();loss_sum=count=0
        for start in range(0,len(order),16):
            x,y=batch(order[start:start+16],tok,sft);opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=IGNORE)
            loss.backward();nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.);opt.step()
            n=int(y.ne(IGNORE).sum());loss_sum+=loss.item()*n;count+=n
        val=evaluate(model,records['validation'],tok,sft)
        rows.append(dict(epoch=epoch,steps=epoch*math.ceil(len(order)/16),train_ce=loss_sum/count,val_ce=val['ce']))
        if val['ce']<best:
            best=val['ce'];save_checkpoint(out/f'{stage}.pt',model,tok,stage,epoch,parent_hash)
    selected,cp=load_checkpoint(out/f'{stage}.pt',tok)
    changed=[name for name,p in selected.named_parameters() if not torch.equal(p,before[name])]
    frozen_unchanged=all(torch.equal(p,before[name]) for name,p in selected.named_parameters() if head_only and not name.startswith('head.'))
    if head_only:assert frozen_unchanged and all(name.startswith('head.') for name in changed)
    result=dict(epoch=cp['epoch'],trainable=trainable,updates=rows[-1]['steps'],parent_hash=parent_hash,
                hyperparameters=dict(epochs=epochs,lr=lr,batch_size=16,sft=sft,head_only=head_only,seed=seed),
                checkpoint_hash=model_hash(selected),changed=changed,frozen_unchanged=frozen_unchanged,
                validation=evaluate(selected,records['validation'],tok,sft))
    (out/f'{stage}-metrics.json').write_text(json.dumps(rows,indent=2))
    return selected,result


def run(out,pre_epochs=25,sft_epochs=25):
    if pre_epochs<1 or sft_epochs<1:raise ValueError('epochs must be positive')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(2026)
    records=make_data();tok=ByteBPE.fit([r['text'] for r in records['train']],steps=64)
    tok.save(out/'tokenizer.json');(out/'data.json').write_text(json.dumps(records,indent=2))
    cfg=Config(epochs=pre_epochs)
    initial=LanguageModel(tok.size,cfg);initial_hash=model_hash(initial)
    base,pre=fit_stage(deepcopy(initial),records,tok,out,'base',pre_epochs,.003,False)
    # Every adaptation branch gets a clone; do not let one experiment mutate another's starting point.
    full,sft=fit_stage(deepcopy(base),records,tok,out,'full-sft',sft_epochs,.001,True)
    scratch,sc=fit_stage(deepcopy(initial),records,tok,out,'scratch-sft',sft_epochs,.001,True)
    head,ho=fit_stage(deepcopy(base),records,tok,out,'head-sft',sft_epochs,.001,True,head_only=True)
    stages={'base':(base,pre),'full-sft':(full,sft),'scratch-sft':(scratch,sc),'head-sft':(head,ho)}
    # All checkpoint selection is finished before the test suite is opened.
    for name,(model,result) in stages.items():
        result['task_test']=task_metrics(model,records['test'],tok)
        result['task_ce_test']=evaluate(model,records['test'],tok,True)
        result['lm_test']=evaluate(model,records['test'],tok,False)
    assert model_hash(base)==sft['parent_hash']==ho['parent_hash']
    assert initial_hash==sc['parent_hash']
    assert all(EVENTS[r['text'].split('event=')[1].split('.')[0]]==r['answer'] for r in records['test'])
    report=dict(vocab=tok.size,parameters=sum(p.numel() for p in base.parameters()),initial_hash=initial_hash,
                config=asdict(cfg),pre_epochs=pre_epochs,sft_epochs=sft_epochs,stages={k:v[1] for k,v in stages.items()},
                baselines=dict(rule_exact=1.,majority_exact=1/3),
                metadata=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=1))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report


def mask_demo():
    tok=ByteBPE();item=dict(prompt='ab ',answer='X',text='ab X')
    x,y=batch([item],tok,True)
    assert x.tolist()==[[BOS,100,101,35,91]]
    assert y.tolist()==[[IGNORE,IGNORE,IGNORE,91,EOS]]
    print('x:',x.tolist(),'y:',y.tolist())
    logits=torch.zeros(1,5,tok.size,dtype=torch.float64,requires_grad=True)
    loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=IGNORE)
    loss.backward();assert logits.grad[0,:3].eq(0).all()
    print('Uniform answer-only loss:',loss.item(),'expected:',math.log(tok.size))


def hand_demo():
    logits=torch.tensor([[math.log(2),0.]],dtype=torch.float64,requires_grad=True)
    loss=F.cross_entropy(logits,torch.tensor([1]));loss.backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[2/3,-2/3]],dtype=torch.float64))
    updated=logits.detach()-.1*logits.grad
    print(dict(loss=loss.item(),gradient=logits.grad.tolist(),updated=updated.tolist(),new_p=updated.softmax(-1).tolist()))


def lora_demo():
    w=torch.tensor([[1.,0.,-1.],[0.,2.,1.]],dtype=torch.float64)
    a=torch.tensor([[1.,0.,1.]],dtype=w.dtype,requires_grad=True)
    b=torch.tensor([[.2],[-.1]],dtype=w.dtype,requires_grad=True)
    x=torch.tensor([2.,1.,3.],dtype=w.dtype)
    y=w@x+b@(a@x)
    torch.testing.assert_close(y,torch.tensor([0.,4.5],dtype=w.dtype))
    # Usual zero update initialization: B=0, A nonzero; A gets zero first gradient, B need not.
    b0=torch.zeros_like(b,requires_grad=True);((w@x+b0@(a@x)).square().sum()).backward()
    assert a.grad.eq(0).all() and b0.grad.abs().sum()>0
    print('LoRA hand output:',y.tolist(),'base weight has no gradient:',not w.requires_grad)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['mask','hand','lora','train','all'],default='mask')
    p.add_argument('--out',type=Path);p.add_argument('--pre-epochs',type=int,default=25);p.add_argument('--sft-epochs',type=int,default=25);a=p.parse_args()
    if a.mode in ('mask','all'):mask_demo()
    if a.mode in ('hand','all'):hand_demo()
    if a.mode in ('lora','all'):lora_demo()
    if a.mode in ('train','all'):
        out=a.out or Path(tempfile.mkdtemp(prefix='lesson20-'))/'run';run(out,a.pre_epochs,a.sft_epochs);print('Run:',out)
if __name__=='__main__':main()
