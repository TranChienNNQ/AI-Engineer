"""Optimizer arithmetic, regularization ablation, and controlled learning failures; CPU only."""
import argparse
import copy
import csv
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import torch
from torch import nn
DTYPE = torch.float64


def setup(seed=2026):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def optimizer_trace():
    """Prescribed gradients, not gradients re-evaluated from a shared objective."""
    result = {}
    for name in ('sgd','momentum','adam'):
        w = nn.Parameter(torch.tensor(1., dtype=DTYPE))
        if name == 'sgd': optimizer = torch.optim.SGD([w], lr=.1)
        elif name == 'momentum': optimizer = torch.optim.SGD([w], lr=.1, momentum=.9)
        else: optimizer = torch.optim.Adam([w], lr=.1, betas=(.9,.99), eps=1e-8, weight_decay=0)
        rows=[];manual_w=1.;moment=0.;variance=0.
        for t,g in enumerate([2.,1.],1):
            w.grad = torch.tensor(g,dtype=DTYPE)
            optimizer.step()
            if name == 'sgd': manual_w -= .1*g
            elif name == 'momentum':
                moment = g if t == 1 else .9*moment+g
                manual_w -= .1*moment
            else:
                moment=.9*moment+.1*g;variance=.99*variance+.01*g*g
                mhat=moment/(1-.9**t);vhat=variance/(1-.99**t)
                manual_w -= .1*mhat/(math.sqrt(vhat)+1e-8)
            assert abs(w.item()-manual_w)<1e-13
            rows.append({'step':t,'gradient':g,'weight':w.item()})
        result[name]=rows
    return result


def decay_demo():
    result={}
    for name in ('sgd_decay','adam_l2','adamw'):
        w=nn.Parameter(torch.tensor(2.,dtype=DTYPE))
        if name=='sgd_decay':opt=torch.optim.SGD([w],lr=.1,weight_decay=.1)
        elif name=='adam_l2':opt=torch.optim.Adam([w],lr=.1,weight_decay=.1,decoupled_weight_decay=False)
        else:opt=torch.optim.AdamW([w],lr=.1,weight_decay=.1)
        w.grad=torch.zeros_like(w)  # A real zero gradient, not grad=None.
        opt.step();result[name]=w.item()
    assert abs(result['sgd_decay']-1.98)<1e-14
    assert abs(result['adamw']-1.98)<1e-14
    assert abs(result['adam_l2']-1.900000005)<1e-13
    w=torch.tensor(2.,dtype=DTYPE,requires_grad=True)
    (.1/2*w.square()).backward()
    assert abs(w.grad.item()-.2)<1e-14
    result['explicit_l2_gradient']=w.grad.item()
    return result


def clip_demo():
    w=nn.Parameter(torch.zeros(2,dtype=DTYPE));w.grad=torch.tensor([3.,4.],dtype=DTYPE)
    pre=torch.nn.utils.clip_grad_norm_([w],1.,error_if_nonfinite=True)
    assert pre.item()==5
    torch.testing.assert_close(w.grad,torch.tensor([.6,.8],dtype=DTYPE),atol=2e-7,rtol=0)
    return {'before':pre.item(),'after':w.grad.norm().item(),'gradient':w.grad.tolist()}


def quadratic_trace(lr, steps=12):
    w=1.;rows=[{'step':0,'w':w,'loss':5*w*w}]
    for step in range(1,steps+1):
        w -= lr*10*w
        rows.append({'step':step,'w':w,'loss':5*w*w})
    return rows


def failure_traces():
    results={}
    for name in ('missing_step','zero_lr','dead_relu'):
        w=nn.Parameter(torch.tensor(-1. if name=='dead_relu' else 0.,dtype=DTYPE))
        opt=torch.optim.SGD([w],lr=0. if name=='zero_lr' else .1)
        rows=[]
        for step in range(4):
            opt.zero_grad(set_to_none=True)
            prediction=torch.relu(w) if name=='dead_relu' else w
            loss=(prediction-1).square();loss.backward()
            before=w.detach().clone()
            if name!='missing_step':opt.step()
            rows.append({'loss':loss.item(),'grad':w.grad.item(),'update':(w-before).item()})
        results[name]=rows
    return results


def tiny_overfit():
    setup()
    X=torch.tensor([[-1.],[-.3],[.3],[1.]],dtype=DTYPE);y=2*X+1
    model=nn.Linear(1,1).to(dtype=DTYPE)
    with torch.no_grad():model.weight.zero_();model.bias.zero_()
    opt=torch.optim.SGD(model.parameters(),lr=.1)
    for _ in range(200):
        opt.zero_grad(set_to_none=True);loss=(model(X)-y).square().mean();loss.backward();opt.step()
    with torch.no_grad():final=(model(X)-y).square().mean().item()
    assert final<1e-8
    return final


@dataclass(frozen=True)
class Config:
    optimizer: str='adam'
    lr: float=.01
    l2: float=0.
    weight_decay: float=0.
    dropout: float=0.
    epochs: int=400
    seed: int=2026
    data_seed: int=17
    clip_norm: float|None=None


def make_data(config=Config()):
    gen=torch.Generator().manual_seed(config.data_seed)
    X=2*torch.rand((120,1),generator=gen,dtype=DTYPE)-1
    y=torch.sin(3*X)+.3*X+.3*torch.randn((120,1),generator=gen,dtype=DTYPE)
    ids=torch.randperm(120,generator=gen)
    split={'train':ids[:24].tolist(),'validation':ids[24:72].tolist(),'test':ids[72:].tolist()}
    data={name:(X[index],y[index]) for name,index in split.items()}
    fingerprint=hashlib.sha256(X.numpy().tobytes()+y.numpy().tobytes()+json.dumps(split,sort_keys=True).encode()).hexdigest()
    return data,split,fingerprint


def make_model(config):
    return nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Dropout(config.dropout),
                         nn.Linear(32,32),nn.Tanh(),nn.Dropout(config.dropout),nn.Linear(32,1)).to(dtype=DTYPE,device='cpu')


def weight_penalty(model):
    # Only Linear weight matrices; biases are excluded by explicit policy.
    return sum(p.square().sum() for name,p in model.named_parameters() if name.endswith('weight'))


def make_optimizer(model,config):
    if config.optimizer=='adam':
        if config.weight_decay:raise ValueError('Use explicit L2 or AdamW in this study.')
        return torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=0)
    if config.optimizer=='adamw':
        if config.l2:raise ValueError('Avoid combining L2 and decay in this ablation.')
        weights=[p for name,p in model.named_parameters() if name.endswith('weight')]
        biases=[p for name,p in model.named_parameters() if name.endswith('bias')]
        return torch.optim.AdamW([{'params':weights,'weight_decay':config.weight_decay},
                                 {'params':biases,'weight_decay':0.}],lr=config.lr)
    raise ValueError('Unknown optimizer')


@torch.no_grad()
def mse(model,pair):
    model.eval();X,y=pair;pred=model(X)
    if pred.shape!=y.shape:raise ValueError('Prediction/target shape mismatch')
    value=(pred-y).square().mean().item()
    if not math.isfinite(value):raise FloatingPointError('Non-finite metric')
    return value


def parameter_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def fit(out,config=Config()):
    if config.epochs<=0 or config.lr<=0 or config.l2<0 or config.weight_decay<0:
        raise ValueError('Invalid config')
    if config.clip_norm is not None and config.clip_norm<=0:raise ValueError('clip_norm must be positive')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    setup(config.seed);data,split,fingerprint=make_data(config)
    model=make_model(config);opt=make_optimizer(model,config)
    write_json(out/'config.json',asdict(config))
    write_json(out/'metadata.json',{'python':platform.python_version(),'torch':str(torch.__version__),
               'device':'cpu','dtype':'float64','threads':1,'split':split,'data_sha256':fingerprint,
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    history=[];best=math.inf;best_epoch=0;best_state=None
    X,y=data['train']
    for epoch in range(1,config.epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        before=parameter_vector(model).clone()
        pred=model(X)
        if pred.shape!=y.shape:raise ValueError('Prediction/target shape mismatch')
        data_loss=(pred-y).square().mean()
        penalty=config.l2/2*weight_penalty(model)
        objective=data_loss+penalty
        if not torch.isfinite(objective):raise FloatingPointError('Non-finite objective')
        objective.backward()
        grads=torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        norm=grads.norm().item()
        if not math.isfinite(norm):raise FloatingPointError('Non-finite gradient')
        if config.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip_norm,error_if_nonfinite=True)
        opt.step()
        after=parameter_vector(model)
        ratio=((after-before).norm()/(before.norm()+1e-12)).item()
        train=mse(model,data['train']);val=mse(model,data['validation'])
        row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'data_loss_online':data_loss.item(),
             'l2_penalty':penalty.item(),'objective_online':objective.item(),'train_mse':train,'val_mse':val,
             'grad_norm_preclip':norm,'update_ratio':ratio,'weight_norm':math.sqrt(weight_penalty(model).item())}
        history.append(row)
        if val<best:
            best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    torch.save({'config':asdict(config),'model':best_state,'epoch':best_epoch,
                'val_mse':best,'data_sha256':fingerprint},out/'best.pt')
    result={'best_epoch':best_epoch,'best_val':best,'last_train':history[-1]['train_mse'],
            'last_val':history[-1]['val_mse'],'last_weight_norm':history[-1]['weight_norm']}
    write_json(out/'summary.json',result)
    return result


def load_best(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=make_model(Config(**checkpoint['config']));model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint


def study(out):
    out=Path(out);candidates=[]
    configs=[('plain',Config()),('l2',Config(l2=.001)),
             ('adamw',Config(optimizer='adamw',weight_decay=.03)),('dropout',Config(dropout=.2))]
    for name,config in configs:
        result=fit(out/name,config);candidates.append({'run':name,**result})
    winner=min(candidates,key=lambda r:r['best_val'])
    model,checkpoint=load_best(out/winner['run']/'best.pt')
    data,_,fingerprint=make_data(Config(**checkpoint['config']))
    assert fingerprint==checkpoint['data_sha256']
    # Baseline is fixed from train; compare baseline validation for context only.
    mean=data['train'][1].mean()
    baseline_val=(data['validation'][1]-mean).square().mean().item()
    test=mse(model,data['test'])  # One test evaluation, after all model selection.
    report={'candidates':candidates,'selected':winner['run'],'baseline_val':baseline_val,
            'test_mse':test,'test_n':48,'rule':'minimum validation MSE; ties choose first declared run'}
    write_json(out/'study.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['optimizers','decay','clip','diagnose','study','all'],default='all')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();setup()
    demos={'optimizers':optimizer_trace,'decay':decay_demo,'clip':clip_demo,
           'diagnose':lambda:{'failures':failure_traces(),'tiny_fit_mse':tiny_overfit(),
                               'quadratic':{str(lr):quadratic_trace(lr) for lr in [.01,.1,.2,.25]}}}
    for name,demo in demos.items():
        if args.mode in (name,'all'):print(name,json.dumps(demo(),indent=2))
    if args.mode in ('study','all'):
        out=args.out if args.out else Path(tempfile.mkdtemp(prefix='lesson13-'))/'study'
        if out.exists():raise FileExistsError('Choose a new output directory')
        print('study',json.dumps(study(out),indent=2));print('Artifacts:',out.resolve())

if __name__=='__main__':main()
