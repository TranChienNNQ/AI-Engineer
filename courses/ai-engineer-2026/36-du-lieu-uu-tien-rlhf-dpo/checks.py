"""Independent arithmetic/finite differences and boundary checks for the teaching lab."""
import copy
import math
import numpy as np
import torch
from data import records,validate,Tokenizer,example,batch,IGNORE,EOS,PAD
from model import TinyLM
from objectives import dpo_loss,sequence_logps,hand
from bandit import optimal


def run():
    torch.set_num_threads(1);checks=[]
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:check(name,True);return
        raise AssertionError(name+' accepted invalid input')
    h=hand();check('DPO hand margin',abs(h['margin']-math.log(1.75))<1e-12)
    check('DPO hand loss',abs(h['loss']-math.log1p(math.exp(-.5*math.log(1.75))))<1e-12)
    check('opposite sign gradients',h['d_pc']<0<h['d_pr'] and abs(h['d_pc']+h['d_pr'])<1e-12)
    # Independent scalar objective uses NumPy, never the torch DPO implementation.
    values=np.array([-.4,-1.2,-.6,-1.]);theta=torch.tensor(values[:2],dtype=torch.float64,requires_grad=True)
    refs=torch.tensor(values[2:],dtype=torch.float64)
    loss,_=dpo_loss(theta[:1],theta[1:],refs[:1],refs[1:],.7);loss.backward()
    def oracle(v):return np.logaddexp(0,-.7*((v[0]-values[2])-(v[1]-values[3])))
    for i in range(2):
        a=values[:2].copy();b=a.copy();a[i]+=1e-6;b[i]-=1e-6
        check(f'finite difference {i}',abs(theta.grad[i].item()-(oracle(a)-oracle(b))/2e-6)<1e-8)
    a=torch.tensor([-.2],requires_grad=True);b=torch.tensor([-1.])
    loss,m=dpo_loss(a,b,a.detach(),b.detach());check('initial margin zero and loss log2',m.item()==0 and abs(loss.item()-math.log(2))<1e-6)
    loss.backward();check('zero margin still has gradient',a.grad.item()<0)
    for beta in [0,-1,float('nan'),float('inf')]:rejects(f'beta guard {beta}',lambda beta=beta:dpo_loss(a,b,a.detach(),b.detach(),beta))
    rejects('reference gradient guard',lambda:dpo_loss(a,b,a,b))
    s=records();t=Tokenizer(s['train']);check('pair counts',[len(x) for x in s.values()]==[96,24,24])
    check('group counts',[len({r['group'] for r in x}) for x in s.values()]==[16,4,4])
    check('no heldout unknown tokens',all(5 not in t.encode(r[k]) for rows in s.values() for r in rows for k in ['prompt','chosen','rejected']))
    for name,mutate in [
        ('split group leakage',lambda d:d['test'][0].update(group=d['train'][0]['group'])),
        ('identical pair',lambda d:d['train'][0].update(rejected=d['train'][0]['chosen'])),
        ('duplicate pair',lambda d:d['train'].append(copy.deepcopy(d['train'][0]))),
        ('reverse conflict',lambda d:d['train'].append(dict(d['train'][0],chosen=d['train'][0]['rejected'],rejected=d['train'][0]['chosen']))),
        ('empty prompt',lambda d:d['train'][0].update(prompt=' '))]:
        d=copy.deepcopy(s);mutate(d);rejects(name,lambda:validate(d))
    x,y=example(t,'route timeout','NET CHECK');check('answer and EOS count',sum(v!=IGNORE for v in y)==3 and y[-1]==EOS)
    first=next(i for i,v in enumerate(y) if v!=IGNORE);check('shift predicts first answer from assistant marker',x[first]==4 and y[first]==t.index['NET'])
    rejects('empty completion',lambda:example(t,'route',''))
    rejects('overlong not truncated',lambda:example(t,'route '*40,'NET'))
    xx,yy=batch(t,[('route timeout','NET CHECK'),('route','AUTH')])
    logits=torch.arange(xx.numel()*t.size,dtype=torch.float64).reshape(*xx.shape,t.size)/79
    lp=sequence_logps(logits,yy);arr=logits.numpy();expected=[]
    for i in range(len(yy)):
        total=0.
        for j,label in enumerate(yy[i].tolist()):
            if label==IGNORE:continue
            row=arr[i,j];total+=row[label]-(np.max(row)+np.log(np.exp(row-np.max(row)).sum()))
        expected.append(total)
    check('independent per-token logsumexp sum',np.allclose(lp.numpy(),expected,atol=1e-12))
    changed=logits.clone();changed[yy==IGNORE]=torch.randn_like(changed[yy==IGNORE])*100
    check('prompt PAD ignored',torch.equal(lp,sequence_logps(changed,yy)))
    no_eos=yy.clone();no_eos[no_eos==EOS]=IGNORE
    check('EOS contributes',not torch.allclose(lp,sequence_logps(logits,no_eos)))
    check('sum is not length average',not torch.allclose(lp,lp/yy.ne(IGNORE).sum(1)))
    rejects('all masked guard',lambda:sequence_logps(logits,torch.full_like(yy,IGNORE)))
    torch.manual_seed(36);model=TinyLM(t.size).eval()
    with torch.no_grad():
        short_x,short_y=batch(t,[('route','AUTH')]);single=sequence_logps(model(short_x),short_y)[0]
        padded=sequence_logps(model(xx),yy)[1]
    check('right PAD invariant model score',torch.allclose(single,padded,atol=2e-6))
    ref=torch.tensor([.5,.5],dtype=torch.float64);reward=torch.tensor([math.log(3),0.],dtype=torch.float64)
    p=optimal(ref,reward,1.);check('closed form .75 .25',torch.allclose(p,torch.tensor([.75,.25],dtype=torch.float64)))
    check('reward offset invariant',torch.allclose(p,optimal(ref,reward+123,1.)))
    check('KL objective log2',abs(h['objective']-math.log(2))<1e-12)
    check('chosen can fall with positive relative margin',math.log(.2/.4)-math.log(.1/.3)>0 and .2<.4)
    return dict(passed=len(checks),checks=len(checks),names=checks)
