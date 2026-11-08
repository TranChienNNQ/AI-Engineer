import copy
import math
from pathlib import Path
import numpy as np
import torch
from data import dataset,train_events,negative_pools,seen_at,eligible,fingerprint,SPEC
from model import MF,bpr_loss
from metrics import ranking_metrics,order,reciprocal_fusion,diversify
from policies import Policies,evaluate,candidate_demo,NAMES
from training import save_bundle,load_bundle,state_hash
from arithmetic import hand


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(420)
    names=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):check(name,True)
        else:raise AssertionError(name+' accepted')
    d=dataset();train=train_events(d);pools=negative_pools(d)
    check('train time cutoff',len(train)==384 and all(e['time']<10 for e in train))
    check('all warm IDs observed',{e['item'] for e in train}==set(range(72)))
    check('request counts',len(d['validation'])==48 and len(d['test'])==54)
    check('test cohorts',[sum(r['cohort']==c for r in d['test']) for c in ['warm','new_item','new_user']]==[36,12,6])
    for req in d['validation']+d['test']:
        ids=eligible(d,req)
        if req['target'] not in ids:raise AssertionError('ineligible target')
    check('all targets eligible',True)
    check('history before event',all(r['target'] not in seen_at(d,r['user'],r['time']) for r in d['test']))
    check('validation available as test history',all(r['target'] in seen_at(d,r['user'],11) for r in d['validation']))
    check('new users empty history',all(not seen_at(d,r['user'],11) for r in d['test'] if r['cohort']=='new_user'))
    check('future catalogue filtered',all(78 not in eligible(d,r) for r in d['test']))
    check('retired catalogue filtered',all(71 not in eligible(d,r) for r in d['test']))
    check('new item not released in validation',all(not(set(range(72,78))&set(eligible(d,r))) for r in d['validation']))
    first=d['test'][0];blocked=eligible(d,first)[:4]
    check('explicit blocks filtered',not(set(blocked)&set(eligible(d,first,blocked))))
    check('no seen candidate',all(not(set(eligible(d,r))&seen_at(d,r['user'],r['time'])) for r in d['test']))
    check('negative pools train-only unseen',all(set(pools[u])==set(range(72))-{e['item'] for e in train if e['user']==u} for u in pools))
    check('future positives not oracle removed',any(r['target'] in pools[r['user']] for r in d['validation']))
    mutated=copy.deepcopy(d)
    for event in mutated['events']:
        if event['time']>=10:event['item']=0
    check('future mutation leaves training unchanged',train_events(d)==train_events(mutated))
    check('future mutation leaves negatives unchanged',all(np.array_equal(v,negative_pools(mutated)[u]) for u,v in pools.items()))
    p=Policies(d,MF());other=Policies(mutated,p.model)
    check('future mutation leaves train stats unchanged',np.array_equal(p.pop,other.pop) and np.array_equal(p.sim,other.sim))
    check('cosine diagonal zero',np.array_equal(p.sim.diagonal(),np.zeros(72)))
    check('cosine symmetric',np.allclose(p.sim,p.sim.T,rtol=1e-12,atol=1e-12))
    a=np.array([[1,0,1],[1,1,0],[0,1,1]],dtype=float);s=a.T@a/np.sqrt(a.sum(axis=0)[:,None]*a.sum(axis=0)[None,:]);np.fill_diagonal(s,0)
    check('cooccurrence hand',np.allclose(s,[[0,.5,.5],[.5,0,.5],[.5,.5,0]]))
    check('tie order stable',order([5,2,3],[1.,1.,0.])==[2,5,3])
    reject('NaN score rejected',lambda:order([1],[float('nan')]))
    reject('duplicate IDs rejected',lambda:order([1,1],[0,1]))
    h=ranking_metrics(['A','B','C','D','E'],['B','E','F'],5)
    check('precision hand',h['precision']==.4)
    check('recall hand',h['recall']==2/3)
    check('MRR hand',h['mrr']==.5)
    expected=(1/math.log2(3)+1/math.log2(6))/(1+1/math.log2(3)+1/math.log2(4))
    check('NDCG hand',abs(h['ndcg']-expected)<1e-12)
    check('empty ranking finite',ranking_metrics([],[1])['ndcg']==0)
    check('short precision fixed K',ranking_metrics([1],[1],5)['precision']==.2)
    reject('duplicate ranking rejected',lambda:ranking_metrics([1,1],[1]))
    reject('empty relevance rejected',lambda:ranking_metrics([1],[]))
    reject('bad K rejected',lambda:ranking_metrics([1],[1],0))
    fused,scores=reciprocal_fusion([[1,2],[2,3]],60)
    check('RRF overlap',fused==[2,1,3] and abs(scores[2]-(1/62+1/61))<1e-12)
    reject('RRF duplicate rejected',lambda:reciprocal_fusion([[1,1]]))
    cat=[dict(id=i,topic=i//3) for i in range(9)]
    check('diversity greedy cap',diversify(list(range(9)),cat,5,2)==[0,1,3,4,6])
    check('diversity can return short',diversify([0,1,2],cat,5,2)==[0,1])
    reject('diversity invalid cap',lambda:diversify([0],cat,5,0))
    with torch.no_grad():p.model.user.weight.zero_();p.model.item.weight.zero_();p.model.bias.weight.zero_()
    check('MF zero scores',np.array_equal(p.raw(first,[0,1],'mf_bpr'),[0.,0.]))
    reject('MF unknown item rejected',lambda:p.raw(first,[72],'mf_bpr'))
    newuser=next(r for r in d['test'] if r['cohort']=='new_user')
    check('new user popularity fallback',p.recommend(newuser,'mf_bpr',5)==order(range(72),p.pop)[:5])
    for kind in NAMES:
        ranks=[p.recommend(req,kind,5) for req in d['test']]
        check(kind+' valid serving lists',all(len(rank)==5 and len(set(rank))==5 and set(rank)<=set(eligible(d,req)) for rank,req in zip(ranks,d['test'])))
    c=candidate_demo(p,d['test']);check('candidate ceiling',c['final_recall']<=c['candidate_recall'])
    req=copy.deepcopy(first);ids=eligible(d,req);check('all blocked returns empty',p.recommend(req,'hybrid_rrf',5,ids)==[])
    model=MF(users=1,items=2,dim=2).double()
    with torch.no_grad():model.user.weight.copy_(torch.tensor([[1.,0.]]));model.item.weight.copy_(torch.tensor([[1.,0.],[0.,1.]]));model.bias.weight.zero_()
    u=torch.tensor([0]);i=torch.tensor([0]);j=torch.tensor([1]);loss,_=bpr_loss(model,u,i,j,0);loss.backward()
    check('BPR hand loss',abs(float(loss.detach())-math.log1p(math.exp(-1)))<1e-12)
    check('BPR hand user gradient',np.allclose(model.user.weight.grad.numpy()[0],hand()['bpr']['gradient_p'],rtol=1e-12,atol=1e-12))
    check('BPR hand item gradients',np.allclose(model.item.weight.grad.numpy(),[hand()['bpr']['gradient_i'],hand()['bpr']['gradient_j']],rtol=1e-12,atol=1e-12))
    base=model.user.weight.detach().clone();eps=1e-6
    for column in range(2):
        values=[]
        for sign in [1,-1]:
            with torch.no_grad():model.user.weight.copy_(base);model.user.weight[0,column]+=sign*eps
            values.append(float(bpr_loss(model,u,i,j,0)[0].detach()))
        check('BPR finite difference user '+str(column),abs((values[0]-values[1])/(2*eps)-model.user.weight.grad[0,column].item())<1e-8)
    with torch.no_grad():model.user.weight.copy_(base)
    torch.optim.SGD(model.parameters(),lr=.1).step()
    check('BPR hand simultaneous update',np.allclose(model.user.weight.detach().numpy()[0],hand()['bpr']['after_p'],rtol=1e-12,atol=1e-12))
    before=float(bpr_loss(model,u,i,j,0)[0].detach());check('BPR pair loss decreases',before<hand()['bpr']['loss'])
    model=MF();before=state_hash(model.state_dict());u=torch.tensor([0,1]);i=torch.tensor([0,12]);j=torch.tensor([24,36]);loss,_=bpr_loss(model,u,i,j);loss.backward()
    check('all MF gradients finite nonzero',all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in model.parameters()))
    torch.optim.SGD(model.parameters(),lr=.1).step();check('actual train state change',state_hash(model.state_dict())!=before)
    save_bundle(out/'test.pt',model,d);restored,b=load_bundle(out/'test.pt',fingerprint(d))
    check('bundle exact scores',torch.equal(model(u,i),restored(u,i)))
    reject('bundle dataset mismatch',lambda:load_bundle(out/'test.pt','wrong'))
    b=torch.load(out/'test.pt',weights_only=True);b['item_ids'][0]=999;torch.save(b,out/'bad.pt')
    reject('bundle ID mapping mismatch',lambda:load_bundle(out/'bad.pt'))
    b=torch.load(out/'test.pt',weights_only=True);b['state']['item.weight'][0,0]+=.1;torch.save(b,out/'bad.pt')
    reject('bundle payload mismatch',lambda:load_bundle(out/'bad.pt'))
    return dict(checks=len(names),passed=len(names),names=names)
