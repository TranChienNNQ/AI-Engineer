import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import dataset,train_events,negative_pools,fingerprint,SPEC
from model import MF,bpr_loss
from policies import Policies,evaluate,candidate_demo,NAMES


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_bundle(path,model,data):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(spec=dict(SPEC),state=state,state_hash=state_hash(state),data_hash=fingerprint(data),
                    user_ids=list(range(48)),item_ids=list(range(72))),path)


def load_bundle(path,expected_data_hash=None):
    b=torch.load(path,map_location='cpu',weights_only=True)
    if set(b)!=set(['spec','state','state_hash','data_hash','user_ids','item_ids']) or b['spec']!=SPEC:raise ValueError('bundle keys/spec')
    if b['user_ids']!=list(range(48)) or b['item_ids']!=list(range(72)):raise ValueError('ID mapping mismatch')
    if expected_data_hash is not None and b['data_hash']!=expected_data_hash:raise ValueError('data snapshot mismatch')
    if state_hash(b['state'])!=b['state_hash']:raise ValueError('state digest mismatch')
    model=MF();expected=model.state_dict()
    if set(expected)!=set(b['state']):raise ValueError('state keys')
    for key,v in b['state'].items():
        if v.shape!=expected[key].shape or v.dtype!=expected[key].dtype or not torch.isfinite(v).all():raise ValueError('state shape/dtype/finite')
    model.load_state_dict(b['state']);model.eval();return model,b


def train(out,epochs=160):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(42)
    data=dataset();model=MF();events=train_events(data);pools=negative_pools(data);rng=np.random.default_rng(42)
    u=torch.tensor([e['user'] for e in events]);i=torch.tensor([e['item'] for e in events]);optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    initial=evaluate(Policies(data,model),data['test'],'mf_bpr') # predeclared random baseline, never checkpoint selection
    best=-1.;best_state=None;best_epoch=None;history=[];sampled_future=0
    future={(e['user'],e['item']) for e in data['events'] if e['time']>=10}
    for epoch in range(1,epochs+1):
        negative=np.array([rng.choice(pools[int(user)]) for user in u]);j=torch.tensor(negative)
        sampled_future+=sum((int(user),int(item)) in future for user,item in zip(u,negative)) # audit only, not sampler input
        optimizer.zero_grad(set_to_none=True);loss,pair=bpr_loss(model,u,i,j,.001);loss.backward();optimizer.step()
        score=evaluate(Policies(data,model),data['validation'],'mf_bpr')['overall']['ndcg']
        history.append(dict(epoch=epoch,loss=float(loss.detach()),pair_loss=float(pair.detach()),validation_ndcg5=score))
        if score>best:best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);policies=Policies(data,model)
    results={name:evaluate(policies,data['test'],name) for name in NAMES};candidate=candidate_demo(policies,data['test'])
    save_bundle(out/'recommendation.pt',model,data);loaded,b=load_bundle(out/'recommendation.pt',fingerprint(data))
    again={name:evaluate(Policies(data,loaded),data['test'],name) for name in NAMES}
    if results!=again:raise AssertionError('bundle ranking mismatch')
    return dict(seed=42,epochs=epochs,updates=epochs,train_pairs=len(events),negatives_per_pair_per_epoch=1,negative_snapshot_time=9,
        learning_rate=.03,regularization=.001,parameter_count=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,best_validation_ndcg5=best,history=history,initial=initial,results=results,candidate_demo=candidate,
        future_positive_negative_draws=sampled_future,data_hash=fingerprint(data),state_hash=b['state_hash'],roundtrip_exact=True,
        bundle_bytes=(out/'recommendation.pt').stat().st_size,
        scope='synthetic clicks without impressions; ID-only MF BPR, exact eligible catalogue evaluation, frozen weights/statistics before test, validation clicks allowed as past request context; no ANN, trained neural ranker, A/B or production service')
