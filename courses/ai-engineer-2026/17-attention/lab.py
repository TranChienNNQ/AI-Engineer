"""Attention from explicit QK-softmax-V to mask/API/MHA checks and learned associative recall."""
import argparse
import copy
import csv
from dataclasses import dataclass,asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader


def setup(seed=2026):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def attention(q,k,v,allowed=None):
    """Explicit math, no projections/dropout. Bool True means ALLOWED; rejects fully masked rows."""
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2]:raise ValueError('Incompatible Q/K/V shape')
    scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all():raise FloatingPointError('Non-finite attention scores')
    if allowed is not None:
        if allowed.dtype!=torch.bool:raise ValueError('Lab mask must be bool with True=allowed')
        mask=torch.broadcast_to(allowed,scores.shape)
        if not mask.any(dim=-1).all():raise ValueError('Every query row needs an allowed key')
        scores=scores.masked_fill(~mask,float('-inf'))
    weights=scores.softmax(-1)
    return weights@v,weights


def numpy_attention(q,k,v,allowed=None):
    """Independent float64 reference for the teaching examples."""
    q,k,v=[np.asarray(x,dtype=np.float64) for x in (q,k,v)]
    s=q@np.swapaxes(k,-2,-1)/np.sqrt(q.shape[-1])
    if allowed is not None:s=np.where(allowed,s,-np.inf)
    ex=np.exp(s-np.max(s,axis=-1,keepdims=True))
    a=ex/ex.sum(axis=-1,keepdims=True)
    return a@v,a


def hand_tensors(requires_grad=False):
    q=torch.tensor([[math.log(2)]],dtype=torch.float64,requires_grad=requires_grad)
    k=torch.tensor([[0.],[1.],[2.]],dtype=torch.float64,requires_grad=requires_grad)
    v=torch.tensor([[1.,0.],[0.,2.],[3.,1.]],dtype=torch.float64,requires_grad=requires_grad)
    return q,k,v


def hand_demo():
    q,k,v=hand_tensors(True);z,a=attention(q,k,v)
    expected=torch.tensor([[1/7,2/7,4/7]],dtype=torch.float64)
    torch.testing.assert_close(a,expected,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(z,torch.tensor([[13/7,8/7]],dtype=torch.float64))
    z[0,0].backward()
    ds=torch.tensor([[-6/49,-26/49,32/49]],dtype=torch.float64)
    torch.testing.assert_close(q.grad,ds@k.detach(),atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(k.grad,ds.T*q.detach(),atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(v.grad,expected.T@torch.tensor([[1.,0.]],dtype=torch.float64))
    nz,na=numpy_attention(q.detach().numpy(),k.detach().numpy(),v.detach().numpy())
    np.testing.assert_allclose(z.detach().numpy(),nz,atol=1e-14,rtol=1e-13)
    masked,ma=attention(q,k,v,torch.tensor([[True,True,False]]))
    return {'scores':(q.detach()@k.detach().T).tolist(),'weights':a.detach().tolist(),'output':z.detach().tolist(),
            'd_scores':ds.tolist(),'dQ':q.grad.tolist(),'dK':k.grad.tolist(),'dV':v.grad.tolist(),
            'masked_weights':ma.detach().tolist(),'masked_output':masked.detach().tolist()}


def matrix_demo():
    q=torch.eye(2,dtype=torch.float64)
    k=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    v=hand_tensors()[2]
    z,a=attention(q,k,v)
    return {'Q':q.tolist(),'K':k.tolist(),'V':v.tolist(),'scores':(q@k.T/math.sqrt(2)).tolist(),
            'weights':a.tolist(),'output':z.tolist()}


def masks_demo():
    setup(7)
    x=torch.randn(1,3,2,dtype=torch.float64)
    causal=torch.ones(3,3,dtype=torch.bool).tril()
    z,a=attention(x,x,x,causal)
    changed=x.clone();changed[:,2]+=10
    cz,_=attention(changed,changed,changed,causal)
    torch.testing.assert_close(z[:,:2],cz[:,:2],atol=0,rtol=0)
    api=F.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],attn_mask=causal,dropout_p=0.)[:,0]
    torch.testing.assert_close(z,api,atol=1e-14,rtol=1e-13)
    is_causal=F.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],is_causal=True,dropout_p=0.)[:,0]
    torch.testing.assert_close(z,is_causal,atol=1e-14,rtol=1e-13)
    # MHA has opposite boolean polarity. Identity projections isolate the same math.
    mha=nn.MultiheadAttention(2,1,bias=False,batch_first=True,dropout=0).to(dtype=torch.float64)
    with torch.no_grad():mha.in_proj_weight.copy_(torch.cat([torch.eye(2,dtype=torch.float64)]*3));mha.out_proj.weight.copy_(torch.eye(2,dtype=torch.float64))
    allowed=torch.tensor([[[True,True,False]]]).expand(1,3,3)
    direct,da=attention(x,x,x,allowed)
    mz,mw=mha(x,x,x,key_padding_mask=torch.tensor([[False,False,True]]),need_weights=True)
    torch.testing.assert_close(direct,mz,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(da,mw,atol=1e-14,rtol=1e-13)
    corrupt=x.clone();corrupt[:,2]=1000
    pz,_=attention(x,corrupt,corrupt,allowed)
    torch.testing.assert_close(direct,pz,atol=0,rtol=0)  # Q is intentionally unchanged.
    try:attention(x,x,x,torch.zeros(3,3,dtype=torch.bool))
    except ValueError:pass
    else:raise AssertionError('All-masked query must fail')
    return {'causal_allowed':causal.tolist(),'causal_weights':a.tolist(),'future_invariance':True,
            'padding_invariance':True,'sdpa_matches':True,'mha_opposite_mask_matches':True,
            'all_masked_rejected':True,'padded_query_output_norm':direct[0,2].norm().item()}


def multihead_demo():
    setup(13);B,T,D,H=2,3,4,2
    mha=nn.MultiheadAttention(D,H,batch_first=True,dropout=0).to(dtype=torch.float64)
    x=torch.randn(B,T,D,dtype=torch.float64)
    api,api_a=mha(x,x,x,need_weights=True,average_attn_weights=False)
    q,k,v=F.linear(x,mha.in_proj_weight,mha.in_proj_bias).chunk(3,dim=-1)
    split=lambda y:y.reshape(B,T,H,D//H).transpose(1,2)
    heads,a=attention(split(q),split(k),split(v))
    joined=heads.transpose(1,2).contiguous().reshape(B,T,D)
    manual=F.linear(joined,mha.out_proj.weight,mha.out_proj.bias)
    torch.testing.assert_close(manual,api,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(a,api_a,atol=1e-14,rtol=1e-13)
    return {'input':[B,T,D],'split_q':list(split(q).shape),'weights':list(a.shape),
            'joined':list(joined.shape),'output':list(api.shape),'parameters':sum(p.numel() for p in mha.parameters()),
            'max_output_error':(manual-api).abs().max().item()}


def invariance_demo():
    setup(9);x=torch.randn(4,3,dtype=torch.float64);perm=torch.tensor([2,0,3,1])
    y,a=attention(x,x,x)
    sy,sa=attention(x[perm],x[perm],x[perm])
    torch.testing.assert_close(sy,y[perm],atol=1e-14,rtol=1e-13)
    q=torch.randn(1,3,dtype=torch.float64)
    z,_=attention(q,x,x);reordered,_=attention(q,x[perm],x[perm])
    torch.testing.assert_close(z,reordered,atol=1e-14,rtol=1e-13)
    constant=torch.tensor([[1.,2.]]*3,dtype=torch.float64)
    a1=torch.tensor([[1/3]*3],dtype=torch.float64);a2=torch.tensor([[.99,.005,.005]],dtype=torch.float64)
    torch.testing.assert_close(a1@constant,a2@constant,atol=1e-14,rtol=1e-13)
    return {'self_permutation_equivariance':True,'cross_memory_permutation_invariance':True,
            'different_attention_same_output':True}


@dataclass(frozen=True)
class Config:
    dim:int=8
    epochs:int=40
    lr:float=.01
    batch_size:int=32
    seed:int=2026
    data_seed:int=17


def make_records(seed=17):
    rng=np.random.default_rng(seed);memories=[];seen=set()
    while len(memories)<192:
        count=int(rng.integers(2,6));keys=rng.choice(np.arange(1,7),size=count,replace=False)
        values=rng.integers(0,3,size=count)
        canonical=tuple(sorted((int(k),int(v)) for k,v in zip(keys,values)))
        if canonical not in seen:seen.add(canonical);memories.append(canonical)
    order=rng.permutation(192)
    splits={int(g):name for name,groups in [('train',order[:128]),('validation',order[128:160]),('test',order[160:])] for g in groups}
    records=[]
    for group,memory in enumerate(memories):
        queries=rng.choice([k for k,v in memory],size=2,replace=False)
        for view,query in enumerate(queries):
            shuffled=[memory[i] for i in rng.permutation(len(memory))]
            keys=[k for k,v in shuffled];vals=[v for k,v in shuffled]
            target=dict(memory)[int(query)]
            records.append({'id':f'g{group:03d}-q{view}','group':group,'split':splits[group],'query':int(query),
                            'keys':keys,'values':vals,'label':target})
    return records


class MemoryDataset(Dataset):
    def __init__(self,records,split):self.rows=[r for r in records if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def collate_memory(rows):
    n=len(rows);s=max(len(r['keys']) for r in rows)
    keys=torch.zeros(n,s,dtype=torch.long);values=torch.zeros(n,s,dtype=torch.long)
    for i,r in enumerate(rows):
        keys[i,:len(r['keys'])]=torch.tensor(r['keys']);values[i,:len(r['values'])]=torch.tensor(r['values'])
    return torch.tensor([r['query'] for r in rows]),keys,values,torch.tensor([r['label'] for r in rows])


def loader(dataset,batch_size=32,training=False,seed=2026):
    return DataLoader(dataset,batch_size=batch_size,shuffle=training,drop_last=False,num_workers=0,
                      collate_fn=collate_memory,generator=torch.Generator().manual_seed(seed))


class RecallAttention(nn.Module):
    def __init__(self,dim=8):
        super().__init__();self.ids=nn.Embedding(7,dim,padding_idx=0)
        self.q_proj=nn.Linear(dim,dim,bias=False);self.k_proj=nn.Linear(dim,dim,bias=False)
        self.values=nn.Embedding(3,dim);self.head=nn.Linear(dim,3)
    def forward(self,query,keys,value_ids):
        if query.ndim!=1 or keys.ndim!=2 or keys.shape!=value_ids.shape or len(query)!=len(keys):
            raise ValueError('Expected query B, keys/values B x S')
        if (query==0).any():raise ValueError('Query cannot be PAD')
        q=self.q_proj(self.ids(query))[:,None,:];k=self.k_proj(self.ids(keys));v=self.values(value_ids)
        z,a=attention(q,k,v,keys.ne(0)[:,None,:])
        return self.head(z[:,0]),a[:,0]


@torch.no_grad()
def evaluate(model,batches):
    model.eval();loss=0.;count=0;hits=0;confusion=torch.zeros(3,3,dtype=torch.long)
    for q,k,v,y in batches:
        logits,a=model(q,k,v);loss+=F.cross_entropy(logits,y,reduction='sum').item();count+=len(y)
        confusion+=torch.bincount(3*y+logits.argmax(-1),minlength=9).reshape(3,3)
        selected=k.gather(1,a.argmax(-1,keepdim=True)).squeeze(1);hits+=int((selected==q).sum())
    if not count or not math.isfinite(loss):raise ValueError('Empty evaluation or non-finite loss')
    return {'loss':loss/count,'accuracy':confusion.diag().sum().item()/count,'key_match_rate':hits/count,
            'n':count,'confusion':confusion.tolist()}


def load_best(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=RecallAttention(checkpoint['config']['dim']).to(device='cpu',dtype=torch.float32)
    model.load_state_dict(checkpoint['model']);model.eval();return model,checkpoint


def predict_memory(checkpoint_path,query,keys,values):
    if not keys or len(keys)!=len(values) or len(set(keys))!=len(keys):raise ValueError('Nonempty unique keys and matched values required')
    if query not in keys or any(k not in range(1,7) for k in keys) or any(v not in range(3) for v in values):
        raise ValueError('Query/key/value outside teaching contract')
    model,checkpoint=load_best(checkpoint_path)
    q,k,v,y=collate_memory([{'query':query,'keys':keys,'values':values,'label':0}])
    with torch.no_grad():logits,a=model(q,k,v);p=logits.softmax(-1)[0]
    return {'value_class':int(p.argmax()),'probabilities':p.tolist(),'attention':a[0].tolist()}


def train(out,config=Config()):
    if config.epochs<=0 or config.dim<=0 or config.lr<=0 or config.batch_size<=0:raise ValueError('Invalid config')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);setup(config.seed)
    records=make_records(config.data_seed);data={s:MemoryDataset(records,s) for s in ['train','validation','test']}
    model=RecallAttention(config.dim).to(device='cpu',dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=config.lr);best=math.inf;best_epoch=0;history=[];step=0
    data_hash=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    write_json(out/'data.json',records);write_json(out/'config.json',asdict(config))
    write_json(out/'metadata.json',{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
               'device':'cpu','dtype':'float32','threads':1,'mask_contract':'True=allowed; key0=PAD; values0..2 valid',
               'data_sha256':data_hash,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for epoch in range(1,config.epochs+1):
        model.train();total=0.;n=0
        for q,k,v,y in loader(data['train'],config.batch_size,True,config.seed+epoch):
            opt.zero_grad(set_to_none=True);logits,a=model(q,k,v);loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=loss.item()*len(y);n+=len(y);step+=1
        tr=evaluate(model,loader(data['train'],config.batch_size));val=evaluate(model,loader(data['validation'],config.batch_size))
        history.append({'epoch':epoch,'global_step':step,'online_loss':total/n,'train_loss':tr['loss'],
                        'train_accuracy':tr['accuracy'],'val_loss':val['loss'],'val_accuracy':val['accuracy'],
                        'val_key_match':val['key_match_rate']})
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch
            torch.save({'model':copy.deepcopy(model.state_dict()),'config':asdict(config),'epoch':epoch,'val_loss':best,
                        'data_sha256':data_hash,'key_ids':list(range(1,7)),'value_classes':[0,1,2]},out/'best.pt')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    selected,checkpoint=load_best(out/'best.pt');val=evaluate(selected,loader(data['validation'],config.batch_size))
    test=evaluate(selected,loader(data['test'],config.batch_size))
    counts=np.bincount([r['label'] for r in data['train'].rows],minlength=3);majority=int(counts.argmax())
    baseline=sum(r['label']==majority for r in data['validation'].rows)/len(data['validation'])
    probe=data['test'].rows[0];prediction=predict_memory(out/'best.pt',probe['query'],probe['keys'],probe['values'])
    with torch.no_grad():
        q,k,v,y=collate_memory([probe]);expected=selected(q,k,v)[0].softmax(-1)[0]
    torch.testing.assert_close(torch.tensor(prediction['probabilities']),expected,rtol=0,atol=0)
    report={'best_epoch':best_epoch,'best_val_loss':best,'global_step':step,
            'parameters':sum(p.numel() for p in selected.parameters()),'validation':val,'test':test,
            'baseline_val_accuracy':{'majority':baseline,'exact_dictionary_rule':1.},'prediction_roundtrip':True,
            'probe':{'query':probe['query'],'keys':probe['keys'],'values':probe['values'],'label':probe['label'],**prediction}}
    write_json(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','matrix','masks','multihead','invariance','train','all'],default='all')
    parser.add_argument('--out',type=Path);args=parser.parse_args();setup()
    for name,demo in [('hand',hand_demo),('matrix',matrix_demo),('masks',masks_demo),('multihead',multihead_demo),('invariance',invariance_demo)]:
        if args.mode in (name,'all'):print(name,json.dumps(demo(),indent=2))
    if args.mode in ('train','all'):
        out=args.out if args.out else Path(tempfile.mkdtemp(prefix='lesson17-'))/'run'
        print('train',json.dumps(train(out),indent=2));print('Artifacts:',out.resolve())

if __name__=='__main__':main()
