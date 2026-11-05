"""Sequence order, hand RNN/BPTT, padding and variable-length classification; CPU only."""
import argparse
import copy
import csv
from dataclasses import asdict,dataclass
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
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
from torch.utils.data import Dataset,DataLoader
CLASSES=['rising','falling']


def setup(seed=2026):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def scalar_reference(values=(1.,-1.,2.),wx=.5,wh=.25,bias=0.,target=.5):
    """Independent scalar Python forward and manual BPTT for last-state squared loss."""
    states=[0.]
    for x in values:states.append(math.tanh(wx*x+wh*states[-1]+bias))
    loss=.5*(states[-1]-target)**2
    upstream=states[-1]-target;dwx=dwh=db=0.;deltas=[]
    for t in range(len(values),0,-1):
        delta=upstream*(1-states[t]**2);deltas.append(delta)
        dwx+=delta*values[t-1];dwh+=delta*states[t-1];db+=delta
        upstream=delta*wh
    return {'values':list(values),'states':states[1:],'loss':loss,'dwx':dwx,'dwh':dwh,'db':db,
            'deltas':list(reversed(deltas))}


def hand_demo():
    answer=scalar_reference()
    rnn=nn.RNN(1,1,batch_first=True).to(dtype=torch.float64)
    with torch.no_grad():
        rnn.weight_ih_l0.fill_(.5);rnn.weight_hh_l0.fill_(.25)
        rnn.bias_ih_l0.zero_();rnn.bias_hh_l0.zero_()
    x=torch.tensor(answer['values'],dtype=torch.float64).reshape(1,-1,1)
    output,hidden=rnn(x);loss=.5*(hidden[0,0,0]-.5).square();loss.backward()
    np.testing.assert_allclose(output.detach().numpy().flatten(),answer['states'],atol=1e-14,rtol=1e-13)
    for p,key in [(rnn.weight_ih_l0,'dwx'),(rnn.weight_hh_l0,'dwh'),
                  (rnn.bias_ih_l0,'db'),(rnn.bias_hh_l0,'db')]:
        assert abs(p.grad.item()-answer[key])<1e-14
    answer['reversed_final']=scalar_reference(tuple(reversed(answer['values'])))['states'][-1]
    return answer


def padding_demo():
    seqs=[torch.tensor([[1.]],dtype=torch.float64),torch.tensor([[2.],[3.],[4.]],dtype=torch.float64)]
    padded=pad_sequence(seqs,batch_first=True);lengths=torch.tensor([1,3],dtype=torch.long)
    rnn=nn.RNN(1,1,batch_first=True,nonlinearity='relu').to(dtype=torch.float64)
    with torch.no_grad():
        rnn.weight_ih_l0.fill_(1);rnn.weight_hh_l0.fill_(1)
        rnn.bias_ih_l0.fill_(.5);rnn.bias_hh_l0.zero_()
    output,hn=rnn(padded)
    gather=output[torch.arange(2),lengths-1]
    _,packed_h=rnn(pack_padded_sequence(padded,lengths,batch_first=True,enforce_sorted=False))
    torch.testing.assert_close(gather,packed_h[0],atol=0,rtol=0)
    assert hn[0,0,0].item()==2.5 and packed_h[0,0,0].item()==1.5
    changed=padded.clone();changed[0,1:]=99
    _,changed_h=rnn(pack_padded_sequence(changed,lengths,batch_first=True,enforce_sorted=False))
    torch.testing.assert_close(changed_h,packed_h,rtol=0,atol=0)
    return {'padded':padded[:,:,0].tolist(),'lengths':lengths.tolist(),
            'naive_last':hn[0,:,0].tolist(),'valid_last':gather[:,0].tolist(),
            'packed_last':packed_h[0,:,0].tolist()}


def gradient_demo():
    result={}
    for a in (.5,1.,1.5):
        h0=torch.tensor(1.,dtype=torch.float64,requires_grad=True);h=h0
        for _ in range(20):h=a*h
        h.backward();expected=a**20
        assert abs(h0.grad.item()-expected)<1e-10
        result[str(a)]={'steps':20,'dh20_dh0':h0.grad.item()}
    return result


def windows_demo():
    series=[10.,12.,11.,13.,15.]
    return [{'input_indices':list(range(t-3,t)),'input':series[t-3:t],
             'target_index':t,'target':series[t]} for t in range(3,len(series))]


@dataclass(frozen=True)
class Config:
    hidden:int=8
    epochs:int=30
    lr:float=.01
    batch_size:int=16
    seed:int=2026
    data_seed:int=17


def make_records(seed=17):
    rng=np.random.default_rng(seed);group_ids=rng.permutation(96)
    split={int(g):name for name,ids in [('train',group_ids[:64]),('validation',group_ids[64:80]),
                                       ('test',group_ids[80:])] for g in ids}
    records=[]
    for group in range(96):
        length=int(rng.integers(4,10));center=float(rng.uniform(-1,1));amplitude=float(rng.uniform(.6,1.2))
        rising=center+amplitude*np.linspace(-1,1,length)+rng.normal(0,.06,length)
        for label,values in [(0,rising),(1,rising[::-1])]:
            records.append({'id':f'g{group:02d}-{label}','group':group,'split':split[group],
                            'label':label,'values':values.astype(np.float32).tolist()})
    return records


def fit_stats(records):
    train=np.concatenate([r['values'] for r in records if r['split']=='train']).astype(np.float64)
    std=float(train.std())
    if std<=0:raise ValueError('Positive train standard deviation required')
    return {'mean':float(train.mean()),'std':std,'feature_count':1,'padding_side':'right','padding_value':0.}


class SequenceDataset(Dataset):
    def __init__(self,records,split,stats):
        self.rows=[r for r in records if r['split']==split];self.stats=stats
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];x=torch.tensor(row['values'],dtype=torch.float32).reshape(-1,1)
        x=(x-self.stats['mean'])/self.stats['std']
        return x,torch.tensor(row['label'],dtype=torch.long)


def collate_sequences(batch):
    sequences,targets=zip(*batch)
    lengths=torch.tensor([len(x) for x in sequences],dtype=torch.long,device='cpu')
    if (lengths<=0).any():raise ValueError('Empty sequences are not supported')
    return pad_sequence(sequences,batch_first=True,padding_value=0.,padding_side='right'),lengths,torch.stack(targets)


def loader(dataset,batch_size=16,training=False,seed=2026):
    return DataLoader(dataset,batch_size=batch_size,shuffle=training,num_workers=0,drop_last=False,
                      collate_fn=collate_sequences,generator=torch.Generator().manual_seed(seed))


class SequenceClassifier(nn.Module):
    def __init__(self,hidden=8):
        super().__init__();self.rnn=nn.RNN(1,hidden,batch_first=True,nonlinearity='tanh')
        self.head=nn.Linear(hidden,2)
    def forward(self,x,lengths):
        if x.ndim!=3 or x.shape[-1]!=1:raise ValueError('Expected B x T x 1')
        if lengths.ndim!=1 or len(lengths)!=len(x) or lengths.dtype!=torch.long or lengths.device.type!='cpu':
            raise ValueError('Lengths must be a CPU long vector matching batch size')
        if (lengths<=0).any() or (lengths>x.shape[1]).any():raise ValueError('Invalid lengths')
        packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        _,hn=self.rnn(packed)  # No hx: independent sequences start with zero hidden state.
        return self.head(hn[-1])


@torch.no_grad()
def evaluate(model,batches):
    model.eval();total=0.;count=0;confusion=torch.zeros(2,2,dtype=torch.long)
    for x,lengths,y in batches:
        logits=model(x,lengths);total+=F.cross_entropy(logits,y,reduction='sum').item();count+=len(y)
        confusion+=torch.bincount(2*y+logits.argmax(1),minlength=4).reshape(2,2)
    if not count:raise ValueError('Empty evaluation set')
    if not math.isfinite(total):raise FloatingPointError('Non-finite loss')
    return {'loss':total/count,'accuracy':confusion.diag().sum().item()/count,
            'n':count,'confusion':confusion.tolist()}


def load_best(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=SequenceClassifier(checkpoint['config']['hidden']).to(device='cpu',dtype=torch.float32)
    model.load_state_dict(checkpoint['model']);model.eval();return model,checkpoint


def predict_sequence(checkpoint_path,values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():raise ValueError('Expected finite nonempty 1D values')
    model,checkpoint=load_best(checkpoint_path);stats=checkpoint['preprocess']
    x=(torch.tensor(values).reshape(1,-1,1)-stats['mean'])/stats['std']
    with torch.no_grad():prob=model(x,torch.tensor([len(values)],dtype=torch.long)).softmax(1)[0]
    label=int(prob.argmax());return {'class_id':label,'class_name':checkpoint['classes'][label],'probabilities':prob.tolist()}


def train(out,config=Config()):
    if config.epochs<=0 or config.hidden<=0 or config.lr<=0 or config.batch_size<=0:raise ValueError('Invalid config')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);setup(config.seed)
    records=make_records(config.data_seed);stats=fit_stats(records)
    datasets={s:SequenceDataset(records,s,stats) for s in ['train','validation','test']}
    model=SequenceClassifier(config.hidden).to(device='cpu',dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=config.lr)
    data_hash=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    write_json(out/'data.json',records);write_json(out/'config.json',asdict(config));write_json(out/'preprocess.json',stats)
    write_json(out/'metadata.json',{'python':platform.python_version(),'torch':str(torch.__version__),
               'numpy':np.__version__,'device':'cpu','dtype':'float32','threads':1,'data_sha256':data_hash,
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    best=math.inf;best_epoch=0;history=[];step=0
    for epoch in range(1,config.epochs+1):
        model.train();total=0.;n=0;max_grad=0.
        for x,lengths,y in loader(datasets['train'],config.batch_size,True,config.seed+epoch):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,lengths),y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            max_grad=max(max_grad,norm.item());opt.step();step+=1;total+=loss.item()*len(y);n+=len(y)
        tr=evaluate(model,loader(datasets['train'],config.batch_size))
        val=evaluate(model,loader(datasets['validation'],config.batch_size))
        history.append({'epoch':epoch,'global_step':step,'online_loss':total/n,'train_loss':tr['loss'],
                        'train_accuracy':tr['accuracy'],'val_loss':val['loss'],'val_accuracy':val['accuracy'],
                        'max_grad_norm_preclip':max_grad})
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch
            torch.save({'model':copy.deepcopy(model.state_dict()),'config':asdict(config),'preprocess':stats,
                        'classes':CLASSES,'epoch':epoch,'val_loss':best,'data_sha256':data_hash},out/'best.pt')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    selected,checkpoint=load_best(out/'best.pt')
    val=evaluate(selected,loader(datasets['validation'],config.batch_size))
    test=evaluate(selected,loader(datasets['test'],config.batch_size))
    val_rows=[r for r in records if r['split']=='validation']
    # Predeclared simple domain rule, not fitted or selected using test.
    endpoint=sum((0 if r['values'][-1]>r['values'][0] else 1)==r['label'] for r in val_rows)/len(val_rows)
    first=datasets['test'].rows[0];prediction=predict_sequence(out/'best.pt',first['values'])
    x,ln,y=collate_sequences([datasets['test'][0]])
    with torch.no_grad():expected=selected(x,ln).softmax(1)[0]
    torch.testing.assert_close(torch.tensor(prediction['probabilities']),expected,atol=0,rtol=0)
    report={'best_epoch':best_epoch,'best_val_loss':best,'global_step':step,
            'parameters':sum(p.numel() for p in model.parameters()),'validation':val,'test':test,
            'baseline_val_accuracy':{'majority':.5,'endpoint_rule':endpoint},
            'preprocess':stats,'prediction_roundtrip':True}
    write_json(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','padding','gradients','windows','train','all'],default='all')
    parser.add_argument('--out',type=Path);args=parser.parse_args();setup()
    for name,demo in [('hand',hand_demo),('padding',padding_demo),('gradients',gradient_demo),('windows',windows_demo)]:
        if args.mode in (name,'all'):print(name,json.dumps(demo(),indent=2))
    if args.mode in ('train','all'):
        out=args.out if args.out else Path(tempfile.mkdtemp(prefix='lesson15-'))/'run'
        print('train',json.dumps(train(out),indent=2));print('Artifacts:',out.resolve())

if __name__=='__main__':main()
