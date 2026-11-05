"""Lookup/gradient arithmetic and task-trained embeddings for synthetic backend tickets; CPU only."""
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
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
PAD=0;UNK=1
CLASSES=['auth','billing','network']
TOPICS=[['login','password','token'],['invoice','payment','refund'],['dns','latency','packet']]
TEMPLATES=['{service} {keyword}','please check {keyword} for {service}',
           '{keyword} error on {service}','{service} reports {keyword} issue',
           'investigate {keyword} affecting {service}','{service} has repeated {keyword} trouble',
           'urgent {service} request about {keyword}','team sees {keyword} failure in {service}']


def setup(seed=2026):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def tokenize(text):
    if not isinstance(text,str):raise ValueError('Expected text')
    tokens=text.lower().split()
    if any(t in {'<pad>','<unk>'} for t in tokens):raise ValueError('Reserved literal token')
    return tokens


def build_vocab(records):
    tokens=sorted({t for r in records if r['split']=='train' for t in tokenize(r['text'])})
    return {'<pad>':PAD,'<unk>':UNK,**{token:i+2 for i,token in enumerate(tokens)}}


def encode(text,vocab):
    tokens=tokenize(text)
    if not tokens:raise ValueError('Empty text is not supported')
    return torch.tensor([vocab.get(t,UNK) for t in tokens],dtype=torch.long)


def masked_mean(vectors,ids):
    mask=ids.ne(PAD).unsqueeze(-1)
    count=mask.sum(1)
    if (count==0).any():raise ValueError('All-padding sample is not supported')
    return (vectors*mask).sum(1)/count.to(vectors.dtype)


def hand_demo():
    weights=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,2.],[1.,1.]],dtype=torch.float64)
    embedding=nn.Embedding(5,2,padding_idx=PAD).to(dtype=torch.float64)
    with torch.no_grad():embedding.weight.copy_(weights)
    ids=torch.tensor([[2,3,2,0]],dtype=torch.long)
    vectors=embedding(ids)
    onehot=F.one_hot(ids,num_classes=5).to(torch.float64)
    torch.testing.assert_close(vectors,onehot@embedding.weight,atol=0,rtol=0)
    pooled=masked_mean(vectors,ids)
    torch.testing.assert_close(pooled,torch.tensor([[2/3,2/3]],dtype=torch.float64))
    loss=.5*pooled.square().sum();loss.backward()
    expected=torch.zeros_like(weights);expected[2]=4/9;expected[3]=2/9
    torch.testing.assert_close(embedding.weight.grad,expected,atol=1e-14,rtol=1e-13)
    gradients=embedding.weight.grad.detach().clone()
    optimizer=torch.optim.SGD(embedding.parameters(),lr=.1);optimizer.step()
    with torch.no_grad():after=.5*masked_mean(embedding(ids),ids).square().sum().item()
    assert abs(after-(17/27)**2)<1e-14
    frozen=nn.Embedding.from_pretrained(weights.clone(),freeze=True,padding_idx=0)
    head=nn.Linear(2,1).to(dtype=torch.float64)
    head(frozen(torch.tensor([2]))).sum().backward()
    assert frozen.weight.grad is None and head.weight.grad is not None
    return {'lookup':vectors.detach()[0].tolist(),'onehot_equivalent':True,'masked_mean':pooled.detach()[0].tolist(),
            'naive_mean':vectors.detach().mean(1)[0].tolist(),'gradient':gradients.tolist(),
            'loss_before':loss.item(),'loss_after':after,'frozen_has_no_gradient':True}


def geometry_demo():
    q=torch.tensor([1.,0.],dtype=torch.float64)
    candidates=torch.tensor([[1.,0.],[10.,10.]],dtype=torch.float64)
    dot=candidates@q
    cosine=F.cosine_similarity(candidates,q[None],dim=1)
    distances=(candidates-q).norm(dim=1)
    assert dot.argmax().item()==1 and cosine.argmax().item()==0
    unit=F.normalize(candidates,dim=1);uq=F.normalize(q,dim=0)
    torch.testing.assert_close((unit-uq).square().sum(1),2-2*cosine,atol=1e-14,rtol=1e-13)
    # Orthogonal coordinate rotation leaves cosine and a suitably rotated head unchanged.
    Q=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    W=torch.tensor([[1.,2.],[-1.,1.]],dtype=torch.float64)
    torch.testing.assert_close(candidates@W.T,(candidates@Q)@(W@Q).T,atol=0,rtol=0)
    torch.testing.assert_close(cosine,F.cosine_similarity(candidates@Q,(q@Q)[None],dim=1))
    return {'dot':dot.tolist(),'cosine':cosine.tolist(),'l2_distance':distances.tolist(),
            'normalized_l2_squared':(unit-uq).square().sum(1).tolist(),'rotation_preserves_logits':True}


def vocab_demo():
    records=[{'split':'train','text':'login error'},{'split':'train','text':'invoice error'},
             {'split':'validation','text':'new login'}]
    vocab=build_vocab(records)
    assert 'new' not in vocab and encode('new login',vocab)[0].item()==UNK
    return {'vocab':vocab,'new_login_ids':encode('new login',vocab).tolist()}


def make_records():
    records=[]
    for tid,template in enumerate(TEMPLATES):
        split='train' if tid<4 else 'validation' if tid<6 else 'test'
        for label,keywords in enumerate(TOPICS):
            for keyword in keywords:
                for service in ['atlas','boreal','cygnus']:
                    records.append({'id':f't{tid}-{keyword}-{service}','group':tid,'split':split,'label':label,
                                    'text':template.format(service=service,keyword=keyword)})
    return records


class TicketDataset(Dataset):
    def __init__(self,records,split,vocab):
        self.rows=[r for r in records if r['split']==split];self.vocab=vocab
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];return encode(row['text'],self.vocab),torch.tensor(row['label'],dtype=torch.long)


def collate_tickets(batch):
    ids,y=zip(*batch)
    return pad_sequence(ids,batch_first=True,padding_value=PAD),torch.stack(y)


def loader(dataset,batch_size=18,training=False,seed=2026):
    return DataLoader(dataset,batch_size=batch_size,shuffle=training,num_workers=0,drop_last=False,
                      collate_fn=collate_tickets,generator=torch.Generator().manual_seed(seed))


class TicketClassifier(nn.Module):
    def __init__(self,vocab_size,dim=4):
        super().__init__();self.embedding=nn.Embedding(vocab_size,dim,padding_idx=PAD)
        self.head=nn.Linear(dim,3)
        # No UNK examples in this train corpus: choose a neutral zero initialization.
        with torch.no_grad():self.embedding.weight[UNK].zero_()
    def forward(self,ids):
        if ids.ndim!=2 or ids.dtype!=torch.long:raise ValueError('Expected B x T long IDs')
        vectors=self.embedding(ids)
        return self.head(masked_mean(vectors,ids))


def oov_rate(rows,vocab):
    tokens=[t for r in rows for t in tokenize(r['text'])]
    return sum(t not in vocab for t in tokens)/len(tokens) if tokens else 0.


def keyword_rule(text):
    tokens=set(tokenize(text))
    hits=[i for i,keywords in enumerate(TOPICS) if tokens.intersection(keywords)]
    if len(hits)!=1:raise ValueError('Rule expects exactly one topic')
    return hits[0]


@torch.no_grad()
def evaluate(model,batches):
    model.eval();total=0.;count=0;confusion=torch.zeros(3,3,dtype=torch.long)
    for ids,y in batches:
        logits=model(ids);total+=F.cross_entropy(logits,y,reduction='sum').item();count+=len(y)
        confusion+=torch.bincount(3*y+logits.argmax(1),minlength=9).reshape(3,3)
    if not count:raise ValueError('Empty evaluation set')
    if not math.isfinite(total):raise FloatingPointError('Non-finite loss')
    return {'loss':total/count,'accuracy':confusion.diag().sum().item()/count,'n':count,'confusion':confusion.tolist()}


@dataclass(frozen=True)
class Config:
    dim:int=4
    epochs:int=60
    lr:float=.03
    batch_size:int=18
    seed:int=2026


def load_best(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TicketClassifier(len(checkpoint['vocab']),checkpoint['config']['dim']).to(device='cpu',dtype=torch.float32)
    model.load_state_dict(checkpoint['model']);model.eval();return model,checkpoint


def predict_text(checkpoint_path,text):
    model,checkpoint=load_best(checkpoint_path)
    ids=encode(text,checkpoint['vocab']).unsqueeze(0)
    with torch.no_grad():prob=model(ids).softmax(1)[0]
    label=int(prob.argmax());return {'class_id':label,'class_name':checkpoint['classes'][label],
                                   'probabilities':prob.tolist(),'unknown_tokens':int((ids==UNK).sum())}


def nearest_tokens(model,vocab,query,k=3):
    if query not in vocab or vocab[query] in (PAD,UNK):raise ValueError('Known non-special query required')
    with torch.no_grad():
        weights=model.embedding.weight
        if weights[vocab[query]].norm()<=1e-8:raise ValueError('Zero query vector')
        scores=F.cosine_similarity(weights,weights[vocab[query]][None],dim=1)
        candidates=[(token,float(scores[idx])) for token,idx in vocab.items()
                    if idx not in (PAD,UNK,vocab[query]) and weights[idx].norm()>1e-8]
    return sorted(candidates,key=lambda x:(-x[1],x[0]))[:k]


def train(out,config=Config()):
    if config.dim<=0 or config.epochs<=0 or config.lr<=0 or config.batch_size<=0:raise ValueError('Invalid config')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);setup(config.seed)
    records=make_records();vocab=build_vocab(records)
    datasets={s:TicketDataset(records,s,vocab) for s in ['train','validation','test']}
    model=TicketClassifier(len(vocab),config.dim).to(device='cpu',dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=0)
    initial=model.embedding.weight.detach().clone()
    data_hash=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    write_json(out/'data.json',records);write_json(out/'vocab.json',vocab);write_json(out/'config.json',asdict(config))
    write_json(out/'metadata.json',{'python':platform.python_version(),'torch':str(torch.__version__),
               'device':'cpu','dtype':'float32','threads':1,'data_sha256':data_hash,
               'tokenizer':'lower_then_whitespace_v1','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    best=math.inf;best_epoch=0;history=[];step=0
    for epoch in range(1,config.epochs+1):
        model.train();total=0.;n=0
        for ids,y in loader(datasets['train'],config.batch_size,True,config.seed+epoch):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(ids),y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
            loss.backward();opt.step();step+=1;total+=loss.item()*len(y);n+=len(y)
        tr=evaluate(model,loader(datasets['train'],config.batch_size))
        val=evaluate(model,loader(datasets['validation'],config.batch_size))
        history.append({'epoch':epoch,'global_step':step,'online_loss':total/n,'train_loss':tr['loss'],
                        'train_accuracy':tr['accuracy'],'val_loss':val['loss'],'val_accuracy':val['accuracy']})
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch
            torch.save({'model':copy.deepcopy(model.state_dict()),'config':asdict(config),'vocab':vocab,
                        'classes':CLASSES,'tokenizer':'lower_then_whitespace_v1','epoch':epoch,
                        'val_loss':best,'data_sha256':data_hash},out/'best.pt')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    selected,checkpoint=load_best(out/'best.pt')
    val=evaluate(selected,loader(datasets['validation'],config.batch_size))
    test=evaluate(selected,loader(datasets['test'],config.batch_size))
    rows=datasets['validation'].rows
    rule=sum(keyword_rule(r['text'])==r['label'] for r in rows)/len(rows)
    counts=torch.bincount(torch.tensor([r['label'] for r in datasets['train'].rows]),minlength=3)
    majority=int(counts.argmax());majority_acc=sum(r['label']==majority for r in rows)/len(rows)
    probe=datasets['test'].rows[0]['text'];prediction=predict_text(out/'best.pt',probe)
    with torch.no_grad():expected=selected(encode(probe,vocab)[None]).softmax(1)[0]
    torch.testing.assert_close(torch.tensor(prediction['probabilities']),expected,atol=0,rtol=0)
    assert torch.equal(selected.embedding.weight[PAD],torch.zeros(config.dim))
    assert torch.equal(selected.embedding.weight[UNK],initial[UNK])
    movement=(selected.embedding.weight-initial).norm().item()
    with (out/'embeddings.tsv').open('w') as f:
        for token,idx in sorted(vocab.items(),key=lambda p:p[1]):
            f.write(token+'\t'+'\t'.join(str(v) for v in selected.embedding.weight[idx].detach().tolist())+'\n')
    report={'best_epoch':best_epoch,'best_val_loss':best,'global_step':step,'vocab_size':len(vocab),
            'parameters':sum(p.numel() for p in selected.parameters()),'validation':val,'test':test,
            'baseline_val_accuracy':{'majority':majority_acc,'keyword_rule':rule},
            'oov_rate':{s:oov_rate(datasets[s].rows,vocab) for s in datasets},
            'embedding_change_norm':movement,'login_neighbors':nearest_tokens(selected,vocab,'login'),
            'prediction_roundtrip':True}
    write_json(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','geometry','vocab','train','all'],default='all')
    parser.add_argument('--out',type=Path);args=parser.parse_args();setup()
    for name,demo in [('hand',hand_demo),('geometry',geometry_demo),('vocab',vocab_demo)]:
        if args.mode in (name,'all'):print(name,json.dumps(demo(),indent=2))
    if args.mode in ('train','all'):
        out=args.out if args.out else Path(tempfile.mkdtemp(prefix='lesson16-'))/'run'
        print('train',json.dumps(train(out),indent=2));print('Artifacts:',out.resolve())

if __name__=='__main__':main()
