"""Actual tiny CPU LM pretraining then full/LoRA/packed-NF4 adaptation; no downloads."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from data import records,Tokenizer,batch,BOS,USER,ASSISTANT,EOS,PAD,IGNORE,TEMPLATE,data_hash
from model import TinyLM,Config,adapt,state_hash,frozen_state,tensor_bytes,TARGETS,merged
from quantization import NF4


def loss_sum(logits,labels):
    count=int(labels.ne(IGNORE).sum())
    if count==0:raise ValueError('all-masked loss')
    return F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=IGNORE,reduction='sum'),count


@torch.no_grad()
def evaluate(model,rows,tok,sft=True):
    model.eval();total=0;tokens=0
    for start in range(0,len(rows),16):
        x,y=batch(rows[start:start+16],tok,sft);loss,count=loss_sum(model(x),y)
        total+=float(loss);tokens+=count
    return dict(ce=total/tokens,ppl=math.exp(total/tokens),tokens=tokens)


@torch.no_grad()
def generate(model,row,tok):
    model.eval();ids=[BOS,USER]+tok.encode(row['prompt'])+[ASSISTANT];output=[]
    for _ in range(3):
        logits=model(torch.tensor([ids]))[0,-1].clone()
        logits[[PAD,BOS,USER,ASSISTANT]]=-torch.inf
        token=int(logits.argmax());output.append(token)
        if token==EOS:break
        ids.append(token)
    return output


def task_metrics(model,rows,tok):
    outputs=[generate(model,row,tok) for row in rows]
    correct=sum(p==tok.encode(r['answer'])+[EOS] for p,r in zip(outputs,rows))
    return dict(n=len(rows),correct=correct,exact=correct/len(rows),
                samples=[dict(prompt=r['prompt'],expected=r['answer'],ids=p,
                              predicted=[tok.inverse.get(i,'EOS' if i==EOS else 'SPECIAL') for i in p]) for r,p in list(zip(rows,outputs))[:6]])


def fit(model,splits,tok,epochs=40,lr=.003,sft=True):
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.0);rng=random.Random(3501)
    best=float('inf');best_state=None;best_epoch=None;history=[]
    for epoch in range(1,epochs+1):
        rows=list(splits['train']);rng.shuffle(rows);model.train();total=0;tokens=0
        for start in range(0,len(rows),16):
            x,y=batch(rows[start:start+16],tok,sft);opt.zero_grad(set_to_none=True)
            raw,count=loss_sum(model(x),y);loss=raw/count;loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();total+=float(raw.detach());tokens+=count
        val=evaluate(model,splits['validation'],tok,sft)
        history.append(dict(epoch=epoch,train_ce=total/tokens,validation_ce=val['ce']))
        if val['ce']<best:
            best=val['ce'];best_state=deepcopy(model.state_dict());best_epoch=epoch
    model.load_state_dict(best_state);model.eval()
    return model,dict(epoch=best_epoch,updates=epochs*math.ceil(len(rows)/16),trainable=sum(p.numel() for p in params),
                      history=history,validation=evaluate(model,splits['validation'],tok,sft),
                      hyperparameters=dict(epochs=epochs,lr=lr,batch_size=16,optimizer='AdamW',weight_decay=0.0,seed=3501,sft=sft))


def adapter_metadata(base,tok,kind,split_hash,rank=4,alpha=8):
    return dict(format='lesson35-adapter-v1',base_hash=state_hash(base.state_dict()),tokenizer_hash=tok.fingerprint(),
                template=TEMPLATE,data_hash=split_hash,model_config=asdict(base.cfg),kind=kind,rank=rank,alpha=alpha,
                targets=TARGETS,codebook_hash=state_hash({'nf4':NF4}) if kind=='packed-nf4-lora' else None,
                quantization='nf4-codebook-block32-fp32-scale-no-DQ-v1' if kind=='packed-nf4-lora' else 'none')


def save_adapter(path,model,metadata):
    if metadata['kind'] not in {'lora','packed-nf4-lora'}:raise ValueError('adapter kind')
    tensors={k:v.detach().clone() for k,v in model.state_dict().items() if k.endswith(('.A','.B'))}
    torch.save(dict(metadata=metadata,frozen_hash=state_hash(frozen_state(model)),adapter=tensors),path)


def load_adapter(path,base,tok,kind,split_hash):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp['metadata']!=adapter_metadata(base,tok,kind,split_hash):raise ValueError('adapter identity mismatch')
    torch.manual_seed(3502);model=adapt(base,kind=='packed-nf4-lora')
    if state_hash(frozen_state(model))!=cp['frozen_hash']:raise ValueError('frozen base/quantizer mismatch')
    wanted={k:v for k,v in model.state_dict().items() if k.endswith(('.A','.B'))}
    if set(wanted)!=set(cp['adapter']):raise ValueError('adapter keys')
    for key,value in cp['adapter'].items():
        if value.shape!=wanted[key].shape or value.dtype!=wanted[key].dtype or not torch.isfinite(value).all():raise ValueError('adapter tensor')
    result=model.load_state_dict(cp['adapter'],strict=False)
    if result.unexpected_keys:raise ValueError('unexpected adapter')
    return model.eval()


def train(out,pre_epochs=40,sft_epochs=50):
    if not 1<=pre_epochs<=100 or not 1<=sft_epochs<=100:raise ValueError('epochs1..100')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(3500)
    splits=records();tok=Tokenizer.fit(splits['train']);split_hash=data_hash(splits)
    initial=TinyLM(tok.size);base,pre=fit(initial,splits,tok,pre_epochs,.003,False)
    torch.save(dict(model=base.state_dict(),config=asdict(base.cfg),vocab=tok.vocab,data_hash=split_hash,template=TEMPLATE),out/'base.pt')
    (out/'data.json').write_text(json.dumps(splits,indent=2));(out/'tokenizer.json').write_text(json.dumps(tok.vocab,sort_keys=True))
    results={};models={};parent=state_hash(base.state_dict())
    for kind in ['full-sft','lora','packed-nf4-lora']:
        torch.manual_seed(3502)
        model=deepcopy(base) if kind=='full-sft' else adapt(base,kind=='packed-nf4-lora')
        before=frozen_state(model);storage=tensor_bytes(model)
        quant_base=evaluate(model,splits['validation'],tok,True)
        model,result=fit(model,splits,tok,sft_epochs,.001 if kind=='full-sft' else .01,True)
        if kind!='full-sft':
            assert state_hash(before)==state_hash(frozen_state(model))
            save_adapter(out/(kind+'.pt'),model,adapter_metadata(base,tok,kind,split_hash))
            loaded=load_adapter(out/(kind+'.pt'),base,tok,kind,split_hash)
            x,_=batch(splits['validation'][:4],tok)
            with torch.no_grad():
                torch.testing.assert_close(model(x),loaded(x),rtol=0,atol=0)
                error=float((model(x)-merged(model)(x)).abs().max())
                torch.testing.assert_close(model(x),merged(model)(x),rtol=2e-5,atol=2e-5)
            result.update(frozen_unchanged=True,adapter_roundtrip=True,merge_max_logit_error=error)
        else:
            torch.save(dict(model=model.state_dict(),config=asdict(model.cfg),base_hash=parent),out/'full-sft.pt')
        result.update(parent_hash=parent,resident_state_bytes=storage,before_sft_validation=quant_base)
        results[kind]=result;models[kind]=model
    # All branches have selected checkpoints before opening test outcomes.
    results['base']=dict(epoch=pre['epoch'],trainable=pre['trainable'],updates=pre['updates'],history=pre['history'],
                         resident_state_bytes=tensor_bytes(base),validation=evaluate(base,splits['validation'],tok,True))
    models['base']=base
    for kind,model in models.items():
        results[kind].update(task_test=task_metrics(model,splits['test'],tok),sft_ce_test=evaluate(model,splits['test'],tok,True),
                             lm_test=evaluate(model,splits['test'],tok,False))
    assert state_hash(base.state_dict())==parent
    return dict(kind='tiny CPU experiment; packed NF4 teaching model is not bitsandbytes QLoRA',
                vocab=tok.size,base_parameters=sum(p.numel() for p in base.parameters()),base_hash=parent,data_hash=split_hash,
                splits={k:dict(rows=len(v),groups=len({tuple(r['group']) for r in v})) for k,v in splits.items()},
                config=asdict(base.cfg),pre_epochs=pre_epochs,sft_epochs=sft_epochs,targets=TARGETS,
                stages=results,baselines=dict(rule_exact=1.0,majority_exact=1/3),
                artifacts='Local inference checkpoints, no optimizer/RNG resume state; runtime files excluded from course ZIP')
