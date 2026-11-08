"""SFT initialization and actual tiny causal-LM DPO, CPU, no external downloads."""
import copy
import random
import torch
from torch.nn import functional as F
from data import BOS,EOS,PAD,IGNORE,Tokenizer,records,digest,batch,pair_batch,prefix,TEMPLATE
from model import TinyLM,state_hash
from objectives import sequence_logps,dpo_loss


def pair_scores(model,tok,rows):
    x,y=pair_batch(tok,rows);v=sequence_logps(model(x),y);return v.chunk(2)


@torch.no_grad()
def evaluate(model,reference,tok,rows,beta):
    model.eval();pc,pr=pair_scores(model,tok,rows);rc,rr=pair_scores(reference,tok,rows)
    loss,margin=dpo_loss(pc,pr,rc,rr,beta)
    return dict(loss=loss.item(),relative_margin=margin.mean().item(),
                relative_wins=int((margin>1e-6).sum()),relative_ties=int((margin.abs()<=1e-6).sum()),
                raw_wins=int((pc>pr).sum()),pairs=len(rows),chosen_logp=pc.mean().item(),rejected_logp=pr.mean().item())


@torch.no_grad()
def generation(model,tok,rows):
    model.eval();unique={r['prompt']:r for r in rows};outputs=[];route=quality=0
    for prompt,r in unique.items():
        ids=prefix(tok,prompt);answer=[]
        for _ in range(4):
            nxt=int(model(torch.tensor([ids]))[0,-1].argmax());ids.append(nxt);answer.append(nxt)
            if nxt==EOS:break
        desired=tok.encode(r['chosen'])+[EOS]
        route+=bool(answer and answer[0]==tok.index[r['label']]);quality+=answer==desired
        outputs.append(dict(prompt=prompt,tokens=[tok.words[t] for t in answer],expected=r['chosen']))
    return dict(prompts=len(unique),route_correct=route,exact_quality=quality,outputs=outputs)


def fit_sft(model,tok,rows,validation,epochs=35):
    # Dedup pair kinds: SFT imitates two styles with equal weight for each prompt.
    unique={r['prompt']:r for r in rows};items=[]
    for prompt,r in unique.items():items.extend([(prompt,r['chosen']),(prompt,f"{r['label']} GUESS")])
    valunique={r['prompt']:r for r in validation};valitems=[]
    for prompt,r in valunique.items():valitems.extend([(prompt,r['chosen']),(prompt,f"{r['label']} GUESS")])
    xv,yv=batch(tok,valitems);optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=0)
    best=float('inf');saved=None;best_epoch=0
    for epoch in range(1,epochs+1):
        order=list(items);random.Random(3600+epoch).shuffle(order);model.train()
        for start in range(0,len(order),16):
            x,y=batch(tok,order[start:start+16]);logits=model(x)
            loss=F.cross_entropy(logits.reshape(-1,tok.size),y.reshape(-1),ignore_index=IGNORE)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad():value=F.cross_entropy(model(xv).reshape(-1,tok.size),yv.reshape(-1),ignore_index=IGNORE).item()
        if value<best:best=value;saved=copy.deepcopy(model.state_dict());best_epoch=epoch
    model.load_state_dict(saved)
    return dict(best_epoch=best_epoch,validation_token_ce=best,epochs=epochs,updates=epochs*((len(items)+15)//16))


def validate_bundle(bundle,expected):
    if bundle['metadata']!=expected:raise ValueError('metadata mismatch')
    if digest(bundle['words'])!=expected['tokenizer_hash']:raise ValueError('tokenizer mismatch')
    if state_hash(bundle['reference'])!=expected['reference_hash']:raise ValueError('reference mismatch')


def experiment(out):
    torch.set_num_threads(1);torch.manual_seed(36);torch.use_deterministic_algorithms(True)
    splits=records();tok=Tokenizer(splits['train']);policy=TinyLM(tok.size)
    sft=fit_sft(policy,tok,splits['train'],splits['validation'])
    reference=copy.deepcopy(policy).eval().requires_grad_(False);reference_hash=state_hash(reference.state_dict());beta=.5
    # Test data is untouched until all checkpoint selection is finished.
    baseline=copy.deepcopy(policy).eval();best=float('inf');saved=None;best_epoch=0;history=[]
    optimizer=torch.optim.AdamW(policy.parameters(),lr=.002,weight_decay=0)
    # Cache reference scores in fixed row order; cache identity is recorded below.
    with torch.no_grad():rc,rr=pair_scores(reference,tok,splits['train'])
    for epoch in range(1,21):
        order=list(range(len(splits['train'])));random.Random(3700+epoch).shuffle(order);policy.train();total=0
        for start in range(0,len(order),16):
            idx=order[start:start+16];rows=[splits['train'][i] for i in idx];pc,pr=pair_scores(policy,tok,rows)
            loss,_=dpo_loss(pc,pr,rc[idx],rr[idx],beta)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step();total+=loss.item()*len(idx)
        val=evaluate(policy,reference,tok,splits['validation'],beta)
        history.append(dict(epoch=epoch,train_loss=total/len(order),validation_loss=val['loss']))
        if val['loss']<best:best=val['loss'];saved=copy.deepcopy(policy.state_dict());best_epoch=epoch
    policy.load_state_dict(saved);policy.eval()
    metadata=dict(seed=36,split_seed=35,template=TEMPLATE,tokenizer_hash=tok.fingerprint,data_hash=digest(splits),
                  train_hash=digest(splits['train']),reference_hash=reference_hash,beta=beta,loss='sigmoid',
                  reduction='sum completion logps including EOS; mean pairs',reference_cache='train row order; no truncation',
                  selection='minimum validation DPO loss; test not used',parameters=sum(p.numel() for p in policy.parameters()))
    checkpoint=dict(metadata=metadata,words=tok.words,policy=policy.state_dict(),reference=reference.state_dict())
    torch.save(checkpoint,out/'checkpoint.pt')
    loaded=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
    validate_bundle(loaded,metadata)
    restored=TinyLM(tok.size).eval();restored.load_state_dict(loaded['policy'])
    if loaded['metadata']!=metadata or state_hash(restored.state_dict())!=state_hash(policy.state_dict()):raise AssertionError('roundtrip')
    metrics={name:dict(preference=evaluate(m,reference,tok,splits['test'],beta),generation=generation(m,tok,splits['test'])) for name,m in [('sft',baseline),('dpo',policy)]}
    assert state_hash(reference.state_dict())==reference_hash and all(p.grad is None for p in reference.parameters())
    return dict(metadata=metadata,split_pairs={k:len(v) for k,v in splits.items()},split_groups={k:len({r['group'] for r in v}) for k,v in splits.items()},
                sft=sft,dpo=dict(best_epoch=best_epoch,validation_loss=best,epochs=20,updates=120,history=history),test=metrics,
                reference_unchanged=True,reference_has_no_grad=True,checkpoint_roundtrip=True,
                scope='synthetic rule pairs; tiny LM trained from scratch; full parameters; no PPO, LoRA or TRL')
