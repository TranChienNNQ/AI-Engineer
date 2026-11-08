import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import dataset,fingerprint,CAPTIONS,SPEC,rule_predict
from preprocess import fit,transform,bow
from model import Align,total_loss,MODALITIES
from metrics import top1,late_fuse


def tensor_features(records,prep):return {k:torch.tensor(v) for k,v in transform(records,prep).items()}


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prep_hash(prep):return hashlib.sha256(json.dumps(prep,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@torch.no_grad()
def evaluate(model,records,prep):
    model.eval();z=model(tensor_features(records,prep));labels=np.array([r['label'] for r in records])
    retrieval={a+'_to_'+b:top1((z[a]@z[b].T).numpy(),labels,labels) for a,b in itertools.permutations(MODALITIES,2)}
    text=torch.tensor(np.stack([bow(s,prep['vocab']) for s in CAPTIONS]))
    candidates=F.normalize(model.encoders['text'](text),dim=-1,eps=1e-8)
    scores={k:(z[k]@candidates.T).numpy() for k in ['image','audio']}
    classify=lambda s:top1(s,labels,np.arange(6))
    mismatched=np.roll(scores['audio'],4,axis=0)
    blank=[dict(r,image=np.zeros_like(r['image']),audio=np.zeros_like(r['audio'])) for r in records]
    blank_z=model(tensor_features(blank,prep))
    perturb=dict(image_only=classify(late_fuse(scores['image'],None)),audio_only=classify(late_fuse(None,scores['audio'])),
        matched_fusion=classify(late_fuse(scores['image'],scores['audio'])),
        mismatched_fusion=classify(late_fuse(scores['image'],mismatched)),
        contradictory_argmax_rows=int(np.sum(scores['image'].argmax(axis=1)!=mismatched.argmax(axis=1))),
        black_image=classify((blank_z['image']@candidates.T).numpy()),silent_audio=classify((blank_z['audio']@candidates.T).numpy()))
    return dict(loss=float(total_loss(z,torch.tensor(labels))),retrieval=retrieval,perturb=perturb,
        image_rule=top1(np.eye(6)[rule_predict(records,'image')],labels,np.arange(6)),
        audio_rule=top1(np.eye(6)[rule_predict(records,'audio')],labels,np.arange(6)),
        image_text_scores=scores['image'].tolist(),audio_text_scores=scores['audio'].tolist(),labels=labels.tolist())


def save_bundle(path,model,prep,data_hash):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    payload=dict(spec=dict(SPEC),state=state,state_hash=state_hash(state),preprocess=prep,preprocess_hash=prep_hash(prep),data_hash=data_hash)
    torch.save(payload,path)


def load_bundle(path,expected_spec=None):
    b=torch.load(path,map_location='cpu',weights_only=True)
    if set(b)!=set(['spec','state','state_hash','preprocess','preprocess_hash','data_hash']):raise ValueError('bundle keys')
    if b['spec']!=(SPEC if expected_spec is None else expected_spec) or b['preprocess']['spec']!=b['spec']:raise ValueError('bundle spec mismatch')
    if prep_hash(b['preprocess'])!=b['preprocess_hash'] or state_hash(b['state'])!=b['state_hash']:raise ValueError('bundle digest mismatch')
    model=Align(len(b['preprocess']['vocab']));expected=model.state_dict()
    if set(expected)!=set(b['state']):raise ValueError('state keys')
    for k,v in b['state'].items():
        if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all():raise ValueError('state shape/dtype/finite')
    model.load_state_dict(b['state']);model.eval();return model,b['preprocess'],b


def train(out,epochs=120):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(41);torch.set_num_threads(1)
    splits=dataset();prep=fit(splits['train']);data_hash=fingerprint(splits);model=Align(len(prep['vocab']))
    x=tensor_features(splits['train'],prep);labels=torch.tensor([r['label'] for r in splits['train']])
    initial=evaluate(model,splits['test'],prep) # planned random baseline, not a selection criterion
    opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0)
    history=[];best=float('inf');best_epoch=None;best_state=None
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True);loss=total_loss(model(x),labels);loss.backward();opt.step()
        with torch.no_grad():val=total_loss(model(tensor_features(splits['validation'],prep)),torch.tensor([r['label'] for r in splits['validation']]))
        value=float(val);history.append(dict(epoch=epoch,train_loss=float(loss.detach()),validation_loss=value))
        if value<best:best=value;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);result=evaluate(model,splits['test'],prep);save_bundle(out/'alignment.pt',model,prep,data_hash)
    loaded,restored,bundle=load_bundle(out/'alignment.pt');roundtrip=evaluate(loaded,splits['test'],restored)
    if result!=roundtrip:raise AssertionError('bundle inference mismatch')
    return dict(seed=41,epochs=epochs,updates=epochs,learning_rate=.005,tau=.15,parameter_count=sum(v.numel() for v in model.parameters()),
        split_counts={k:len(v) for k,v in splits.items()},data_hash=data_hash,preprocess_hash=prep_hash(prep),state_hash=bundle['state_hash'],
        best_epoch=best_epoch,best_validation_loss=best,initial=initial,test=result,history=history,roundtrip_exact=True,
        bundle_bytes=(out/'alignment.pt').stat().st_size,vocab=prep['vocab'],fit_ids=prep['fit_ids'],
        scope='synthetic six seen concepts; full-batch supervised positive-set contrastive learning; no pretrained CLIP/CLAP/LLM, ASR or visual generation; inference bundle, not exact training resume')
