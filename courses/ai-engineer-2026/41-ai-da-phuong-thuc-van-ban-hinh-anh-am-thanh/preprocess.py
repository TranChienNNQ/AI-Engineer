import re
import unicodedata
import numpy as np
from data import SPEC
from media import patchify,stft_power


def words(text):
    if not isinstance(text,str):raise ValueError('text required')
    return re.findall(r'[^\W\d_]+',unicodedata.normalize('NFC',text).lower())


def bow(text,vocab):
    out=np.zeros(len(vocab),dtype=np.float32)
    for token in words(text):
        if token in vocab:out[vocab.index(token)]+=1
    if out.sum()==0:raise ValueError('no known text tokens')
    return out/out.sum()


def raw_features(records,vocab):
    images=[];audios=[];texts=[]
    for r in records:
        if r['image'].dtype!=np.uint8 or r['image'].shape!=(16,16,3) or r['audio'].shape!=(800,):raise ValueError('record media shape/dtype')
        images.append(patchify(r['image'].astype(np.float64)/255,4).ravel())
        power,_=stft_power(r['audio']);audios.append(np.log1p(power).ravel())
        texts.append(bow(r['text'],vocab))
    return dict(image=np.array(images,dtype=np.float32),audio=np.array(audios,dtype=np.float32),text=np.array(texts,dtype=np.float32))


def fit(train):
    vocab=sorted({token for r in train for token in words(r['text'])});raw=raw_features(train,vocab)
    return dict(spec=dict(SPEC),vocab=vocab,fit_ids=[r['id'] for r in train],
        mean={k:v.mean(axis=0).tolist() for k,v in raw.items() if k!='text'},
        std={k:np.maximum(v.std(axis=0),.05).tolist() for k,v in raw.items() if k!='text'})


def transform(records,prep):
    if prep['spec']!=SPEC:raise ValueError('preprocess spec mismatch')
    raw=raw_features(records,prep['vocab'])
    for k in ['image','audio']:
        mean=np.asarray(prep['mean'][k],dtype=np.float32);std=np.asarray(prep['std'][k],dtype=np.float32)
        if mean.shape!=raw[k].shape[1:] or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):raise ValueError('normalizer mismatch')
        raw[k]=(raw[k]-mean)/std
    return raw
